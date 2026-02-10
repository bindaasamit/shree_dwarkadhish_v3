#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V5')  # Add project root to path
import pandas as pd
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta
import concurrent.futures
import yfinance as yf
import pretty_errors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger
#------------------------------------------------------------------------------
###            Import Modules
#------------------------------------------------------------------------------
from config import cfg_nifty 
from config import cfg_vars 
from src.utils import util_funcs


logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error
#------------------------------------------------------------------------------
###            Get Historical Data from yfinance and transform to required format
#------------------------------------------------------------------------------

forex_file = 'C:/Users/Amit/Downloads/Abhishek/2019_2026 GBPJPY_M15.csv'

def read_forex_data():
    '''
    Read NSE_DATA table from SQLite DB and filter based on start_date and nifty_list
    '''

    db_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/db/'
    db_name = "shree_dwarkadhish.db"
    db_path = db_dir + db_name
    table_name ='Forex'       
    query = "SELECT Date_GMT, Open, High, Low, Close, Volume FROM Forex"
    
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    forex_df = pd.read_sql_query(query, conn)
    
    print(f"No of Records read from Forex: {len(forex_df)}")
    print(min(forex_df['Date_GMT']), max(forex_df['Date_GMT']))
       
    return(forex_df)

#read the above forex file
def load_forex_data(forex_file):
    try:
        df = pd.read_csv(forex_file)
        logger.info(f"Successfully read forex data from {forex_file}")
    except Exception as e:
        logger.error(f"Error reading & transforming forex file {forex_file}: {e}")
        return None
    return df   

# Function to generate lower_QP and higher_QP
def generate_lower_higher_QP(no):
    step = 0.250
    lower_QP = (no // step) * step
    higher_QP = lower_QP + step
    #double_HQP = 2 * higher_QP

    return lower_QP, higher_QP    

def transform_forex_data(df):
    # Rename the columns
    forex_column_mapping = {
        'Date_GMT': 'date_gmt',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=forex_column_mapping)

    #-----------------------------------------------
    ### Apply Daylight Savings Logic
    #-----------------------------------------------
    ### Convert date_gmt to datetime and then generate est and ist times
    df['date_gmt'] = pd.to_datetime(df['date_gmt'])
    # Convert GMT to EST and IST
    df['date_est'] = df['date_gmt'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    #### Create daylight_savings column based on EST DST
    df['daylight_savings'] = df['date_est'].apply(lambda x: 'on' if x.utcoffset() == pd.Timedelta(hours=-4) else 'off')
    df['date_ist'] = df['date_gmt'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    # Extract year, month, week from date_ist
    df['year'] = df['date_ist'].dt.year
    df['month'] = df['date_ist'].dt.month
    df['day_of_week'] = df['date_ist'].dt.day_name()  # Changed from week        

    # Add candle_flag column
    df['candle_flag'] = df.apply(lambda row: 'bullish' if row['open'] < row['close'] else 'bearish', axis=1)

       # Reorder columns
    order_cols = ['date_gmt', 'date_est', 'date_ist', 'year', 'month', 'day_of_week', 'daylight_savings', 'open', 'low', 'high','close', 
    'volume','candle_flag']
    #,'entry_price','target_price','stop_loss_price','success_flag']
    df = df[order_cols]
    return df

def parse_forex_records(df):
    # Add new columns
    df['group_date'] = None
    df['direction'] = None
    df['LQP'] = None
    df['HQP'] = None
    df['entry_price'] = None
    df['target_price'] = None
    df['stop_loss_price'] = None
    df['comments'] = None
    df['success_flag'] = None
    df['risk_reward'] = None
    df['volume_signal'] = None  # New column
    rr2_flag = "no"  # Set to "no" for RR 1:1, "yes" for RR 1:2

    # Initialize batch variables
    batch_close = None
    LQP = None
    HQP = None
    entry_price = None
    target_price = None
    stop_loss_price = None
    direction_batch = None
    batch_completed = False
    batch_start_idx = None
    start_date = None

    for idx, row in df.iterrows():
        #--------------------------------------------------------
        # Check for new batch start
        #--------------------------------------------------------
        if ((row['daylight_savings'] == 'off') and (row['date_ist'].hour == 13) and (row['date_ist'].minute == 30) or
            (row['daylight_savings'] == 'on') and (row['date_ist'].hour == 12) and (row['date_ist'].minute == 30)):
            # Start new batch
            batch_close = row['close']
            start_date = row['date_ist'].date()
            df.at[idx, 'group_date'] = start_date
            if row['close'] > row['open']:
                direction_batch = 'bullish'
            elif row['close'] < row['open']:
                direction_batch = 'bearish'
            else:
                direction_batch = 'neutral'
            df.at[idx, 'direction'] = direction_batch
            LQP, HQP = generate_lower_higher_QP(row['close'])
            df.at[idx, 'LQP'] = LQP
            df.at[idx, 'HQP'] = HQP
            entry_price = None  # Reset for new batch
            target_price = None
            stop_loss_price = None
            batch_completed = False
            batch_start_idx = idx  # Add this
        else:
            df.at[idx, 'group_date'] = None
            df.at[idx, 'direction'] = None
            df.at[idx, 'LQP'] = None
            df.at[idx, 'HQP'] = None
        #-------------------------------------------------------------------------------------------
        # Start Parsing from next row onwards
        #-------------------------------------------------------------------------------------------
        if LQP is not None and idx > batch_start_idx and not batch_completed:
            #--------------------------------------------------------------------
            # Set entry_price, target_price, stop_loss_price
            #--------------------------------------------------------------------
            df.at[idx, 'group_date'] = start_date
            if entry_price is None:
                if direction_batch == 'bearish':
                    if LQP < row['low'] and row['high'] < HQP:
                        pass  # Move to next record
                    elif row['low'] < LQP and row['high'] < HQP:
                        entry_price = LQP
                        if rr2_flag == "no":
                            target_price = entry_price - 0.250
                        else:
                            target_price = entry_price - 2*(0.250)
                        stop_loss_price = entry_price + 0.250
                        # Set volume_signal
                        max_vol_prior = df.loc[batch_start_idx:idx-1, 'volume'].max() if idx > batch_start_idx else 0
                        if row['volume'] > max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'increasing'
                        elif row['volume'] < max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'decreasing'
                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif LQP < row['low'] and HQP < row['high']:
                        entry_price = HQP
                        # Set volume_signal
                        max_vol_prior = df.loc[batch_start_idx:idx-1, 'volume'].max() if idx > batch_start_idx else 0
                        if row['volume'] > max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'increasing'
                        elif row['volume'] < max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'decreasing'

                        if df.at[idx, 'volume_signal'] == 'decreasing':
                            if rr2_flag == "no":
                                target_price = entry_price - 0.250
                            else:
                                target_price = entry_price - 2*(0.250)
                            stop_loss_price = entry_price + 0.250
                        else:
                            if rr2_flag == "no":
                                target_price = entry_price + 0.250
                            else:
                                target_price = entry_price + 2*(0.250)
                            stop_loss_price = entry_price - 0.250
                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif row['low'] < LQP and HQP < row['high']:
                        df.at[idx, 'comments'] = 'both_breached'
                elif direction_batch in ['bullish', 'neutral']:
                    if LQP < row['low'] and row['high'] < HQP:
                        pass  # Move to next record
                    elif row['low'] < LQP and row['high'] < HQP:
                        entry_price = LQP
                        # Set volume_signal
                        max_vol_prior = df.loc[batch_start_idx:idx-1, 'volume'].max() if idx > batch_start_idx else 0
                        if row['volume'] > max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'increasing'
                        elif row['volume'] < max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'decreasing'

                        if df.at[idx, 'volume_signal'] == 'increasing':
                            if rr2_flag == "no":
                                target_price = entry_price + 0.250
                            else:
                                target_price = entry_price + 2*(0.250)
                            stop_loss_price = entry_price - 0.250
                        else:
                            if rr2_flag == "no":
                                target_price = entry_price - 0.250
                            else:
                                target_price = entry_price - 2*(0.250)
                            stop_loss_price = entry_price + 0.250

                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif LQP < row['low'] and HQP < row['high']:
                        entry_price = HQP
                        
                        # Set volume_signal
                        max_vol_prior = df.loc[batch_start_idx:idx-1, 'volume'].max() if idx > batch_start_idx else 0
                        if row['volume'] > max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'increasing'
                        elif row['volume'] < max_vol_prior:
                            df.at[idx, 'volume_signal'] = 'decreasing'
                        if rr2_flag == "no":
                            target_price = entry_price + 0.250
                        else:
                            target_price = entry_price + 2*(0.250)
                        stop_loss_price = entry_price - 0.250
                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif row['low'] < LQP and HQP < row['high']:
                        df.at[idx, 'comments'] = 'both_breached'

            #--------------------------------------------------------------------
            # Set success_flag, risk_reward
            #--------------------------------------------------------------------
            # Do it only if <entry/target/stop_loss>_price is set
            if entry_price is not None:
                if direction_batch == 'bearish':
                    if target_price < row['low'] and row['high'] < stop_loss_price:
                        pass
                    elif row['low'] < target_price and row['high'] < stop_loss_price:
                        df.at[idx, 'success_flag'] = 'pass'
                        if rr2_flag == "no":
                            df.at[idx, 'risk_reward'] = 1
                        else:
                            df.at[idx, 'risk_reward'] = 2
                        batch_completed = True
                    elif target_price < row['low'] and stop_loss_price < row['high']:
                        df.at[idx, 'success_flag'] = 'fail'
                        df.at[idx, 'risk_reward'] = -1
                        batch_completed = True
                    elif row['low'] < target_price and stop_loss_price < row['high']:
                        pass
                else:
                    if stop_loss_price < row['low'] and row['high'] < target_price:
                        pass
                    elif row['low'] < stop_loss_price and row['high'] < target_price:
                        df.at[idx, 'success_flag'] = 'fail'
                        df.at[idx, 'risk_reward'] = -1
                        batch_completed = True
                    elif stop_loss_price < row['low'] and target_price < row['high']:
                        df.at[idx, 'success_flag'] = 'pass'
                        if rr2_flag == "no":
                            df.at[idx, 'risk_reward'] = 1
                        else:
                            df.at[idx, 'risk_reward'] = 2
                        batch_completed = True
                    elif row['low'] < stop_loss_price and target_price < row['high']:
                        pass

    # Check if any comments == 'both_breached' and update success row
    if (df['comments'] == 'both_breached').any():
        success_mask = df['success_flag'].notna()
        if success_mask.any():
            success_idx = df[success_mask].index[0]
            df.at[success_idx, 'comments'] = 'check_group'
            
    # Filter to keep only rows where group_date is not empty
    df = df[df['group_date'].notna()]

    # Update order_cols to include new columns
    order_cols = ['date_gmt', 'date_est', 'date_ist', 'daylight_savings', 'year', 'month', 'day_of_week',
                'open', 'low', 'high', 'close', 'volume', 'candle_flag',
                'group_date', 'direction', 'LQP', 'HQP', 
                'entry_price', 'target_price', 'stop_loss_price', 'volume_signal',
                'success_flag', 'risk_reward','comments']
    df = df[order_cols]
    return df

def main():
    #Step1. Read the data & Apply the transformations
    read_file_flag = 'True'
    if read_file_flag == 'True':
        forex_df = load_forex_data(forex_file)
    else:
        forex_df = read_forex_data()
    forex_df = transform_forex_data(forex_df.copy())
    
    # Apply transformations and parsing to each group, accumulate results
    all_results = []   
    # Group by day and filter groups where first record has time 13:30 or 12:30
    forex_df['day'] = forex_df['date_ist'].dt.date
    grouped = forex_df.groupby('day')
    for name, group in grouped:
        # Parse through the records one by one and generate the required columns
        group_df = parse_forex_records(group)
        all_results.append(group_df)

    if all_results:
        forex_results_df = pd.concat(all_results, ignore_index=True)
    else:
        forex_results_df = pd.DataFrame()

    # Make timezone-aware columns naive for Excel compatibility
    if not forex_results_df.empty:
        forex_results_df['date_est'] = forex_results_df['date_est'].dt.tz_localize(None)
        forex_results_df['date_ist'] = forex_results_df['date_ist'].dt.tz_localize(None)

    forex_results_df.to_excel('C:/Users/Amit/Downloads/Abhishek/RR2_Forex_Historical_Results.xlsx', index=False)

    # Create summary DataFrame: one row per group_date
    if not forex_results_df.empty:
        summary_df = forex_results_df.dropna(subset=['group_date']).groupby('group_date').apply(lambda g: pd.Series({
            'day_of_week': g['day_of_week'].dropna().iloc[0] if not g['day_of_week'].dropna().empty else None,  # Added
            'direction': g['direction'].dropna().iloc[0] if not g['direction'].dropna().empty else None,
            'LQP': g['LQP'].dropna().iloc[0] if not g['LQP'].dropna().empty else None,
            'HQP': g['HQP'].dropna().iloc[0] if not g['HQP'].dropna().empty else None,
            'entry_price': g['entry_price'].dropna().iloc[0] if not g['entry_price'].dropna().empty else None,
            'target_price': g['target_price'].dropna().iloc[0] if not g['target_price'].dropna().empty else None,
            'stop_loss_price': g['stop_loss_price'].dropna().iloc[0] if not g['stop_loss_price'].dropna().empty else None,
            'volume_signal': g['volume_signal'].dropna().iloc[0] if not g['volume_signal'].dropna().empty else None,
            'success_flag': g['success_flag'].dropna().iloc[0] if not g['success_flag'].dropna().empty else None,
            'risk_reward': g['risk_reward'].dropna().iloc[0] if not g['risk_reward'].dropna().empty else None,
            'comments': g['comments'].dropna().iloc[0] if not g['comments'].dropna().empty else None,
        })).reset_index()

        summary_df.to_excel('C:/Users/Amit/Downloads/Abhishek/RR2_Forex_Summary_Results.xlsx', index=False)
        print(f"Summary results saved to RR2_Forex_Summary_Results.xlsx with {len(summary_df)} rows.")
    else:
        print("No data to summarize.")

# ============================================================================
#                                MAIN WORKFLOW
# ============================================================================
if __name__ == "__main__":   
    main()


