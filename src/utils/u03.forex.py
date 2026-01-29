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

forex_file = 'C:/Users/Amit/Downloads/Abhishek/Forex_Sample.xlsx'


#read the above forex file
def load_forex_data(forex_file):
    try:
        df = pd.read_excel(forex_file)
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

       # Reorder columns
    order_cols = ['date_gmt', 'date_est', 'date_ist', 'daylight_savings', 'open', 'low', 'high','close', 
    'volume',]
    #,'entry_price','target_price','stop_loss_price','success_flag']
    df = df[order_cols]
    return df

def parse_forex_records(df):
    # Add new columns
    df['new_batch_start'] = None
    df['direction'] = None
    df['LQP'] = None
    df['HQP'] = None
    df['entry_price'] = None
    df['target_price'] = None
    df['stop_loss_price'] = None
    df['comments'] = None
    df['success_flag'] = None
    df['risk_reward'] = None

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
        if (row['daylight_savings'] == 'off') and (row['date_ist'].hour == 13) and (row['date_ist'].minute == 30):
            # Start new batch
            batch_close = row['close']
            start_date = row['date_ist'].date()
            df.at[idx, 'new_batch_start'] = start_date
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
            df.at[idx, 'new_batch_start'] = None
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
            df.at[idx, 'new_batch_start'] = start_date
            if entry_price is None:
                if direction_batch == 'bearish':
                    if LQP < row['low'] and row['high'] < HQP:
                        pass  # Move to next record
                    elif row['low'] < LQP and row['high'] < HQP:
                        entry_price = LQP
                        target_price = entry_price - 0.250
                        stop_loss_price = entry_price + 0.250
                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif LQP < row['low'] and HQP < row['high']:
                        entry_price = HQP
                        target_price = entry_price - 0.250
                        stop_loss_price = entry_price + 0.250
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
                        target_price = entry_price + 0.250
                        stop_loss_price = entry_price - 0.250
                        df.at[idx, 'entry_price'] = entry_price
                        df.at[idx, 'target_price'] = target_price
                        df.at[idx, 'stop_loss_price'] = stop_loss_price
                    elif LQP < row['low'] and HQP < row['high']:
                        entry_price = HQP
                        target_price = entry_price + 0.250
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
                        df.at[idx, 'risk_reward'] = 1
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
                        df.at[idx, 'risk_reward'] = 1
                        batch_completed = True
                    elif row['low'] < stop_loss_price and target_price < row['high']:
                        pass

    # Filter to keep only rows where new_batch_start is not empty
    df = df[df['new_batch_start'].notna()]

    # Update order_cols to include new columns
    order_cols = ['date_gmt', 'date_est', 'date_ist', 'daylight_savings', 
                'open', 'low', 'high', 'close', 'volume', 
                'new_batch_start', 'direction', 'LQP', 'HQP', 
                'entry_price', 'target_price', 'stop_loss_price', 'comments', 
                'success_flag', 'risk_reward']
    df = df[order_cols]
    return df

def main():
    #Step1. Read the data & Apply the transformations
    forex_df = load_forex_data(forex_file)
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

    forex_results_df.to_excel('C:/Users/Amit/Downloads/Abhishek/Forex_Transformed1.xlsx', index=False)

# ============================================================================
#                                MAIN WORKFLOW
# ============================================================================
if __name__ == "__main__":   
    main()


