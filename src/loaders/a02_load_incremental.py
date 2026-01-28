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
def load_incremental_data(incremental_dir):
    files = os.listdir(incremental_dir)
    print(f"Found {len(files)} files in {incremental_dir} to process.")
    for file in files:
        full_file_name = os.path.join(incremental_dir, file)
        print(f".Processing: {full_file_name}...")
        # Read the data
        try:
            # Determine file type and read accordingly
            if file.endswith('.csv'):
                df = pd.read_csv(full_file_name)
            elif file.endswith('.xlsx') or file.endswith('.xls'):
                df = pd.read_excel(full_file_name)
            else:
                print(f"Unsupported file type for {file}. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading {file}: {e}. Skipping.")
            continue
        
        #Preprocess the data before loading to DB
        # Select the required columns
        df = df[cfg_vars.required_columns]
        # Rename the columns
        df = df.rename(columns=cfg_vars.column_mapping)        

        #Add the 3 new columns
        #df['series_start_dt'] = ''
        #df['series_end_dt']  = ''
        #df['series_name'] = series_name

        #Remove other series
        print("Count of records before filtering series:", len(df))
        df = df[df['scty_series'] == 'EQ']
        print("Count of records after filtering series:", len(df))

        df['trade_dt'] = pd.to_datetime(df['trade_dt']).dt.strftime('%Y-%m-%d %H:%M:%S')    
        #Load the incremental Data to DB
        db_path = cfg_vars.db_dir + cfg_vars.db_name
        table_name='historical_stocks'       

        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()

        logger.info(f"Loaded with {len(df)} rows.")         
        logger.info("DataFrame loaded to sqlite DB successfully.")

def daily_to_weekly(daily_data):
    """
    Convert daily stock data to weekly data.
    Returns:
    DataFrame with weekly OHLCV data for "each" stock
    """
    # Ensure trade_dt is datetime
    daily_data['trade_dt'] = pd.to_datetime(daily_data['trade_dt'])
    
    # Group by tckr_symbol
    grouped = daily_data.groupby('tckr_symbol')
    
    weekly_list = []
    for symbol, group in grouped:
        print(f".....{symbol}")
        # Set trade_dt as index
        group = group.set_index('trade_dt').sort_index()
        
        # Resample to weekly (ending on Sunday)
        weekly = group.resample('W').agg({
            'open_price': 'first',
            'high_price': 'max',
            'low_price': 'min',
            'closing_price': 'last',
            'total_trading_volume': 'sum'
        }).dropna()
        
        # Shift index back by 6 days to label with Monday
        weekly.index = weekly.index - pd.Timedelta(days=6)
        
        # Reset index and add symbol
        weekly = weekly.reset_index()
        weekly['tckr_symbol'] = symbol
        
        weekly_list.append(weekly)
    
    # Concatenate all weekly data
    weekly_data = pd.concat(weekly_list, ignore_index=True)
    
    # Rename columns for consistency
    weekly_data.rename(columns={
        'trade_dt': 'trade_dt',
        'open_price': 'open_price',
        'high_price': 'high_price',
        'low_price': 'low_price',
        'closing_price': 'closing_price',
        'total_trading_volume': 'volume'
    }, inplace=True)
    
    return weekly_data

def generate_ema_signals(weekly_data, nifty_list, span_short=20, span_long=50):
    """
    Generate EMA crossover signals for each stock individually.
    """
    processed_list = []
    
    for symbol in nifty_list:
        # Extract records for the stock
        stock_data = weekly_data[weekly_data['tckr_symbol'] == symbol].copy()
        
        if stock_data.empty:
            continue
        
        # Order by tckr_symbol and Date (though tckr_symbol is the same, Date ascending)
        stock_data = stock_data.sort_values(by=['tckr_symbol', 'trade_dt'])
        
        # Calculate EMAs
        stock_data['wema20'] = stock_data['closing_price'].ewm(span=span_short, adjust=False).mean().round(2)
        stock_data['wema50'] = stock_data['closing_price'].ewm(span=span_long, adjust=False).mean().round(2)
        
        # Generate signals
        stock_data['wema-signal'] = np.where(stock_data['wema20'] > stock_data['wema50'], True, False)

        # In generate_ema_signals, after calculating EMAs:
        # Calculate slope of W-EMA20
        stock_data['wema20_slope'] = stock_data['wema20'].diff()

        # Generate signals
        conditions = [
            (stock_data['closing_price'] > stock_data['wema50']) & (stock_data['wema20'] > stock_data['wema50']) & (stock_data['wema20_slope'] > 0),
            (stock_data['closing_price'] < stock_data['wema50']) & (stock_data['wema20'] < stock_data['wema50']) & (stock_data['wema20_slope'] < 0)
        ]
        choices = ['bullish', 'bearish']
        stock_data['wema_signal'] = np.select(conditions, choices, default='')
        
        processed_list.append(stock_data)
    
    # Concatenate all processed stocks
    if processed_list:
        weekly_data = pd.concat(processed_list, ignore_index=True)
    else:
        weekly_data = pd.DataFrame()  # Empty if no data
    
    return weekly_data
    
def read_nse_data(nifty_list,start_date):
    '''
    Read NSE_DATA table from SQLite DB and filter based on start_date and nifty_list
    '''
    db_path = cfg_vars.db_dir + cfg_vars.db_name
    table_name='historical_stocks'       
    query = cfg_vars.nse_data_read_query
    
    all_stocks_df = util_funcs.read_data(db_path, table_name, query) 
    print(f"No of Records read from NSE_DATA: {len(all_stocks_df)}.")
    print(f"Max trade date in NSE_DATA: {max(all_stocks_df['trade_dt'])}")
    # Filter for records after 2021-01-01 and where tckr_symbol is in nifty_list
    all_stocks_df['trade_dt'] = pd.to_datetime(all_stocks_df['trade_dt'], errors='coerce')  # Ensure datetime format
    
    all_stocks_df = all_stocks_df[
        (all_stocks_df['trade_dt'] >= pd.to_datetime(start_date)) & 
        (all_stocks_df['trade_dt'] <= pd.to_datetime(date.today())) & 
        (all_stocks_df['tckr_symbol'].isin(nifty_list))]

    print(f"No of Records post filtering from NSE_DATA: {len(all_stocks_df)}.")
    #all_stocks_df.to_excel('D:/myCourses/shree_dwarkadhish_v3/data/output/results/sdmodel/all_stocks_filtered.xlsx', index=False)
    all_stocks_df['trade_dt'] = all_stocks_df['trade_dt'].dt.strftime('%Y-%m-%d')
    print(min(all_stocks_df['trade_dt']), max(all_stocks_df['trade_dt']))
    return(all_stocks_df)

def main():
    incremental_dir = 'C:/Users/Amit/Downloads/nse_data'
    start_date = '2020-01-01'
    weekly_data_path = cfg_vars.weekly_data_dir + f'stocks_weekly_data.xlsx'
    
    first_load = 'yes'

    ###Step1. Load Incremental Data from all files in the directory to Sqlite DB
    if first_load == 'yes':
        load_incremental_data(incremental_dir)
        first_load = 'no'

    ###Step2. Read Data from Sqlite DB for required stocks and date range
    # Sort the list alphabetically
    nifty_list = sorted(list(set(cfg_nifty.nifty_fno + cfg_nifty.nifty_sectoral + cfg_nifty.nifty_mid_small_caps)))
    stocks_df = read_nse_data(nifty_list, start_date)
    
    # Generate weekly data for all stocks
    weekly_data = daily_to_weekly(stocks_df)
    weekly_data = generate_ema_signals(weekly_data,nifty_list)

    weekly_data['trade_dt'] = pd.to_datetime(weekly_data['trade_dt']).dt.strftime('%Y-%m-%d')
    order_cols = ['tckr_symbol', 'trade_dt', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume', 'wema20', 'wema50', 'wema-signal']
    weekly_data = weekly_data[order_cols]
    
    # Save to Excel
    weekly_data.to_excel(weekly_data_path, index=False)
    print(f"Weekly data saved to {weekly_data_path}")    
    
if __name__ == "__main__":
    main()