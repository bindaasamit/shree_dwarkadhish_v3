#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V5')  # Add project root to path
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta, datetime
import concurrent.futures
import yfinance as yf
import pretty_errors
import traceback

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error


#-------------------------------------------------------------------------------------------------- 
#                         All Weekly Metrics
#--------------------------------------------------------------------------------------------------

## Read Daily OHLC Data
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

## Load Incremental to Historical_Stocks Database
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

## Convert Daily → Weekly (Proper Trading-Week Aggregation)
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
        'trade_dt': 'w_start_date',
        'open_price': 'w_open',
        'high_price': 'w_high',
        'low_price': 'w_low',
        'closing_price': 'w_close',
        'total_trading_volume': 'w_volume'
    }, inplace=True)
    return weekly_data

## Weekly EMA Stack (Trend Structure)
def add_weekly_ema_stack(df):
    df['w_ema10'] = df['w_close'].ewm(span=10, adjust=False).mean()
    df['w_ema20'] = df['w_close'].ewm(span=20, adjust=False).mean()
    df['w_ema50'] = df['w_close'].ewm(span=50, adjust=False).mean()

    df['flag_weekly_trend_stack'] = (
        (df['w_ema10'] > df['w_ema20']) &
        (df['w_ema20'] > df['w_ema50'])
    )
    return df

## Weekly Range Position (MOST IMPORTANT FILTER) (Detects distribution / range breakouts)
def add_weekly_range_position(df, lookback_weeks=26):
    df['w_high_26'] = df['w_high'].rolling(lookback_weeks).max()
    df['w_low_26']  = df['w_low'].rolling(lookback_weeks).min()

    df['weekly_range_pos'] = (
        (df['w_close'] - df['w_low_26']) /
        (df['w_high_26'] - df['w_low_26'])
    )

    df['flag_weekly_range_strength'] = df['weekly_range_pos'] > 0.75

    return df

## Weekly ATR Compression / Expansion Filter (Identifies fuel vs exhaustion)
def add_weekly_atr_filter(df, atr_period=14):
    high = df['w_high']
    low  = df['w_low']
    close = df['w_close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df['w_atr'] = tr.rolling(atr_period).mean()
    df['w_atr_ratio'] = df['w_atr'] / df['w_close']

    df['w_atr_median_26'] = df['w_atr_ratio'].rolling(26).median()

    df['flag_weekly_atr_ok'] = df['w_atr_ratio'] < df['w_atr_median_26']

    return df

## Weekly BBW Compression (OPTIONAL but Powerful). Use only weekly, not daily.
def add_weekly_bbw_filter(df, bb_period=20):
    ma = df['w_close'].rolling(bb_period).mean()
    std = df['w_close'].rolling(bb_period).std()

    bb_upper = ma + 2 * std
    bb_lower = ma - 2 * std

    df['w_bbw'] = (bb_upper - bb_lower) / ma
    df['w_bbw_pct_25'] = df['w_bbw'].rolling(30).quantile(0.25)

    df['flag_weekly_bbw_compression'] = df['w_bbw'] < df['w_bbw_pct_25']
    return df

## Generate a Weekly Score
def add_weekly_structure_score(df):
    df['weekly_structure_score'] = (
        df['flag_weekly_range_strength'].astype(int) * 3 +
        df['flag_weekly_trend_stack'].astype(int) * 2 +
        df['flag_weekly_atr_ok'].astype(int) * 1 +
        df.get('flag_weekly_bbw_compression', False).astype(int) * 1
    )


    ## Finaly Cosmetics
    df['w_start_date'] = pd.to_datetime(df['w_start_date']).dt.strftime('%Y-%m-%d')

    order_cols = ['tckr_symbol', 'w_start_date', 'flag_weekly_trend_stack', 'flag_weekly_range_strength',
    'flag_weekly_atr_ok', 'flag_weekly_bbw_compression', 'weekly_structure_score',
    'w_open', 'w_high', 'w_low', 'w_close', 'w_volume',
    'w_ema10', 'w_ema20', 'w_ema50', 'w_high_26', 'w_low_26', 'weekly_range_pos',
    'w_atr', 'w_atr_ratio', 'w_atr_median_26', 'w_bbw', 'w_bbw_pct_25']
    df = df[order_cols]


    return df
