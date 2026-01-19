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
from src.classes.scanner.cl_Swing_Strategy import SwingTradingStrategyV2

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error


#------------------------------------------------------------------------------

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
        'trade_dt': 'Date',
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'closing_price': 'Close',
        'total_trading_volume': 'Volume'
    }, inplace=True)
    
    return weekly_data


# Example usage
if __name__ == "__main__":
    sector = 'test'  # Define sector
    start_date = '2024-01-01'
    
    """
    match sector: 
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'movers': nifty_list = cfg_nifty.nifty_movers
        case 'fno_sects': nifty_list = list(set(cfg_nifty.nifty_fno + cfg_nifty.nifty_sectoral))
        case 'cash': nifty_list = cfg_nifty.nifty_cash  
        case _: print("Invalid Sector")
    """
    # Define nifty_list based on sector
    nse_list = cfg_nifty.nifty_movers + \
                cfg_nifty.nifty_fno + \
                cfg_nifty.nifty_sectoral + \
                cfg_nifty.nifty_mid_small_caps
    nifty_list = list(set(nse_list))  # Remove duplicates
    
    swing_path1 = cfg_vars.swing_model_dir1 + f'{sector}_weekly_data.xlsx'

    stocks_df = read_nse_data(nifty_list, start_date)
    
    # Generate weekly data for all stocks
    weekly_data = daily_to_weekly(stocks_df)

    weekly_data['Date'] = pd.to_datetime(weekly_data['Date']).dt.strftime('%Y-%m-%d')
    order_cols = ['tckr_symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    weekly_data = weekly_data[order_cols]
    
    # Save to Excel
    weekly_data.to_excel(swing_path1, index=False)
    print(f"Weekly data saved to {swing_path1}")

