#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V3')  # Add project root to path
import pandas as pd
import pandas_ta as ta
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
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs, util_bear, util_bull

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

rp_path = cfg_vars.rpmodel_dir + 'f2_rp_model_metrics.xlsx'

#------------------------------------------------------------------------------
###            Main Workflow
#------------------------------------------------------------------------------
def read_nse_data(nifty_list,start_date):
    '''
    Read NSE_DATA table from SQLite DB and filter based on start_date and nifty_list
    '''
    db_path = cfg_vars.db_dir + cfg_vars.db_name
    table_name='nse_data'       
    query = cfg_vars.nse_data_read_query
    
    all_stocks_df = util_funcs.read_data(db_path, table_name, query) 
    print(f"No of Records read from NSE_DATA: {len(all_stocks_df)}.")
 
    # Filter for records after 2021-01-01 and where tckr_symbol is in nifty_list
    all_stocks_df['trade_dt'] = pd.to_datetime(all_stocks_df['trade_dt'], errors='coerce')  # Ensure datetime format
    all_stocks_df = all_stocks_df[
        (all_stocks_df['trade_dt'] >= pd.to_datetime(start_date)) & 
        (all_stocks_df['trade_dt'] <= pd.to_datetime(date.today())) & 
        (all_stocks_df['tckr_symbol'].isin(nifty_list))]
    print(f"No of Records post 2001-01-01 and in nifty_list from NSE_DATA: {len(all_stocks_df)}.")
    #all_stocks_df.to_excel('D:/myCourses/shree_dwarkadhish_v3/data/output/results/sdmodel/all_stocks_filtered.xlsx', index=False)
    all_stocks_df['trade_dt'] = all_stocks_df['trade_dt'].dt.strftime('%Y-%m-%d')
    print(min(all_stocks_df['trade_dt']), max(all_stocks_df['trade_dt']))
    return(all_stocks_df)
    
def main():
    sector ='test'
    start_date = '2024-01-01'
    bull_or_bear_flag = 'bull'  # 'bull' or 'bear'
    bull_no = 1
    bear_no = 1
    
    if sector =='test':
        nifty_list = cfg_nifty.nifty_test_list
    elif sector =='fno':   
        nifty_list = cfg_nifty.nifty_fno_list
    else:
        nifty_list = cfg_nifty.nifty_cash_list


    print("...........Step1. Read all Stocks Data NSE_DATA................")
    stocks_df = read_nse_data(nifty_list,start_date)   

    print("...........Step2. Identify Bullish Trades......................")
    bullish_groups = []
    for symbol, group in stocks_df.groupby('tckr_symbol'):
        # Prepare df_daily for the stock
        print("                       ")
        print("==> Processiong Symbol:", symbol)
        df_daily = group[['trade_dt', 'tckr_symbol', 'open_price', 'high_price', 'low_price', 'closing_price', 'total_trading_volume']].copy()
        df_daily = df_daily.set_index('trade_dt').rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'closing_price': 'close',
            'total_trading_volume': 'volume'
        })
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily = df_daily.sort_index()
        
        # Invoke scan_bull1
        group_df=util_bull.scan_bull1(df_daily)
        bullish_groups.append(group_df.copy())
    
    # Concatenate the groups back into stocks_df
    stocks_bull1_df = pd.concat(bullish_groups).reset_index(drop=True)
    
    #Order the Columns
    order_cols = ['tckr_symbol','trade_dt', 'open', 'high', 'low', 'close', 'volume',
                  'rsi_14', 'sma_20', 'pivot', 'weekly_rsi_14', 'weekly_rsi_4', 'bull1_flag']
    stocks_bull1_df = stocks_bull1_df[order_cols]
    
    # Remove records where tckr_symbol is empty or has spaces
    stocks_bull1_df = stocks_bull1_df[stocks_bull1_df['tckr_symbol'].notna() & (stocks_bull1_df['tckr_symbol'].str.strip() != '')]

    stocks_bull1_df.to_excel(rp_path, index=False)
    print("Bull1 Bullish Processing is Done!")
    #print(stocks_bull1_df.head())
    
if __name__ == "__main__":
    main()