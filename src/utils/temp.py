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

def read_nse_data(nifty_list):
    #Read NSE_DATA table from SQLite DB and filter based on start_date and nifty_list
    db_path = cfg_vars.db_dir + cfg_vars.db_name
    table_name='nse_data'       
    query = cfg_vars.nse_data_read_query
    all_stocks_df = util_funcs.read_data(db_path, table_name, query) 
    
    # Filter for records after 2021-01-01 and where tckr_symbol is in nifty_list
    all_stocks_df['trade_dt'] = pd.to_datetime(all_stocks_df['trade_dt'], errors='coerce')  # Ensure datetime format
    all_stocks_df = all_stocks_df[
                (all_stocks_df['tckr_symbol'].isin(nifty_list))]
    all_stocks_df['trade_dt'] = pd.to_datetime(all_stocks_df['trade_dt'], format='%Y-%m-%d')
    print(min(all_stocks_df['trade_dt']), max(all_stocks_df['trade_dt']))
    print(f"No of Records read from NSE_DATA: {len(all_stocks_df)}.")
    print("Date Range in all_stocks_df:", min(all_stocks_df['trade_dt']), max(all_stocks_df['trade_dt']))
    return(all_stocks_df)

def get_volatility_master(volatility_path):
    ### Get Volatility Series Master
    series_master_df = pd.read_excel(volatility_path, sheet_name='Sheet1',index_col=None)
    series_master_df['series_start_dt'] = pd.to_datetime(series_master_df['series_start_dt'], format='%Y-%m-%d')
    series_master_df['series_end_dt'] = pd.to_datetime(series_master_df['series_end_dt'], format='%Y-%m-%d')
    return series_master_df

def get_nifty_list(sector):
    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'movers': nifty_list = cfg_nifty.nifty_movers
        case 'fno_sects': nifty_list = list(set(cfg_nifty.nifty_fno + cfg_nifty.nifty_sectoral))
        case 'cash': nifty_list = cfg_nifty.nifty_cash  
        case _: print("Invalid Sector")
    return(sorted(nifty_list))

def get_up_percentile_moves(moves):
    """Compute percentiles for up moves."""
    if len(moves) == 0:
        return [np.nan] * 5
    return np.percentile(moves, [50, 60, 70, 80, 90])

def get_down_percentile_moves(moves):
    """Compute percentiles for down moves."""
    if len(moves) == 0:
        return [np.nan] * 5
    return np.percentile(moves, [50, 60, 70, 80, 90])

def extract_data_details(all_stocks_df, series_master_df, nifty_list):
    # Initialize a list to collect results
    results = []
    
    # For each stock from nifty_list
    for stock in nifty_list:
        print(f"...{stock}")
        # For each series from series_master_df
        for idx, series_row in series_master_df.iterrows():
            #print(f"...{series_row['series_name'] if 'series_name' in series_row else idx}")
            start_dt = series_row['series_start_dt']
            end_dt = series_row['series_end_dt']
            
            # Get the data from all_stocks_df (filter for stock and date range)
            filtered_df = all_stocks_df[
                (all_stocks_df['tckr_symbol'] == stock) &
                (all_stocks_df['trade_dt'] >= start_dt) &
                (all_stocks_df['trade_dt'] <= end_dt)
            ]
            
            if not filtered_df.empty:
                # Sort by trade_dt to ensure chronological order
                filtered_df = filtered_df.sort_values('trade_dt')
                
                # Capture values
                series_open = filtered_df.iloc[0]['open_price']  # First trading day open
                series_high = filtered_df['high_price'].max()   # Highest high of the period
                series_low = filtered_df['low_price'].min()     # Lowest low of the period
                up_move_pct = (series_high - series_open)/series_open * 100      # series_high - series_open
                down_move_pct = (series_open - series_low)/series_open * 100       # series_open - series_low
                
                # Prepare row to append (include series_name if it exists in series_master_df)
                result_row = {
                    'symbol': stock,
                    'series_start_dt': start_dt,
                    'series_end_dt': end_dt,
                    'series_open': series_open,
                    'series_high': series_high,
                    'series_low': series_low,
                    'up_move_pct': up_move_pct,
                    'down_move_pct': down_move_pct
                }
                # Optionally add series_name if present
                if 'series_name' in series_row:
                    result_row['series_name'] = series_row['series_name']
                results.append(result_row)
    # Create the DataFrame
    stock_hist_series_volatility_df = pd.DataFrame(results)
    return stock_hist_series_volatility_df

def get_percentile_moves(stock_hist_series_volatility_df,nifty_list):
    percentile_results = []
    
    for stock in nifty_list:
        # Filter stock_hist_series_volatility_df on each stock
        df = stock_hist_series_volatility_df[stock_hist_series_volatility_df['symbol'] == stock]
        
        # Up moves
        up_move_pct = df['up_move_pct'].dropna().values
        up_50, up_60, up_70, up_80, up_90 = get_up_percentile_moves(up_move_pct)
        
        # Down moves
        down_move_pct = df['down_move_pct'].dropna().values
        down_50, down_60, down_70, down_80, down_90 = get_down_percentile_moves(down_move_pct)
        
        # Add to results
        percentile_results.append({
            'tckr_symbol': stock,
            'down_50': round(down_50,1),
            'down_60': round(down_60,1),
            'down_70': round(down_70,1),
            'down_80': round(down_80,1),
            'down_90': round(down_90,1),
            'up_50': round(up_50,1),
            'up_60': round(up_60,1),
            'up_70': round(up_70,1),
            'up_80': round(up_80,1),
            'up_90': round(up_90,1)
        })
    
    # Create the percentile DataFrame
    stock_percentile_df = pd.DataFrame(percentile_results)
    return stock_percentile_df

def get_series_spread(current_series_df, stock_percentile_df):
    merged_df = pd.merge(current_series_df, stock_percentile_df, on='tckr_symbol', how='inner')
    merged_df['price_d50'] = merged_df['open_price'] * (1-merged_df['down_50'] / 100)
    merged_df['price_d60'] = merged_df['open_price'] * (1-merged_df['down_60'] / 100)
    merged_df['price_d70'] = merged_df['open_price'] * (1-merged_df['down_70'] / 100)
    merged_df['price_d80'] = merged_df['open_price'] * (1-merged_df['down_80'] / 100)
    merged_df['price_d90'] = merged_df['open_price'] * (1-merged_df['down_90'] / 100)

    merged_df['price_u50'] = merged_df['open_price'] * (1+merged_df['up_50'] / 100) 
    merged_df['price_u60'] = merged_df['open_price'] * (1+merged_df['up_60'] / 100)
    merged_df['price_u70'] = merged_df['open_price'] * (1+merged_df['up_70'] / 100)
    merged_df['price_u80'] = merged_df['open_price'] * (1+merged_df['up_80'] / 100)
    merged_df['price_u90'] = merged_df['open_price'] * (1+merged_df['up_90'] / 100) 

    order_cols = ['tckr_symbol', 'trade_dt', 'open_price', 'price_d50', 'price_d60', 'price_d70',
            	'price_d80', 'price_d90', 'price_u50', 'price_u60', 'price_u70', 
                'price_u80', 'price_u90', 'down_50', 'down_60', 'down_70',	
                'down_80', 'down_90', 'up_50', 'up_60',	'up_70', 'up_80', 'up_90']
    merged_df = merged_df[order_cols]
    return merged_df

def calculate_timing_and_payoffs(all_stocks_df,curr_series_open_spread_df,nifty_list):
    print (".............Ensure todays records are added to the database NSE_DATA before running this step..........")
    current_dt = pd.to_datetime(date.today(), format='%Y-%m-%d')
    curr_series_close_df = all_stocks_df[
        (all_stocks_df['trade_dt'] == current_dt) &
        (all_stocks_df['tckr_symbol'].isin(nifty_list))
    ][['tckr_symbol', 'trade_dt', 'closing_price']]
    print("No of records for current date:", len(curr_series_close_df))

    #Join with curr_series_open_spread_df to get open_price and price levels
    curr_series_close_spread_df = pd.merge(curr_series_close_df, curr_series_open_spread_df, on='tckr_symbol', how='inner')
    curr_series_close_spread_df.rename(columns={'closing_price':'close'}, inplace=True)
    
    ### For Long Positions
    # Add Progress Ratios
    curr_series_close_spread_df["progress_u80"] = ( curr_series_close_spread_df["close"] - curr_series_close_spread_df["open_price"]) / ( curr_series_close_spread_df["price_u80"] - curr_series_close_spread_df["open_price"])
    # Add room left (%)  - This single column decides: Options vs cash/Trade vs skip
    curr_series_close_spread_df["room_left_u80_pct"] = ( curr_series_close_spread_df["price_u80"] - curr_series_close_spread_df["close"]) / curr_series_close_spread_df["close"] * 100
        
    ### For Short Positions
    # Add Progress Ratios
    curr_series_close_spread_df["progress_d80"] = ( curr_series_close_spread_df["open_price"] - curr_series_close_spread_df["close"]) / ( curr_series_close_spread_df["open_price"] - curr_series_close_spread_df["price_d80"])
    # Add room left (%)  - This single column decides: Options vs cash/Trade vs skip
    curr_series_close_spread_df["room_left_d80_pct"] = ( curr_series_close_spread_df["close"] - curr_series_close_spread_df["price_d80"]) / curr_series_close_spread_df["close"] * 100

    order_cols = ['tckr_symbol', 'trade_dt_x', 
                'progress_u80', 'room_left_u80_pct', 
                'price_d50', 'price_d60', 'price_d70', 'price_d80', 'price_d90', 
                'up_50', 'up_60', 'up_70', 'up_80', 'up_90',
                'progress_d80', 'room_left_d80_pct',
                'price_u50', 'price_u60', 'price_u70', 'price_u80', 'price_u90',
                'down_50', 'down_60', 'down_70', 'down_80', 'down_90',
                'close', 'trade_dt_y', 'open_price']                  
    curr_series_close_spread_df = curr_series_close_spread_df[order_cols]
    curr_series_close_spread_df.drop(columns=['trade_dt_y'], inplace=True)
    curr_series_close_spread_df.rename(columns={'trade_dt_x': 'trade_dt'}, inplace=True)
    columns_to_round = ['progress_u80', 'room_left_u80_pct', 'progress_d80', 'room_left_d80_pct']
    for col in columns_to_round:
        curr_series_close_spread_df[col] = np.ceil(curr_series_close_spread_df[col] * 100) / 100
    return curr_series_close_spread_df


#----------------------------------------------------------------------------------------------------------------------------
#                                           Main Workflow
#-----------------------------------------------------------------------------------------------------------------------------
#We are trying to do the below - answer the questions
#                       1. Is a direction active?
#                       2. Is the move too late?
#                       3. Is the payoff worth it?
#                       4. What instrument should I use?
#-----------------------------------------------------------------------------------------------------------------------------

def main():
    ### Step1. Get Nifty List
    sector ='fno_sects'  # 'test' / 'movers' / 'fno_sects' / 'cash'
    nifty_list = get_nifty_list(sector)

    ###Step2. Get Volatility Series Master
    volatility_path = 'D:/myCourses/shree_dwarkadhish_v5/data/input/master_series_expiry_daterange_summary.xlsx'
    series_master_df = get_volatility_master(volatility_path)

    ###Step3. Read all NSE Data from SQLite DB
    all_stocks_df = read_nse_data(nifty_list)
    
    ###Step4. Extract the details
    stock_hist_series_volatility_df = extract_data_details(all_stocks_df, series_master_df, nifty_list)

    ###Step5. Compute percentiles for each stock
    stock_percentile_df= get_percentile_moves(stock_hist_series_volatility_df,nifty_list)
    #percentiles_path2 = cfg_vars.volatility_dir + 'stock_percentile_volatility.xlsx'
    #stock_percentile_df.to_excel(percentiles_path2, index=False)
    
    ###Step6. Apply these percentiles to the current series
    current_series_start_dt = pd.to_datetime('2026-01-01', format='%Y-%m-%d')
    # Filter all_stocks_df for current_series_start_dt and nifty_list
    curr_series_open_df = all_stocks_df[
        (all_stocks_df['trade_dt'] == current_series_start_dt) &
        (all_stocks_df['tckr_symbol'].isin(nifty_list))
    ][['tckr_symbol', 'trade_dt', 'open_price']]
    #curr_series_path = cfg_vars.volatility_dir + 'current_series_spread.xlsx'
    #curr_series_open_df.to_excel(curr_series_path, index=False)
    
    ### Step7 Join current_series_df with stock_percentile_df to get expected up and down moves
    curr_series_open_spread_df = get_series_spread(curr_series_open_df, stock_percentile_df) 
    #merged_path = cfg_vars.volatility_dir + 'curr_series_spread.xlsx'
    #curr_series_open_spread_df.to_excel(merged_path, index=False)

    ###Step9. Calculate timing and payoffs based on current close prices
    curr_series_close_spread_df= calculate_timing_and_payoffs(all_stocks_df,curr_series_open_spread_df,nifty_list)
    merged_path1 = cfg_vars.volatility_dir + 'curr_series_close_spread.xlsx'
    curr_series_close_spread_df.to_excel(merged_path1, index=False)

if __name__ == "__main__": 
    main()    