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

def process_nse_stocks(sector, nifty_list, start_date):
    '''
    Process all stocks in NSE_DATA for EMA Scalping signals
    '''
    print("...........Step1. Read all Stocks Data NSE_DATA................")
    
    stocks_df = read_nse_data(nifty_list, start_date)   

    # Ensure High is highest and Low is lowest
    stocks_df['high_price'] = stocks_df[['open_price', 'high_price', 'closing_price']].max(axis=1) + 1
    stocks_df['low_price'] = stocks_df[['open_price', 'low_price', 'closing_price']].min(axis=1) - 1
    
    all_signals_df = []
    all_data_df = []
    
    print("Start Processing NSE Stocks for Swing Signals Generation:")
    for symbol in nifty_list:
        print(f"..{symbol}...")
        stock_data = stocks_df[stocks_df['tckr_symbol'] == symbol].copy()
        
        if stock_data.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        
        # Prepare data for EMASignalGenerator
        stock_data['trade_dt'] = pd.to_datetime(stock_data['trade_dt'])
        stock_data = stock_data.set_index('trade_dt').sort_index()
        stock_data = stock_data.rename(columns={
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'closing_price': 'Close',
            'total_trading_volume': 'Volume'
        })
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        try:
            # Initialize strategy
            strategy = SwingTradingStrategyV2(stock_data, symbol=symbol)
            #Run the Strategy
            signals, data_with_indicators = strategy.run()

            # Convert signals to DataFrame for analysis
            signals_df = pd.DataFrame(signals)
            data_with_indicators_df = pd.DataFrame(data_with_indicators)

            all_signals_df.append(signals_df)
            all_data_df.append(data_with_indicators_df)
            #print(f"Generated {len(signals)} signals for {symbol}")
            #print("                             ")
        except Exception as e:
            print(f"Error processing {symbol}: {traceback.format_exc()}")
            continue
    return all_signals_df, all_data_df

def transform_signals(combined_signals_df, filter_date):
    # combined_signals_df is already concatenated, so no need to concat here
    
    # Calculate duration in days between exit_date and date (before formatting)
    combined_signals_df['duration_days'] = (combined_signals_df['exit_date'] - combined_signals_df['date']).dt.days
    
    # Set Date Format
    combined_signals_df['date'] = combined_signals_df['date'].dt.strftime('%Y-%m-%d')
    combined_signals_df['exit_date'] = combined_signals_df['exit_date'].dt.strftime('%Y-%m-%d')

    # Calculate Profit% in a week & 2 weeks & 1 month
    combined_signals_df['pnl_1w_pct'] = combined_signals_df['pnl_pct'] / (combined_signals_df['bars_held'] / 5)
    combined_signals_df['pnl_2w_pct'] = combined_signals_df['pnl_pct'] / (combined_signals_df['bars_held'] / 10)
    combined_signals_df['pnl_1m_pct'] = combined_signals_df['pnl_pct'] / (combined_signals_df['bars_held'] / 22)

    # Round specified columns to 2 decimal places
    cols_to_round = ['stop_loss', 'target_1', 'target_2', 'rsi', 'rvol', 'pnl_pct', 'pnl_1w_pct', 'pnl_2w_pct', 'pnl_1m_pct']
    combined_signals_df[cols_to_round] = combined_signals_df[cols_to_round].round(2)

    # Reorder the Columns (add 'duration_days' to the list)
    order_cols = ['symbol', 'date', 'entry_price', 'exit_reason', 'regime', 'signal', 'pattern', 
                'impulse_score','quality_score', 'impulse_details','quality_details', 
                'pnl_1w_pct', 'pnl_2w_pct', 'pnl_1m_pct', 'exit_date', 'exit_price',  'exit_details', 
                'duration_days', 'stop_loss', 'target_1', 'target_2', 'pnl_pct', 'rsi', 'rvol', 'bars_held', 
                'position_size', 'index']
    combined_signals_df = combined_signals_df[order_cols]    

    # Filter Data
    combined_signals_filt_df = combined_signals_df[pd.to_datetime(combined_signals_df['date']) > pd.to_datetime(filter_date)]
    
    return combined_signals_filt_df

    # Filter where type is not empty
    #combined_signals_filt_df = combined_signals_df    ### Remove this line if not needed
    #combined_signals_filt_df = combined_signals_df[combined_signals_df['type'] != '']
    
# ============================================================================
#                                MAIN WORKFLOW
# ============================================================================
if __name__ == "__main__":    
    sector ='fno_sects'  # 'test' / 'movers' / 'fno_sects' / 'cash'
    start_date = '2024-01-01'
    #filter_date = '2025-11-01'
    filter_date = start_date

    swing_path1 = cfg_vars.swing_model_dir + f'{sector}_signals.xlsx'
    summary_path = cfg_vars.swing_model_dir + f'{sector}_exit_summary.xlsx'
    swing_path2 = cfg_vars.swing_model_dir + f'{sector}_data.xlsx'

    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'movers': nifty_list = cfg_nifty.nifty_movers
        case 'fno_sects': nifty_list = list(set(cfg_nifty.nifty_fno + cfg_nifty.nifty_sectoral))
        case 'cash': nifty_list = cfg_nifty.nifty_cash  
        case _: print("Invalid Sector")
    
    all_signals_df, all_data_df = process_nse_stocks(sector, nifty_list, start_date)


    if all_signals_df:
        #--------------------------------------------------------------
        # Export All Signals
        #--------------------------------------------------------------
        signals_df = pd.concat(all_signals_df, ignore_index=True) 
        signals_filt_df = transform_signals(signals_df, filter_date)
        signals_filt_df['date'] = pd.to_datetime(signals_filt_df['date']).dt.strftime('%Y-%m-%d')
        signals_filt_df.to_excel(swing_path1, index=False)
        sheets = {
            'TREND_LONG': (signals_filt_df['regime'] == 'TRENDING') & (signals_filt_df['signal'] == 'LONG'),
            'TREND_SHORT': (signals_filt_df['regime'] == 'TRENDING') & (signals_filt_df['signal'] == 'SHORT'),
            'CONSOLIDATING': signals_filt_df['regime'] == 'CONSOLIDATING'}
        with pd.ExcelWriter(swing_path1) as writer:
            for sheet_name, mask in sheets.items():
                signals_filt_df[mask].to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Swing Signal Generation! Total signals: {len(signals_filt_df)}")

        #--------------------------------------------------------------
        # Export All Data
        #--------------------------------------------------------------
        data_df = pd.concat(all_data_df, ignore_index=True)
        data_df['date'] = pd.to_datetime(data_df['date']).dt.strftime('%Y-%m-%d')
        combined_data_df = data_df.merge(signals_filt_df, on=['symbol', 'date'], how='left')

        order_cols1 = ['symbol', 'date', 'entry_price', 'exit_reason', 'regime', 'signal', 'pattern',
       'impulse_score', 'impulse_details','quality_score', 'quality_details', 'pnl_1w_pct', 'pnl_2w_pct',
       'pnl_1m_pct', 'exit_date', 'exit_price',  'exit_details',
       'duration_days', 'stop_loss', 'target_1', 'target_2', 'pnl_pct', 'rsi',
       'rvol', 'bars_held', 'position_size', 'Open', 'High', 'Low', 'Close', 'Volume',  'EMA3_High',
       'EMA3_Low', 'EMA3_Close', 'EMA8_High', 'EMA8_Low', 'EMA8_Close',
       'EMA20', 'EMA50', 'EMA200', 'SuperTrend', 'ST_Direction', 'ATR',
       'Trend', 'RSI', 'RSI_Slope', 'RSI_Rising', 'Avg_Volume_20', 'RVol',
       'BBW', 'ER', 'BBW_Expanding', 'Regime', 'True_Range', 'Avg_TR',
       'Range_OK','index']
        combined_data_df = combined_data_df[order_cols1]
        combined_data_df.to_excel(swing_path2, index=False)
        
        #--------------------------------------------------------------
        # Export Signal Summary for each Pattern
        #--------------------------------------------------------------
        summary = signals_df.groupby(['regime', 'signal', 'pattern', 'exit_reason']).size().unstack(fill_value=0)
        print("Exit Reason Summary by Signal and Pattern:")
        summary.to_excel(summary_path, index=True)
        print(f"Exit summary saved to {summary_path}")
    else:
        print("No signals generated for any stock.")