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


#------------------------------------------------------------------------------
# Function to get Monday date from buy_date
def get_monday_date(input_date):
    if input_date == '':
        return pd.NaT
    input_date = pd.to_datetime(input_date)
    day = input_date.weekday()  # 0=Monday
    if day == 0:
        return input_date.strftime('%Y-%m-%d')
    else:
        monday_date = input_date - pd.Timedelta(days=day)
        return monday_date.strftime('%Y-%m-%d')

def add_weekly_signals(df: pd.DataFrame,weekly_data_path) -> pd.DataFrame:
    ## Get Weekly Data
    weekly_df = pd.read_excel(weekly_data_path)
    weekly_df['w_start_date'] = pd.to_datetime(weekly_df['w_start_date'])
    #weekly_df = weekly_data_df.reset_index()
    # weekly_df.rename(columns={'index': 'w_start_date'}, inplace=True)  # Removed: incorrect, no 'index' column exists
    
    
    # Ensure date columns are datetime before converting to date objects
    df['w_start_date'] = pd.to_datetime(df['w_start_date'])
    weekly_df['w_start_date'] = pd.to_datetime(weekly_df['w_start_date'])

    # Convert to date objects for matching
    df['w_start_date'] = df['w_start_date'].dt.date
    weekly_df['w_start_date'] = weekly_df['w_start_date'].dt.date
    

    #weekly_df.to_excel(cfg_vars.swing_reversal_rsi_model_dir + 'debug_weekly_data.xlsx', index=False)
    # Join Daily with Weekly_data_df 
    weekly_cols = ['tckr_symbol', 'w_start_date', 
        'weekly_structure_score', 'flag_weekly_trend_stack', 
        'flag_weekly_range_strength', 'flag_weekly_atr_ok', 
        'flag_weekly_bbw_compression']


    df = df.merge(weekly_df[weekly_cols], 
              left_on=['tckr_symbol', 'w_start_date'], 
              right_on=['tckr_symbol', 'w_start_date'], 
              how='inner')
    #df = df.set_index('date')
    #print ("Columns after merging with weekly data:", df.columns)
    
    #df.drop('w_start_date', axis=1, inplace=True)

    # Opt into future downcasting behavior to suppress warning
    pd.set_option('future.no_silent_downcasting', True)

    return df
    
def calculate_pivot_points(df):
    """
    Calculate Pivot Points and set pivot_signal based on close price.
    
    Parameters:
    df: DataFrame with 'high', 'low', 'close' columns
    
    Returns:
    DataFrame with Pivot Points and pivot_signal added
    """
    # Calculate Pivot Point (PP)
    df['pp'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate R1, S1, R2, S2
    df['r1'] = 2 * df['pp'] - df['low']
    df['s1'] = 2 * df['pp'] - df['high']
    df['r2'] = df['pp'] + (df['high'] - df['low'])
    df['s2'] = df['pp'] - (df['high'] - df['low'])
    
    # Set pivot_signal based on close
    df['pivot_signal'] = ''
    for i in range(len(df)):
        close = df['close'].iloc[i]
        pp = df['pp'].iloc[i]
        r1 = df['r1'].iloc[i]
        s1 = df['s1'].iloc[i]
        s2 = df['s2'].iloc[i]
        
        if pd.notna(pp) and pd.notna(r1) and pd.notna(s1) and pd.notna(s2):
            if close > r1:
                df.loc[df.index[i], 'pivot_signal'] = 'strong_bullish'
            elif pp < close <= r1:
                df.loc[df.index[i], 'pivot_signal'] = 'healthy_bullish'
            elif s1 <= close <= pp:
                df.loc[df.index[i], 'pivot_signal'] = 'weak_bullish'
            elif s2 < close < s1:
                df.loc[df.index[i], 'pivot_signal'] = 'weak_bearish'
            elif close <= s2:
                df.loc[df.index[i], 'pivot_signal'] = 'strong_bearish'
    
    return df
import numpy as np
import pandas as pd

def apply_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refined hard gate logic.
    
    Hard Gates (must pass):
        - RSI Momentum
        - EMA Alignment
        - WMA Alignment
        - EMA Support
        - Strong Breakout Quality
        - ATR Expansion
        - Volume Expansion

    Soft Quality Filters (scored but not mandatory):
        - Not Overextended
        - Distance From Resistance

    Returns:
        - hard_gate_score (hard conditions only)
        - quality_score (soft filters)
        - hard_gate_buy (True if all hard gates pass)
    """

    """
    Checks if a stock passes the filters specified in the image.
    Expects df to have columns: 'open', 'high', 'low', 'close', 'volume' 
    with a DatetimeIndex.
    """
    # Ensure column names are lowercase for pandas_ta
    df.columns = [col.lower() for col in df.columns]
    
    # 1. Calculate Daily Indicators
    df['rsi_9'] = ta.rsi(df['close'], length=9)
    df['ema_3'] = ta.ema(df['close'], length=3)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['wma_21'] = ta.wma(df['close'], length=21)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # 2. Resample to Weekly for Weekly Indicators
    # Aggregating daily data into weekly bars
    df_weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate Weekly Indicators
    df_weekly['w_ema_3'] = ta.ema(df_weekly['close'], length=3)
    df_weekly['w_wma_21'] = ta.wma(df_weekly['close'], length=21)
    
    # 3. Map Weekly values back to Daily timeframe (Forward Fill)
    # This allows comparing Daily values against the current Weekly value
    df['w_ema_3'] = df_weekly['w_ema_3'].reindex(df.index, method='ffill')
    df['w_wma_21'] = df_weekly['w_wma_21'].reindex(df.index, method='ffill')
    
    # Assign column aliases for conditions
    df['ema_3_d'] = df['ema_3']
    df['ema_3_w'] = df['w_ema_3']
    df['wma_21_d'] = df['wma_21']
    df['wma_21_w'] = df['w_wma_21']
    df['prev_high'] = df['high'].shift(1)

    # Opt into future downcasting behavior to suppress warning
    pd.set_option('future.no_silent_downcasting', True)

    # Replace None with NaN to handle comparisons
    df = df.replace({None: np.nan})

    # -----------------------------
    # HARD GATES
    # -----------------------------

    # 1. RSI Momentum
    cond_rsi = df['rsi_9'] > 50

    # 2. EMA Trend Alignment
    cond_ema_alignment = df['ema_3_d'] > df['ema_3_w']

    # 3. WMA Trend Alignment
    cond_wma_alignment = df['wma_21_d'] > df['wma_21_w']

    # 4. EMA Support
    cond_ema_support = df['close'] > df['ema_21']

    # 5. Strong Breakout Quality
    cond_close_above_prev_high = df['close'] > df['prev_high']

    candle_position = (
        (df['close'] - df['low']) /
        (df['high'] - df['low']).replace(0, np.nan)
    )

    cond_strong_close = candle_position > 0.7

    cond_breakout_quality = cond_close_above_prev_high & cond_strong_close

    # 6. ATR Expansion
    cond_atr_expansion = df['atr_14'] > df['atr_14'].shift(5)

    # 7. Volume Expansion
    cond_volume_expansion = (
        df['volume'] > 1.5 * df['volume'].rolling(20).mean()
    )

    hard_conditions = [
        cond_rsi,
        cond_ema_alignment,
        cond_wma_alignment,
        cond_ema_support,
        cond_breakout_quality,
        cond_atr_expansion,
        cond_volume_expansion
    ]

    df['hard_gate_score'] = sum(hard_conditions)
    df['hard_gate_buy'] = df['hard_gate_score'] == len(hard_conditions)

    # -----------------------------
    # SOFT QUALITY FILTERS
    # -----------------------------

    # 1. Not Overextended (within 3% of EMA21)
    cond_not_overextended = (
        (df['close'] - df['ema_21']) / df['ema_21']
    ) < 0.03

    # 2. Distance From Resistance (2% room to R1)
    cond_resistance_room = (
        (df['r1'] - df['close']) / df['close']
    ) > 0.02

    soft_conditions = [
        cond_not_overextended,
        cond_resistance_room
    ]

    df['quality_score'] = sum(soft_conditions)

    # Calculate percentage change in closing price
    df['pct_change'] = df['close'].pct_change() * 100

    # Return the filtered DataFrame with only rows that pass all conditions
    return df

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

def process_nse_stocks(sector, nifty_list, start_date):
    '''
    Process all stocks in NSE_DATA for strategy signals using detect_buy_signals (one stock at a time)
    '''
    print("...........Step1. Read all Stocks Data NSE_DATA................")
    
    stocks_df = read_nse_data(nifty_list, start_date)
    stocks_df = stocks_df.rename(columns={
            'trade_dt': 'date',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'closing_price': 'close',
            'total_trading_volume': 'volume'
        })
    stocks_df = stocks_df[['date','tckr_symbol','open', 'high', 'low', 'close', 'volume']]

    # Ensure High is highest and Low is lowest
    # Use raw high and low (no adjustments)
    stocks_df['high'] = stocks_df['high']
    stocks_df['low'] = stocks_df['low']
    
    all_signals_df = []  # List to hold DataFrames with signals for each stock
    
    print("Start Processing NSE Stocks for Strategy Signals Generation (one stock at a time):")
    for symbol in nifty_list:
        print(f"..{symbol}...")
        stock_data = stocks_df[stocks_df['tckr_symbol'] == symbol].copy()
        
        if stock_data.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        
        # Prepare data for detect_buy_signals
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.set_index('date').sort_index()

        try:
            # Process one stock at a time using detect_buy_signals
            # Calculate Supertrend, Bollinger Bands, and Pivot Points
            stock_data = calculate_pivot_points(stock_data)
            result_df = apply_hard_gates(stock_data)
            
            #result_df = detect_buy_signals(stock_data, tolerance=0.001)
            
            # Add symbol column for identification
            #result_df['symbol'] = symbol
            
            # Collect the result (DataFrame with signals)
            if not result_df.empty:
                all_signals_df.append(result_df)

            # Count signals
            #buy_count = len(result_df[result_df['buy_signal'] == True])
            #watchlist_count = len(result_df[result_df['watchlist'] == True])    
            #print(f"{symbol}: {watchlist_count // 5} flat periods found (5 days each), {buy_count} buy signals generated")
        
        except Exception as e:
            print(f"Error processing {symbol}: {traceback.format_exc()}")
            continue
    
    # Optionally, combine all results into a single DataFrame if needed
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)
        #print(f"Total signals across all stocks: {len(combined_df[combined_df['buy_signal'] == True])}")
        # You can save or return combined_df as needed
    
    return all_signals_df  # Returns list of DataFrames, one per stock