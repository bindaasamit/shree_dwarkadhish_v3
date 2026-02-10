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

def calculate_ema(df):
    ##### Compute exit EMAs (global, once)
    df['ema5']  = df['close'].ewm(span=5, adjust=False).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_supertrend(df, period=10, multiplier=3):
    """
    Calculate Supertrend indicator
    
    Parameters:
    df: DataFrame with 'high', 'low', 'close' columns
    period: ATR period (default 10)
    multiplier: ATR multiplier (default 3)
    
    Returns:
    DataFrame with supertrend columns added
    """
    # Calculate ATR
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Calculate ATR14
    df['atr14'] = df['tr'].rolling(window=14).mean()
    
    # Calculate basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']

    # Initialize final bands
    df['final_ub'] = 0.0
    df['final_lb'] = 0.0
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 0  # 1 for bullish (green), -1 for bearish (red)
    
    for i in range(period, len(df)):
        # Final upper band
        if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i-1] or df['close'].iloc[i-1] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
        else:
            df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i-1]
        
        # Final lower band
        if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i-1]
        
        # Supertrend
        if df['supertrend'].iloc[i-1] == df['final_ub'].iloc[i-1]:
            if df['close'].iloc[i] <= df['final_ub'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
        else:
            if df['close'].iloc[i] >= df['final_lb'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
    
    return df

def calculate_bbw(df, period=20, std_multiplier=2):
    """
    Calculate Bollinger Bands and set volatility_signal based on close price.
    
    Parameters:
    df: DataFrame with 'close' column
    period: Period for SMA and std (default 20)
    std_multiplier: Multiplier for standard deviation (default 2)
    
    Returns:
    DataFrame with Bollinger Bands and volatility_signal added
    """
    # Calculate middle band (SMA)
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    df['bb_std'] = df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['bb_upper'] = df['bb_middle'] + (std_multiplier * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (std_multiplier * df['bb_std'])
    df['bbw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Set volatility_signal based on close
    df['volatility_signal'] = ''
    for i in range(len(df)):
        close = df['close'].iloc[i]
        upper = df['bb_upper'].iloc[i]
        middle = df['bb_middle'].iloc[i]
        lower = df['bb_lower'].iloc[i]
        
        if pd.notna(upper) and pd.notna(middle) and pd.notna(lower):
            if close > upper:
                df.loc[df.index[i], 'volatility_signal'] = 'ignore_sbull'
            elif middle < close <= upper:
                df.loc[df.index[i], 'volatility_signal'] = 'healthy_bullish'
            elif lower <= close <= middle:
                df.loc[df.index[i], 'volatility_signal'] = 'bearish_bias'
            elif close < lower:
                df.loc[df.index[i], 'volatility_signal'] = 'ignore_sbear'
            #print(f"close={close},upper={upper},middle={middle}, lower={lower}, volatility_signal={df.loc[df.index[i], 'volatility_signal']}")
    
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

def get_weekly_trend(df: pd.DataFrame,weekly_data_path) -> pd.DataFrame:
    weekly_data_df = pd.read_excel(weekly_data_path)
    weekly_data_df = weekly_data_df.set_index('trade_dt')
    weekly_data_df.index = pd.to_datetime(weekly_data_df.index)
        
    for idx, row in df.iterrows():
        if row['buy_date'] == '9999-12-31':
            continue # Skip if no buy date
        #print(f"Processing weekly trend for {row['tckr_symbol']} on buy date {row['buy_date']}")
        # Convert buy_date to datetime
        buy_date = pd.to_datetime(row['buy_date'])
        day = buy_date.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
        if day == 0:  # Monday
            monday_date = buy_date
        else:
            monday_date = buy_date - pd.Timedelta(days=day)
    
        # Get the close from weekly_data_df for the Monday date
        symbol = row['tckr_symbol']
        weekly_row = weekly_data_df[(weekly_data_df.index == monday_date) & (weekly_data_df['tckr_symbol'] == symbol)]
        if not weekly_row.empty:
            weekly_trend = weekly_row['wema-signal'].iloc[0]  # Change 'wema_signal' to 'W-EMA-Signal'
        else:
            weekly_trend = None
          
        #print(f"......weekly trend close: {weekly_trend}")
        df.at[idx, 'weekly_trend'] = weekly_trend
    return df

def apply_breakout_strategy(df):
    df = df.copy()
    
    # ------------------------- 
    # 1. Highest High (120 days)
    # -------------------------
    df['hh_120'] = df['high'].shift(1).rolling(120).max()  # Highest high of the last 120 days (up to yesterday)
    df['hh_120_two_days_ago'] = df['high'].shift(2).rolling(120).max()  # 120-day highest high as of two days ago

    df['flag_120d_breakout'] = (
        (df['close'] > df['hh_120']) &
        (df['close'].shift(1) < df['hh_120_two_days_ago'])
    )

    # ------------------------------------------------- 
    # 2. % Price Change > 3% & todays candle is Bullish
    # -------------------------------------------------
    df['pct_change'] = df['close'].pct_change() * 100
    df['flag_pct_move'] = (df['pct_change'] > 3) & (df['open'] < df['close'])

    # ------------------------- 
    # 3. Strong Volume
    # -------------------------
    df['vol_20_avg'] = df['volume'].rolling(20).mean()
    df['flag_volume'] = df['volume'] > df['vol_20_avg']

    # ------------------------- 
    # 4. RSI > 65
    # -------------------------
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['flag_rsi'] = df['rsi_14'] > 65

    # ------------------------- 
    # 5. Supertrend Green
    # -------------------------
    # ASSUMPTION:
    # df['supertrend'] exists
    # df['supertrend_direction'] = 1 (green), -1 (red)

    df['flag_supertrend'] = df['supertrend_direction'] == 1

    # ------------------------- 
    # 6. Bollinger Band Width at 20–30 day low
    # -------------------------

    df['bbw_min_30'] = df['bbw'].rolling(30).min()
    df['flag_bbw_compression'] = df['bbw'] <= df['bbw_min_30']

    # ------------------------- 
    # 7. Generate BuyExit Date
    # -------------------------
    # Calculate EMA-20,EMA-23 and EMA-50
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    
    # ------------------------- 
    # 8. Absolute Daily Liquidity Filter
    # -------------------------
    df['turnover'] = df['close'] * df['volume']
    df['avg_turnover_20'] = df['turnover'].rolling(20).mean()
    df['flag_liquidity_turnover'] = df['avg_turnover_20'] >= 2.5e8   # ₹25 Cr

    # ------------------------- 
    # BUY SIGNAL LOGIC
    # -------------------------
    df['buy_signal'] = (
        df['flag_120d_breakout'] &
        df['flag_volume'] &
        df['flag_rsi'] &
        df['flag_supertrend'] &
        #df['flag_bbw_compression'] & 
        df['flag_pct_move']
    )
    df['buy_date'] = df.index.strftime('%Y-%m-%d').where(df['buy_signal'], '')
    
    #Alternative approach using breakout_score
    df['breakout_score'] = (
        df['flag_120d_breakout'].astype(int) * 3 +
        df['flag_volume'].astype(int) * 2 +
        df['flag_bbw_compression'].astype(int) * 2 +
        df['flag_rsi'].astype(int) * 1 +
        df['flag_supertrend'].astype(int) * 1 +
        df['flag_pct_move'].astype(int) * 1
    )
    #df['buy_signal'] = df['breakout_score'] >= 7

    df['buyexit_date'] = df.index.strftime('%Y-%m-%d').where(df['close'] < df['ema20'], '')

    ###Amit Trade Exit Should be done here....
    
    return df

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

        # Convert columns to numeric to avoid string operations
        stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
        stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
        stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
        stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
        stock_data['volume'] = pd.to_numeric(stock_data['volume'], errors='coerce')

        # Calculate Supertrend, Bollinger Bands, and Pivot Points
        stock_data = calculate_ema(stock_data)
        stock_data = calculate_supertrend(stock_data)
        stock_data = calculate_bbw(stock_data)
        stock_data = calculate_pivot_points(stock_data)
    
        try:
            # Process one stock at a time using detect_buy_signals
            result_df = apply_breakout_strategy(stock_data)
            # Collect the result (DataFrame with signals)
            all_signals_df.append(result_df)
            buy_count = len(result_df[result_df['buy_signal'] == True])        
        except Exception as e:
            print(f"Error processing {symbol}: {traceback.format_exc()}")
            continue
    
    # Optionally, combine all results into a single DataFrame if needed
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)
        #print("Columns in combined_df:", combined_df.columns)
        print(f"Total signals across all stocks: {len(combined_df[combined_df['buy_signal'] == True])}")
        # You can save or return combined_df as needed
    
    return all_signals_df  # Returns list of DataFrames, one per stock

def apply_hard_structural_exit_with_pnl(df, close_df, max_early_days=3):
    """
    1. Applies HARD structural exits (breakout rejection, volume failure, no expansion)
    2. Computes pct_profit_5days exactly as defined

    pct_profit_5days logic:
    - % diff between entry close and close after 5 candles
    """

    df = df.sort_values(['tckr_symbol', 'date']).copy()

    df['exit_signal'] = False
    df['exit_reason'] = ''  
    #df['buyexit_date'] = ''  # Initialize as empty string for consistency
    df['pct_profit_5days'] = np.nan

    # Pre-sort close_df once
    close_df = close_df.sort_values(['tckr_symbol', 'date'])

    # -------------------------------
    # PART 1: HARD STRUCTURAL EXITS
    # -------------------------------
    for symbol, sdf in df.groupby('tckr_symbol'):
        sdf = sdf.reset_index()

        buy_indices = sdf.index[sdf['buy_signal']].tolist()

        for buy_idx in buy_indices:
            buy_row = sdf.loc[buy_idx]

            breakout_level = buy_row['hh_120']
            atr = buy_row['atr14']
            vol_avg = buy_row['vol_20_avg']

            window = sdf.loc[buy_idx : buy_idx + max_early_days]

            if len(window) < 2:
                continue

            # 1. Breakout rejection
            rej = window[window['close'] < breakout_level]
            if not rej.empty:
                exit_row = rej.iloc[0]
                df.at[exit_row['index'], 'exit_signal'] = True
                df.at[exit_row['index'], 'exit_reason'] = 'breakout_rejection'
                df.at[exit_row['index'], 'buyexit_date'] = exit_row['date']
                continue

            # 2. Volume failure
            vol_fail = window[window['volume'] < 0.7 * vol_avg]
            if not vol_fail.empty:
                exit_row = vol_fail.iloc[0]
                df.at[exit_row['index'], 'exit_signal'] = True
                df.at[exit_row['index'], 'exit_reason'] = 'volume_failure'
                df.at[exit_row['index'], 'buyexit_date'] = exit_row['date']
                continue

            # 3. No expansion
            if window['high'].max() < breakout_level + atr:
                exit_row = window.iloc[-1]
                df.at[exit_row['index'], 'exit_signal'] = True
                df.at[exit_row['index'], 'exit_reason'] = 'no_expansion'
                df.at[exit_row['index'], 'buyexit_date'] = exit_row['date']

    # -------------------------------
    # PART 2: pct_profit_5days
    # -------------------------------
    for idx in df[df['buy_signal']].index:
        symbol = df.at[idx, 'tckr_symbol']
        buy_date = df.at[idx, 'date']
        entry_close = df.at[idx, 'close']
        
        # Always use 5 days forward
        forward_candles = 5

        symbol_prices = close_df[close_df['tckr_symbol'] == symbol].reset_index(drop=True)

        pos = symbol_prices.index[symbol_prices['date'] == buy_date]
        if len(pos) == 0:
            continue

        future_pos = pos[0] + forward_candles
        if future_pos >= len(symbol_prices):
            continue

        future_close = symbol_prices.loc[future_pos, 'close']

        df.at[idx, 'pct_profit_5days'] = (
            (future_close - entry_close) / entry_close
        ) * 100

    return df

#------------------------------------------------------------------------------
#   Main Workflow Functions
#------------------------------------------------------------------------------
def gen_basic_breakout_signals(sector, nifty_list, start_date):
    all_signals_df = process_nse_stocks(sector, nifty_list, start_date)
    df = pd.concat(all_signals_df, ignore_index=False)
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    ord_cols = ['tckr_symbol','date', 
        'buy_signal', 'breakout_score', 'flag_120d_breakout', 
        'flag_volume', 'flag_bbw_compression', 'flag_rsi', 'flag_supertrend', 'flag_pct_move', 
        'pivot_signal','volatility_signal', 'pct_change', 'buy_date', 'buyexit_date', 'rsi', 
        'ema5','ema10','ema20','close','high','hh_120','atr14','volume','vol_20_avg']
        #'bbw_min_30', 'vol_20_avg','rsi_14','open', 'high', 'low', 'close', 'volume', 
        #'h-l', 'h-pc', 'l-pc', 'tr', 'atr', 'atr14', 'basic_ub', 'basic_lb', 
        #'final_ub', 'final_lb', 'supertrend',
        #'supertrend_direction', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
        #'bbw', 'pp', 'r1', 's1', 'r2', 's2', 'hh_120','hh_120_two_days_ago']
    df = df[ord_cols]

    close_df = df[['tckr_symbol', 'date', 'close']]
    return df, close_df

def gen_trade_group_summary(df):
    # 1. Sort combined_df by 'tckr_symbol' and 'date'
    df = df.sort_values(by=['tckr_symbol', 'date']).reset_index(drop=True)

    # 2. Extract groups of records for each stock where the first has buy_date populated and the last has buyexit_date populated
    grouped_records = []
    for symbol, group in df.groupby('tckr_symbol'):
        group = group.reset_index(drop=True)
        buy_indices = group[group['buy_date'] != ''].index
        exit_indices = group[group['buyexit_date'] != ''].index
        
        for buy_idx in buy_indices:
            # Find the next exit after the buy (if any)
            next_exits = exit_indices[exit_indices > buy_idx]
            exit_date = ''
            if len(next_exits) > 0:
                exit_idx = next_exits[0]
                exit_date = group.loc[exit_idx, 'buyexit_date']
            
            # Always create a summary record for the buy
            record = group.loc[buy_idx, [
                'tckr_symbol', 'date', 'buy_signal', 'breakout_score',
                'flag_120d_breakout', 'flag_volume', 'flag_pct_move', 'flag_rsi', 
                'flag_bbw_compression', 'flag_supertrend', 'pivot_signal', 
                'volatility_signal', 'pct_change', 'buy_date',
                'ema5','ema10','ema20','close','high','hh_120','atr14','volume','vol_20_avg']].copy()
                #'price_move_pct_5d', 'exit_ema_type', 'exit_signal']].copy()
            record['buyexit_date'] = exit_date
            grouped_records.append(record)

    # Create grouped_df from the collected records
    grouped_df = pd.DataFrame(grouped_records)

    # Calculate w_start_date for grouped_df (Get Start of the Week)
    grouped_df['w_start_date'] = grouped_df['buy_date'].apply(get_monday_date)

    # 3. grouped_df now contains one record per trade group with the specified columns
    # (Optional: Save or further process grouped_df as needed)
    # grouped_df.to_excel('grouped_trades.xlsx', index=False)
    return grouped_df

# Function to get Monday date from buy_date
def get_monday_date(buy_date_str):
    if buy_date_str == '':
        return pd.NaT
    buy_date = pd.to_datetime(buy_date_str)
    day = buy_date.weekday()  # 0=Monday
    if day == 0:
        return buy_date
    else:
        return buy_date - pd.Timedelta(days=day)

def add_weekly_signals(df: pd.DataFrame,weekly_data_path) -> pd.DataFrame:
    ## Get Weekly Data
    weekly_data_df = pd.read_excel(weekly_data_path)
    weekly_data_df['w_start_date'] = pd.to_datetime(weekly_data_df['w_start_date'])
    
    # Join with weekly_data_df on tckr_symbol and w_start_date (left join to keep all df rows)
    # Calculate trade_duration
    df['trade_duration'] = df.apply(
        lambda row: 9999 if row['buyexit_date'] == '' else (pd.to_datetime(row['buyexit_date']) - pd.to_datetime(row['buy_date'])).days,
        axis=1)

    df = df.merge(weekly_data_df,on=['tckr_symbol', 'w_start_date'],how='left', suffixes=('', '_weekly'))
    
    ord_cols = ['tckr_symbol', 'buy_date', 'buyexit_date', 'trade_duration', 
                'buy_signal', 'breakout_score', 'weekly_structure_score',
                'flag_120d_breakout', 'flag_volume', 'flag_pct_move', 'flag_rsi', 'flag_supertrend', 
                'flag_weekly_trend_stack', 'flag_weekly_range_strength', 'flag_weekly_atr_ok', 'flag_weekly_bbw_compression',
                'flag_bbw_compression', 'pivot_signal', 'volatility_signal', 'pct_change', 
                'close','high','hh_120','atr14','volume','vol_20_avg','date']
                #'w_start_date',   'w_open', 'w_high', 'w_low', 'w_close', 'w_volume', 'w_ema10', 'w_ema20', 'w_ema50', 'w_high_26', 'w_low_26', 'weekly_range_pos', 'w_atr', 'w_atr_ratio', 'w_atr_median_26', 'w_bbw', 'w_bbw_pct_25']
    df = df[ord_cols]
    
    return df