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

def check_supertrend_flat(df, start_idx, tolerance=0.001):
    """
    Check if supertrend is flat for exactly 5 days starting from start_idx
    
    Parameters:
    df: DataFrame with supertrend values
    start_idx: Start index position
    tolerance: Tolerance for flatness check (default 0.001 = 0.1%)
    
    Returns:
    Boolean indicating if supertrend is flat for 5 days
    """
    if start_idx + 5 > len(df):
        return False
    
    st_values = df['supertrend'].iloc[start_idx:start_idx+5]
    
    # Check if all values are close to each other (within tolerance)
    mean_val = st_values.mean()
    if mean_val == 0:
        return False
    
    variations = abs(st_values - mean_val) / mean_val
    flat_check = all(variations <= tolerance)
    
    # Additional slope constraint: abs(st[t] - st[t-4]) / close < 0.15 * ATR
    slope_check = abs(st_values.iloc[-1] - st_values.iloc[0]) / df['close'].iloc[start_idx + 4] < 0.15 * df['atr14'].iloc[start_idx + 4]
    
    return flat_check and slope_check

def detect_strategy_signals(df, tolerance=0.001):
    """
    Detect buy signals based on the refined strategy:
    1. Supertrend must be GREEN (bullish)
    2. Within the GREEN period, find exactly 5 consecutive days where supertrend is flat
    3. Record highest high from those 5 flat period candles
    4. Generate buy signal when close > highest high (after the flat period)
    5. After buy signal, remain active until exit conditions are met:
       - Close < EMA50 for 2 consecutive days
       - OR (Close < EMA23 for 2 bars AND RSI < 50 AND trend_score < 40)
       - OR Supertrend flips RED
    6. Only one watchlist entry per GREEN stretch (first flat period found)
    
    Parameters:
    df: DataFrame with OHLC data
    tolerance: Tolerance for flatness check (default 0.001 = 0.1%)
    
    Returns:
    DataFrame with signals added
    """
    df = calculate_supertrend(df.copy())
    
    # Calculate EMA-20,EMA-23 and EMA-50
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema23'] = df['close'].ewm(span=23, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # Calculate RSI signal flag
    df['rsi_signal_flag'] = ''
    for i in range(2, len(df)):
        rsi_current = df['rsi'].iloc[i]
        rsi_prev1 = df['rsi'].iloc[i-1]
        rsi_prev2 = df['rsi'].iloc[i-2]
        if pd.notna(rsi_current) and pd.notna(rsi_prev1) and pd.notna(rsi_prev2):
            if rsi_current > 60 and rsi_current > rsi_prev1 and rsi_prev1 > rsi_prev2:
                df.loc[df.index[i], 'rsi_signal_flag'] = 'bullish'
            elif rsi_current < 40 and rsi_current < rsi_prev1 and rsi_prev1 < rsi_prev2:
                df.loc[df.index[i], 'rsi_signal_flag'] = 'bearish'
    
    # Initialize price_move_pct columns
    #df['price_move_pct_5d'] = np.nan
    #df['price_move_pct_10d'] = np.nan
    #df['price_move_pct_15d'] = np.nan
    # Calculate price_move_pct using shift
    df['price_move_pct_5d'] = np.ceil(((df['close'].shift(-5) - df['close']) / df['close']) * 100)
    df['price_move_pct_10d'] = np.ceil(((df['close'].shift(-10) - df['close']) / df['close']) * 100)
    df['price_move_pct_15d'] = np.ceil(((df['close'].shift(-15) - df['close']) / df['close']) * 100)

# Remove the previous loop for price_move_pct calculation.

    # Initialize signal columns
    df['watchlist'] = False
    df['watchlist_active'] = False  # Currently active watchlist
    df['highest_high_flat'] = np.nan
    df['flat_period_start'] = ''
    df['flat_period_end'] = ''
    df['buy_signal'] = False
    df['buy_date'] = ''  # Renamed from 'signal_date'
    df['position_active'] = False  # Track if a position is active after buy
    df['buyexit_signal'] = False
    df['buyexit_date'] = ''
    df['exit_reason'] = ''  # New column for exit reason
    
    # Track state
    current_watchlist_high = None
    watchlist_active = False
    position_active = False
    green_period_watchlist_added = False  # Flag to ensure only one watchlist per GREEN period
    last_buy_index = None  # Track the index of the last buy signal
    current_trend_score = None  # Trend score for the current flat period
    consecutive_ema50_breach = 0  # New: Track consecutive EMA50 breaches
    
    i = 0
    while i < len(df):
        current_direction = df['supertrend_direction'].iloc[i]
        
        # Reset flag when entering a new GREEN period
        if current_direction == 1 and (i == 0 or df['supertrend_direction'].iloc[i-1] == -1):
            green_period_watchlist_added = False
        
        # Check if currently GREEN and no watchlist added for this GREEN period yet
        if current_direction == 1 and not green_period_watchlist_added:
            # Check if we can look ahead 5 days
            if i + 5 <= len(df):
                # Check if next 5 days are all GREEN
                next_5_directions = df['supertrend_direction'].iloc[i:i+5]
                
                if all(next_5_directions == 1):
                    # Check if these 5 days are flat
                    if check_supertrend_flat(df, i, tolerance):
                        # Calculate trend quality score
                        flat_window = df.iloc[:i+5]
                        score = trend_quality_score(flat_window)
                        current_trend_score = score  # Store for potential position
                        
                        # Found a valid flat period!
                        flat_period_data = df.iloc[i:i+5]
                        watchlist_high = flat_period_data['high'].max()
                        
                        # Mark the flat period (exactly 5 days) and add trend_score
                        for j in range(i, i+5):
                            df.loc[df.index[j], 'watchlist'] = True
                            df.loc[df.index[j], 'highest_high_flat'] = watchlist_high
                            df.loc[df.index[j], 'flat_period_start'] = df.index[i].strftime('%Y-%m-%d')
                            df.loc[df.index[j], 'flat_period_end'] = df.index[i+4].strftime('%Y-%m-%d')
                            df.loc[df.index[j], 'trend_score'] = score  # Add trend_score column
                        
                        # Activate watchlist for future candles
                        current_watchlist_high = watchlist_high
                        watchlist_active = True
                        green_period_watchlist_added = True  # Prevent multiple watchlists in same GREEN period
                        
                        # Jump to end of flat period and continue
                        i = i + 5
                        continue
        
        # If watchlist is active, check for buy signal
        if watchlist_active and current_watchlist_high is not None:
            df.loc[df.index[i], 'watchlist_active'] = True
            df.loc[df.index[i], 'highest_high_flat'] = current_watchlist_high
            
            # Check if close breaks above highest high
            if df['close'].iloc[i] > current_watchlist_high:
                df.loc[df.index[i], 'buy_signal'] = True
                df.loc[df.index[i], 'buy_date'] = df.index[i].strftime('%Y-%m-%d')
                position_active = True  # Activate position after buy
                last_buy_index = i  # Track the buy index
                consecutive_ema50_breach = 0  # Reset consecutive breach counter on new position
                
                # Deactivate watchlist after buy signal (but position remains active)
                watchlist_active = False
                current_watchlist_high = None

            # Deactivate watchlist if Supertrend turns RED
            elif current_direction == -1:
                watchlist_active = False
                current_watchlist_high = None
                green_period_watchlist_added = False  # Reset for next GREEN period
        
        # If position is active, check for exit signals
        if position_active:
            df.loc[df.index[i], 'position_active'] = True
            
            # Check exit conditions
            reason = None
            ### Exit Criteria-1 ('ema50 breached for 2 consecutive days')
            # Update consecutive breach counter
            if df['close'].iloc[i] < df['ema50'].iloc[i]:
                consecutive_ema50_breach += 1
            else:
                consecutive_ema50_breach = 0

            if consecutive_ema50_breach >= 2:
                reason = 'ema50 breached for 2 consecutive days'

            ### Exit Criteria-2 (Close < EMA23 for 2 bars AND RSI < 50 AND trend_score < 40) 
            if (df['rsi'].iloc[i] < 50 and 
                df['high'].iloc[i] < df['high'].iloc[i-1] and 
                df['atr14'].iloc[i] < df['atr14'].iloc[i-1] < df['atr14'].iloc[i-2]):
                reason = 'early exit: RSI <50, lower high, ATR contracting'

            if reason:
                df.loc[df.index[i], 'buyexit_signal'] = True
                df.loc[df.index[i], 'buyexit_date'] = df.index[i].strftime('%Y-%m-%d')
                if last_buy_index is not None:
                    df.loc[df.index[last_buy_index], 'exit_reason'] = reason
                position_active = False
                last_buy_index = None
                current_trend_score = None  # Reset                
        
        i += 1
    
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
    Process all stocks in NSE_DATA for strategy signals using detect_strategy_signals (one stock at a time)
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
        
        # Prepare data for detect_strategy_signals
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.set_index('date').sort_index()
        
        try:
            # Process one stock at a time using detect_strategy_signals
            result_df = detect_strategy_signals(stock_data, tolerance=0.001)
            
            # Add symbol column for identification
            result_df['symbol'] = symbol
            
            # Collect the result (DataFrame with signals)
            all_signals_df.append(result_df)

            # Count signals
            buy_count = len(result_df[result_df['buy_signal'] == True])
            watchlist_count = len(result_df[result_df['watchlist'] == True])    
            #print(f"{symbol}: {watchlist_count // 5} flat periods found (5 days each), {buy_count} buy signals generated")
        
        except Exception as e:
            print(f"Error processing {symbol}: {traceback.format_exc()}")
            continue
    
    # Optionally, combine all results into a single DataFrame if needed
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)
        print(f"Total signals across all stocks: {len(combined_df[combined_df['buy_signal'] == True])}")
        # You can save or return combined_df as needed
    
    return all_signals_df  # Returns list of DataFrames, one per stock

def summarize_signals(combined_df, nifty_list):
    '''
    Summarize signals DataFrame into a summary DataFrame per stock
    '''
    # Create summary DataFrame
    summary_rows = []
    
    for symbol in nifty_list:
        stock_df = combined_df[combined_df['symbol'] == symbol]
        stock_df = stock_df.set_index('date').sort_index()  # Set date as index for datetime operations
        unique_flat_starts = stock_df[stock_df['flat_period_start'] != '']['flat_period_start'].unique()
        
        for flat_start in unique_flat_starts:
            group = stock_df[stock_df['flat_period_start'] == flat_start]
            if group.empty:
                continue
            
            tckr_symbol = symbol
            highest_high_flat = group['highest_high_flat'].iloc[0]
            flat_period_start = flat_start
            flat_period_end = group['flat_period_end'].iloc[0]
            watchlist_active = True  # Always True for flat periods
            
            # Find buy signal after flat_period_end
            flat_end_date = pd.to_datetime(flat_period_end)
            subsequent = stock_df[stock_df.index > flat_end_date]
            buy_row = subsequent[subsequent['buy_signal'] == True]
            
            if not buy_row.empty:
                buy_signal = True
                buy_date = buy_row['buy_date'].iloc[0]
                buy_close = buy_row['close'].iloc[0]
                rsi_signal_flag = buy_row['rsi_signal_flag'].iloc[0] if not buy_row.empty else ''
            else:
                buy_signal = False
                buy_date = '9999-12-31'
                buy_close = None
                rsi_signal_flag = ''
            
            # Find "buyexit" signal after buy (if buy exists)
            if buy_signal:
                buyexit_row = subsequent[subsequent['buyexit_signal'] == True]
                if not buyexit_row.empty:
                    buyexit_signal = True
                    buyexit_date = buyexit_row['buyexit_date'].iloc[0]
                    buyexit_close = buyexit_row['close'].iloc[0]
                else:
                    buyexit_signal = False
                    buyexit_date = '9999-12-31'
                    buyexit_close = None
            else:
                buyexit_signal = False
                buyexit_date = '9999-12-31'
                buyexit_close = None
            
            # Durations
            if buy_signal:
                duration_watchlist_to_buy = (pd.to_datetime(buy_date) - pd.to_datetime(flat_period_start)).days
                if buyexit_signal:
                    duration_buy_to_buyexit = (pd.to_datetime(buyexit_date) - pd.to_datetime(buy_date)).days
                else:
                    duration_buy_to_buyexit = 9999
            else:
                duration_watchlist_to_buy = 9999
                duration_buy_to_buyexit = 9999
            
            # Price differences
            if buy_signal:
                flat_start_date = pd.to_datetime(flat_period_start)
                watchlist_close = stock_df.loc[flat_start_date, 'close']
                price_diff_watchlist_buy = buy_close - watchlist_close
                if buyexit_signal:
                    profit_or_loss = buyexit_close - buy_close
                    profit_or_loss_percent = (profit_or_loss / buy_close) * 100
                else:
                    profit_or_loss = None
                    profit_or_loss_percent = None
            else:
                price_diff_watchlist_buy = None
                profit_or_loss = None
                profit_or_loss_percent = None

            # Create row
            row = {
                'tckr_symbol': tckr_symbol,
                'highest_high_flat': highest_high_flat,
                'flat_period_start': flat_period_start,
                'flat_period_end': flat_period_end,
                'watchlist_active': watchlist_active,
                'buy_signal': buy_signal,
                'buyexit_signal': buyexit_signal,
                'buy_date': buy_date,
                'buyexit_date': buyexit_date,
                'buy_close': buy_close,  # Added
                'buyexit_close': buyexit_close,  # Added
                'duration_watchlist_to_buy': duration_watchlist_to_buy,
                'duration_buy_to_buyexit': duration_buy_to_buyexit,
                'price_diff_watchlist_buy': price_diff_watchlist_buy,
                'profit_or_loss': profit_or_loss,
                'profit_or_loss_percent': profit_or_loss_percent,
                'trend_score': group['trend_score'].iloc[0],  # Add trend_score from the flat period group                
                'rsi_signal_flag': rsi_signal_flag,
                'exit_reason': buyexit_row['exit_reason'].iloc[0] if not buyexit_row.empty and 'exit_reason' in buyexit_row.columns else '',  # Safe access                            
                'price_move_pct_5d': buy_row['price_move_pct_5d'].iloc[0] if not buy_row.empty and 'price_move_pct_5d' in buy_row.columns else np.nan,
                'price_move_pct_10d': buy_row['price_move_pct_10d'].iloc[0] if not buy_row.empty and 'price_move_pct_10d' in buy_row.columns else np.nan,
                'price_move_pct_15d': buy_row['price_move_pct_15d'].iloc[0] if not buy_row.empty and 'price_move_pct_15d' in buy_row.columns else np.nan
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Filter out trades where duration from buy to exit is 50 days or more
    duration_watchlist_to_buy_threshold = 30
    summary_df = summary_df[~((summary_df['buy_signal'] == True) & (summary_df['duration_watchlist_to_buy'] > duration_watchlist_to_buy_threshold))]
    # NEW: Deduplicate by symbol and buy_date to remove rows where multiple flat periods lead to the same buy
    summary_df = summary_df.drop_duplicates(subset=['tckr_symbol', 'buy_date'], keep='first')
    return summary_df

def clamp(value, min_val=0, max_val=1):
    return max(min(value, max_val), min_val)

def trend_quality_score(data,
                        atr_lookback=5,
                        range_lookback=15,
                        efficiency_lookback=10):
    """
    Returns Trend Quality Score between 0 and 100.
    Calibrated to detect trending conditions (moderate momentum) rather than strong trends.
    Use score >40-50 as threshold to act on trade signals.
    """

    if len(data) < max(atr_lookback, range_lookback, efficiency_lookback) + 5:
        return 0

    latest = data.iloc[-1]
    score = 0

    # ===============================
    #1️⃣ ATR Expansion (0–20) - Full score at 30% ATR growth (detects moderate volatility increase)
    # ===============================
    atr_now = latest["atr14"]
    atr_then = data["atr14"].iloc[-atr_lookback]

    atr_ratio = atr_now / atr_then if atr_then > 0 else 0
    atr_score = clamp((atr_ratio - 1) / 0.2) * 20  # Changed from 0.5
    score += atr_score

    # ===============================
    # 2️⃣ EMA Separation (0–20) - Full score at 0.5% distance from EMA20 (detects basic trend separation)
    # ===============================
    ema20 = latest["ema20"]
    ema_dist = abs(latest["close"] - ema20) / ema20

    ema_score = clamp(ema_dist / 0.005) * 20  # Changed from 0.015
    score += ema_score

    # ===============================
    # 3️⃣ Range Expansion (0–20) - Full score at 2% range (detects moderate price swings)
    # ===============================
    recent_high = data["high"].iloc[-range_lookback:].max()
    recent_low = data["low"].iloc[-range_lookback:].min()
    range_ratio = (recent_high - recent_low) / latest["close"]

    range_score = clamp(range_ratio / 0.02) * 20  # Changed from 0.04
    score += range_score

    # ===============================
    # 4️⃣ Supertrend Slope (0–20) - Full score at 0.5% slope (detects directional momentum)
    # ===============================
    st_now = latest["supertrend"]
    st_then = data["supertrend"].iloc[-5]
    st_slope = abs(st_now - st_then) / latest["close"]

    st_score = clamp(st_slope / 0.005) * 20  # Changed from 0.01
    score += st_score

    # ===============================
    # 5️⃣ Price Efficiency (0–20) - Full score at 40% efficiency (detects somewhat directional movement)
    # ===============================
    price_move = abs(latest["close"] - data["close"].iloc[-efficiency_lookback])
    path_length = data["close"].iloc[-efficiency_lookback:].diff().abs().sum()

    efficiency = price_move / path_length if path_length > 0 else 0
    efficiency_score = clamp(efficiency / 0.4) * 20  # Changed from 0.6
    score += efficiency_score

    return round(score, 2)

def backtest_signals(summary_df, initial_capital=100000, position_size_pct=0.1):
    """
    Backtest the Supertrend-based swing signals from summary_df (long-only).
    
    Args:
        summary_df: DataFrame from summarize_signals (with buy/exit dates and prices).
        initial_capital: Starting capital.
        position_size_pct: Percentage of capital per trade.
    
    Returns:
        pandas.DataFrame: Trade log with per-trade details.
        (Metrics are printed separately for simplicity.)
    """
    capital = initial_capital
    trade_log = []
    peak_capital = initial_capital
    max_drawdown = 0
    returns = []
    
    for _, signal in summary_df.iterrows():
        if not signal['buy_signal']:
            continue  # Skip if no buy signal
        
        symbol = signal['tckr_symbol']
        entry_price = signal['buy_close']
        exit_price = signal['buyexit_close']
        
        if entry_price is None:
            continue  # Skip invalid entries
        
        # If no exit, assume exit at last close (placeholder; in real use, fetch from data)
        if exit_price is None:
            exit_price = entry_price * 1.05  # Placeholder: assume 5% gain; replace with actual logic
        
        # Position size
        position_size = capital * position_size_pct
        shares = position_size / entry_price
        
        # P&L
        pnl = (exit_price - entry_price) * shares
        capital += pnl
        returns.append(pnl / position_size)
        
        # Drawdown
        peak_capital = max(peak_capital, capital)
        drawdown = (peak_capital - capital) / peak_capital
        max_drawdown = max(max_drawdown, drawdown)
        
        # Log trade
        trade_log.append({
            'symbol': symbol,
            'entry_date': signal['buy_date'],
            'exit_date': signal['buyexit_date'] if signal['buyexit_signal'] else 'Open',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'profit_or_loss': signal.get('profit_or_loss', pnl / shares),  # Use existing or calculated
            'exit_reason': signal.get('exit_reason', '')  # Add this
        })
    
    # Metrics (printed separately)
    total_return = (capital - initial_capital) / initial_capital
    wins = sum(1 for trade in trade_log if trade['pnl'] > 0)
    win_rate = wins / len(trade_log) if trade_log else 0
    
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    print(f"Backtest Metrics:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Final Capital: {capital:.2f}")
    
    # Return trade log as DataFrame
    return pd.DataFrame(trade_log)
