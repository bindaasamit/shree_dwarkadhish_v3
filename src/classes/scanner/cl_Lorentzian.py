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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from enum import Enum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator, ADXIndicator

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_Reversal_RSI import (process_nse_stocks,
    read_nse_data,
    apply_hard_gates,
    calculate_pivot_points,
    get_monday_date,
    add_weekly_signals)

#------------------------------------------------------------------------------
###            Lorentzian Classifier
#------------------------------------------------------------------------------

def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def cci(high, low, close, length):
    tp = (high + low + close) / 3
    ma = tp.rolling(length).mean()
    md = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)


def adx(high, low, close, length):
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/length, adjust=False).mean()


def wt(df, channel_len=10, avg_len=11):
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    esa = hlc3.ewm(span=channel_len, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=channel_len, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d)
    return ci.ewm(span=avg_len, adjust=False).mean()


# ==========================================================
# LORENTZIAN DISTANCE
# ==========================================================

def lorentzian_distance(a, b):
    s = 0.0
    for x, y in zip(a, b):
        diff = abs(x - y)
        if math.isinf(diff) or math.isnan(diff):
            continue  # Skip inf/nan
        try:
            s += math.log(1 + diff)
        except ValueError:
            continue  # Skip if log fails
    return s


# ==========================================================
# KERNELS (WITH REGRESSION LEVEL)
# ==========================================================

def rational_quadratic(src, h, r, x):
    n = len(src)
    yhat = np.full(n, np.nan)

    for i in range(x, n):
        start = max(0, i - h + 1)
        idx = np.arange(start, i + 1)
        weights = (1 + ((i - idx)**2)/(2*r*h*h))**(-r)
        weights /= weights.sum()
        yhat[i] = np.sum(weights * src[idx])
    return yhat


def gaussian(src, h, x):
    n = len(src)
    yhat = np.full(n, np.nan)

    for i in range(x, n):
        start = max(0, i - h + 1)
        idx = np.arange(start, i + 1)
        weights = np.exp(-((i - idx)**2)/(2*h*h))
        weights /= weights.sum()
        yhat[i] = np.sum(weights * src[idx])
    return yhat


# ==========================================================
# EXACT PINE PARITY ENGINE
# ==========================================================

# ==========================================================
# EXACT PINE PARITY ENGINE (UPDATED)
# ==========================================================

def run_exact_lorentzian(df):

    df = df.copy().reset_index(drop=True)

    # FEATURES
    df['f1'] = rsi(df['close'], 14)
    df['f2'] = wt(df)
    df['f3'] = cci(df['high'], df['low'], df['close'], 20)
    df['f4'] = adx(df['high'], df['low'], df['close'], 20)
    df['f5'] = rsi(df['close'], 9)

    feature_cols = ['f1','f2','f3','f4','f5']
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # TRAIN LABEL (4 BAR LOOKAHEAD)
    y = np.where(
        df['close'].shift(-4) > df['close'], 1,
        np.where(df['close'].shift(-4) < df['close'], -1, 0)
    )

    predictions = []
    signal = []
    prev_signal = 0

    maxBarsBack = 2000
    neighborsCount = 8

    for bar in range(len(df)):

        if bar < maxBarsBack:
            predictions.append(0)
            signal.append(prev_signal)
            continue

        distances = []
        preds = []
        lastDistance = -1.0

        sizeLoop = min(maxBarsBack-1, bar-1)

        for i in range(sizeLoop):

            if i % 4 == 0:
                continue

            current = np.array(df.loc[bar, feature_cols].values, dtype=float)
            hist = np.array(df.loc[i, feature_cols].values, dtype=float)
            if pd.isna(current).any() or pd.isna(hist).any():
                continue

            d = lorentzian_distance(current, hist)

            if d >= lastDistance:
                lastDistance = d
                distances.append(d)
                preds.append(round(y[i]))

                if len(preds) > neighborsCount:
                    idx = round(neighborsCount*3/4)
                    lastDistance = distances[idx]
                    distances.pop(0)
                    preds.pop(0)

        prediction = sum(preds)
        predictions.append(prediction)

        if prediction > 0:
            prev_signal = 1
        elif prediction < 0:
            prev_signal = -1

        signal.append(prev_signal)

    df['prediction'] = predictions
    df['signal'] = signal

    # ================= KERNEL =================
    df['yhat1'] = rational_quadratic(df['close'].values, 8, 8, 25)
    df['yhat2'] = gaussian(df['close'].values, 6, 25)

    df['isBullishRate'] = df['yhat1'].shift(1) < df['yhat1']
    df['isBearishRate'] = df['yhat1'].shift(1) > df['yhat1']

    df['isBullishChange'] = df['isBullishRate'] & (~df['isBullishRate'].shift(1).fillna(False).infer_objects(copy=False))
    df['isBearishChange'] = df['isBearishRate'] & (~df['isBearishRate'].shift(1).fillna(False).infer_objects(copy=False))

    # ENTRY
    df['startLong'] = (df['signal'] == 1) & (df['signal'].diff() != 0)
    df['startShort'] = (df['signal'] == -1) & (df['signal'].diff() != 0)

    # DYNAMIC EXIT
    df['endLong'] = df['isBearishChange']
    df['endShort'] = df['isBullishChange']

    return df

# ==========================================================
# EXECUTION
# ==========================================================

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
        # Convert numeric columns to float to avoid dtype issues
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        stock_data[numeric_cols] = stock_data[numeric_cols].astype(float)
        

        try:
            # Process one stock at a time using the Lorentzian Classifier
            result_df = run_exact_lorentzian(stock_data)
            
            # Collect the result (DataFrame with signals)
            if not result_df.empty:
                all_signals_df.append(result_df)
        except Exception as e:
            print(f"Error processing {symbol}: {traceback.format_exc()}")
            continue
    
    # Optionally, combine all results into a single DataFrame if needed
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)
        #print(f"Total signals across all stocks: {len(combined_df[combined_df['buy_signal'] == True])}")
        # You can save or return combined_df as needed
    
    return all_signals_df  # Returns list of DataFrames, one per stock
