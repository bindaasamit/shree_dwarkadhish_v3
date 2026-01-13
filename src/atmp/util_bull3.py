import pandas as pd
import pandas_ta as ta

def scan_bull3(df_daily):
    """
    Applies filters extracted from the scanner image.
    Expects a DataFrame with a DatetimeIndex and OHLCV columns.
    """
    # 1. Calculate Technical Indicators
    # Daily RSI(14) and RSI(2)
    df_daily['rsi14'] = ta.rsi(df_daily['close'], length=14)
    df_daily['rsi2'] = ta.rsi(df_daily['close'], length=2)
    
    # Daily Pivot Point (Standard calculation: (Prev High + Prev Low + Prev Close) / 3)
    df_daily['pivot'] = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3

    # 2. Define Filter Logic (applying to the most recent row)
    latest = df_daily.iloc[-1]
    prev = df_daily.iloc[-2]

    # Filter 1: Daily Rsi(14) Crossed above 59
    cond1 = (latest['rsi14'] > 59) and (prev['rsi14'] <= 59)
    
    # Filter 2: Daily Rsi(2) Greater than 40
    cond2 = latest['rsi2'] > 40
    
    # Filter 3: Daily Rsi(2) Less than 80
    cond3 = latest['rsi2'] < 80
    
    # Filter 4: Daily Close Greater than equal to Daily Pivot point
    cond4 = latest['close'] >= latest['pivot']
    
    # Filter 5: Daily Volume Greater than equal to 1 day ago Volume
    cond5 = latest['volume'] >= prev['volume']

    # Final Check: Stock must pass all filters
    is_bullish = all([cond1, cond2, cond3, cond4, cond5])
    
    return is_bullish

