import pandas as pd
import pandas_ta as ta

def scan_bull4(df_daily):
    """
    Applies filters specifically from the bull4.png image.
    df_daily: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    """
    # 1. Calculate Technical Indicators
    df_daily['rsi14'] = ta.rsi(df_daily['close'], length=14)
    df_daily['rsi2'] = ta.rsi(df_daily['close'], length=2)
    
    # Daily Pivot Point: (Prev High + Prev Low + Prev Close) / 3
    df_daily['pivot'] = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3
    
    # Daily Avg_Price_3Days: Simple Moving Average of Close for 3 periods
    df_daily['avg_price_3d'] = ta.sma(df_daily['close'], length=3)

    # 2. Get data for the current and previous candles
    curr = df_daily.iloc[-1]
    prev = df_daily.iloc[-2]

    # 3. Define Scanner Logic
    # Daily Rsi(14) Crossed above 40
    cond1 = (curr['rsi14'] > 40) and (prev['rsi14'] <= 40)
    
    # Daily Rsi(2) Crossed above 30
    cond2 = (curr['rsi2'] > 30) and (prev['rsi2'] <= 30)
    
    # Daily Rsi(2) Less than 80
    cond3 = curr['rsi2'] < 80
    
    # Daily Close Greater than Daily Pivot point
    cond4 = curr['close'] > curr['pivot']
    
    # Daily Close Greater than equal to Daily Avg_Price_3Days
    cond5 = curr['close'] >= curr['avg_price_3d']

    # Final Check
    passes_all = all([cond1, cond2, cond3, cond4, cond5])
    
    return passes_all

