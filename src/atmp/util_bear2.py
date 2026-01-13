import pandas_ta as ta

def scan_bear2(df_daily):
    # Resample daily data to weekly
    df_weekly = df_daily.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Daily Indicators
    df_daily['rsi14'] = ta.rsi(df_daily['close'], length=14)
    df_daily['sma20'] = ta.sma(df_daily['close'], length=20)
    df_daily['avg_price_3d'] = df_daily['close'].rolling(window=3).mean()
    
    # Weekly Indicator
    df_weekly['rsi14_w'] = ta.rsi(df_weekly['close'], length=14)
    
    curr, prev = df_daily.iloc[-1], df_daily.iloc[-2]
    curr_w = df_weekly.iloc[-1]
    
    cond1 = curr_w['rsi14_w'] < 50
    cond2 = curr['rsi14'] < 49 and prev['rsi14'] >= 49
    cond3 = curr['close'] <= curr['sma20']
    cond4 = prev['high'] > prev['sma20']
    cond5 = curr['close'] <= curr['avg_price_3d']
    
    return all([cond1, cond2, cond3, cond4, cond5])