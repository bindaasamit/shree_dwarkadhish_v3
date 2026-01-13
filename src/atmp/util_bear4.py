import pandas_ta as ta

def scan_bear4(df_daily):
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
    df_daily['wma13'] = ta.wma(df_daily['close'], length=13)
    df_daily['pivot'] = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3
    df_daily['bb_upper'] = ta.bbands(df_daily['close'], length=20, std=2)['BBU_20_2.0']
    
    # Weekly Indicators
    df_weekly['rsi14'] = ta.rsi(df_weekly['close'], length=14)
    
    curr, prev = df_daily.iloc[-1], df_daily.iloc[-2]
    curr_w = df_weekly.iloc[-1]
    
    conditions = [
        curr_w['rsi14'] < 49,
        curr['rsi14'] < 45 and prev['rsi14'] >= 45,
        curr['close'] < curr['wma13'],
        curr['close'] < curr['pivot'],
        curr['volume'] >= prev['volume'],
        curr['close'] < curr['bb_upper'],
        prev['rsi14'] < 50
    ]
    return all(conditions)