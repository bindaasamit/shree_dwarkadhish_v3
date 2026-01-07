import pandas_ta as ta

def scan_bear1(df_daily):
    # Resample daily data to weekly
    df_weekly = df_daily.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Daily Calculations
    df_daily['rsi14'] = ta.rsi(df_daily['close'], length=14)
    df_daily['wma13'] = ta.wma(df_daily['close'], length=13)
    
    # Pivot R1: (2 * Pivot) - Low
    pivot = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3
    df_daily['pivot_r1'] = (2 * pivot) - df_daily['low'].shift(1)
    
    # Bollinger Upper Band
    df_daily['bb_upper'] = ta.bbands(df_daily['close'], length=20, std=2)['BBU_20_2.0']
    
    # Weekly Calculation
    df_weekly['w_rsi14'] = ta.rsi(df_weekly['close'], length=14)
    
    curr, prev = df_daily.iloc[-1], df_daily.iloc[-2]
    curr_w = df_weekly.iloc[-1]
    
    conditions = [
        curr_w['w_rsi14'] < 49,
        curr['rsi14'] < 40 and prev['rsi14'] >= 40,
        curr['close'] < curr['wma13'],
        curr['close'] < curr['pivot_r1'],
        curr['volume'] >= prev['volume'],
        curr['close'] < curr['bb_upper']
    ]
    return all(conditions)

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

def scan_bear3(df_daily):
    # Indicators
    df_daily['rsi2'] = ta.rsi(df_daily['close'], length=2)
    df_daily['rsi14'] = ta.rsi(df_daily['close'], length=14)
    # Mid Bollinger Band is typically a 20-period SMA
    df_daily['mid_bb'] = ta.bbands(df_daily['close'], length=20, std=2)['BBM_20_2.0']
    
    curr, prev = df_daily.iloc[-1], df_daily.iloc[-2]
    
    cond1 = curr['rsi2'] < 60 and prev['rsi2'] >= 60
    cond2 = curr['rsi14'] < 40 and prev['rsi14'] >= 40
    cond3 = curr['close'] < curr['mid_bb']
    
    return all([cond1, cond2, cond3])

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