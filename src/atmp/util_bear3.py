import pandas_ta as ta

def scan_bear3(df):
    # Indicators
    df['rsi2'] = ta.rsi(df['close'], length=2)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    # Mid Bollinger Band is typically a 20-period SMA
    df['mid_bb'] = ta.bbands(df['close'], length=20, std=2)['BBM_20_2.0']
    
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    cond1 = curr['rsi2'] < 60 and prev['rsi2'] >= 60
    cond2 = curr['rsi14'] < 40 and prev['rsi14'] >= 40
    cond3 = curr['close'] < curr['mid_bb']
    
    return all([cond1, cond2, cond3])