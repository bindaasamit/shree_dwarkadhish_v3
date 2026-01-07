import pandas as pd
import pandas_ta as ta

def scan_bull1(df_daily):
    """
    Scans for bullish signals based on a set of technical indicators and conditions.
    The function evaluates daily and weekly RSI, moving averages, pivot points, volume, and price levels
    to determine if the latest candle meets bullish criteria.
    
    df_daily: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    Index should be a DatetimeIndex.
    
    Logic Steps:
    1. Validate input: Ensure at least 14 data points for reliable RSI calculations.
    2. Calculate daily indicators: RSI(14), SMA(20), and Pivot Point.
    3. Resample data to weekly and calculate weekly RSI(14) and RSI(4).
    4. Map weekly RSI values back to the daily DataFrame using forward fill.
    5. Extract the latest (current) and previous candle data.
    6. Check for invalid (NaN/None) values in key indicators; return False if any are invalid.
    7. Evaluate 7 bullish conditions based on crossings, thresholds, and comparisons.
    8. Return True only if all conditions are met.
    """
    #print("....Inside scan_bull1 function")
    # Step 1: Validate input data length
    # RSI requires at least 14 periods; insufficient data leads to unreliable results
    if len(df_daily) < 14:
        print("==> Not Enough Data, so skipping :- ",df_daily['tckr_symbol'].iloc[0] if 'tckr_symbol' in df_daily.columns else 'Unknown')
        #df_daily['bull1_flag'] = False
        #df_daily.reset_index(inplace=True)  # Reset index to make trade_dt a column
        return df_daily  # Return df_daily even if too short, with bull1_flag as False
    
    #print("....Calculating Daily Indicators")
    # Step 2: Calculate daily technical indicators
    # - RSI(14): Measures momentum on a 14-day scale
    # - SMA(20): 20-day simple moving average for trend assessment
    # - Pivot: Standard pivot point based on previous day's H/L/C for support/resistance
    df_daily['rsi_14'] = ta.rsi(df_daily['close'], length=14)
    df_daily['sma_20'] = ta.sma(df_daily['close'], length=20)
    df_daily['pivot'] = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3

    #print("....Resampling for Weekly Indicators")
    # Step 3: Resample daily data to weekly for broader trend analysis
    # Aggregate OHLCV using standard weekly logic (e.g., weekly close is Friday's close)
    logic = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_weekly = df_daily.resample('W').apply(logic)
    # Calculate weekly RSI(14) and RSI(4) for momentum on weekly scale
    df_weekly['rsi_w14'] = ta.rsi(df_weekly['close'], length=14)
    df_weekly['rsi_w4'] = ta.rsi(df_weekly['close'], length=4)
    
    #print("... Map weekly values back to the daily dataframe for final comparison")
    # Step 4: Align weekly indicators with daily data
    # Use forward fill to assign the latest weekly RSI to each daily row for comparison
    df_daily['weekly_rsi_14'] = df_weekly['rsi_w14'].reindex(df_daily.index, method='ffill')
    df_daily['weekly_rsi_4'] = df_weekly['rsi_w4'].reindex(df_daily.index, method='ffill')

    

    # Parse through each row starting from the second row (index 1) to have a previous row
    df_daily.reset_index(inplace=True)  # Reset index to make trade_dt a column
    #Convert trade_dt to string format YYYY-MM-DD for Excel output
    df_daily['trade_dt'] = pd.to_datetime(df_daily['trade_dt']).dt.strftime('%Y-%m-%d')
    # Initialize bull1_flag column
    df_daily['bull1_flag'] = False

    for i in range(1, len(df_daily)):
        curr = df_daily.iloc[i]
        prev = df_daily.iloc[i-1]

        
        # Check for invalid (NaN/None) values in key indicators
        if (pd.isna(curr['rsi_14']) or pd.isna(prev['rsi_14']) or
            pd.isna(curr['weekly_rsi_14']) or pd.isna(curr['weekly_rsi_4']) or
            pd.isna(curr['sma_20']) or pd.isna(curr['pivot']) or
            pd.isna(curr['volume']) or pd.isna(prev['volume'])):
            continue  # Skip if any NaN, flag remains False
        
        # Evaluate conditions
        cond1 = (curr['rsi_14'] > 59) and (prev['rsi_14'] <= 59)
        cond2 = curr['weekly_rsi_14'] > 55
        cond3 = curr['weekly_rsi_4'] > 60
        cond4 = curr['close'] >= curr['sma_20']
        cond5 = curr['volume'] >= prev['volume']
        cond6 = curr['close'] > curr['pivot']

        if curr['trade_dt'] == pd.Timestamp('2025-12-19'):
            print(f"Conditions on {curr['trade_dt']}: 1={cond1}, 2={cond2}, 3={cond3}, 4={cond4}, 5={cond5}, 6={cond6}")
        
        # Set flag if all conditions are met
        if all([cond1, cond2, cond3, cond4, cond5, cond6]):
            df_daily.loc[i, 'bull1_flag'] = True
    
    
    return df_daily  # Return the modified DataFrame

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