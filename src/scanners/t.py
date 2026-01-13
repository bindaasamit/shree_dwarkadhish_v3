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
    return all(variations <= tolerance)


def detect_strategy_signals(df, tolerance=0.001):
    """
    Detect buy signals based on the refined strategy:
    1. Supertrend must be GREEN (bullish)
    2. Within the GREEN period, find exactly 5 consecutive days where supertrend is flat
    3. Record highest high from those 5 flat period candles
    4. Generate buy signal when close > highest high (after the flat period)
    5. Only one watchlist entry per GREEN stretch (first flat period found)
    
    Parameters:
    df: DataFrame with OHLC data
    tolerance: Tolerance for flatness check (default 0.001 = 0.1%)
    
    Returns:
    DataFrame with signals added
    """
    df = calculate_supertrend(df.copy())
    
    # Initialize signal columns
    df['watchlist'] = False
    df['watchlist_active'] = False  # Currently active watchlist
    df['highest_high_flat'] = np.nan
    df['flat_period_start'] = ''
    df['flat_period_end'] = ''
    df['buy_signal'] = False
    df['signal_date'] = ''
    
    # Track state
    current_watchlist_high = None
    watchlist_active = False
    green_period_watchlist_added = False  # Flag to ensure only one watchlist per GREEN period
    
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
                        # Found a flat period!
                        flat_period_data = df.iloc[i:i+5]
                        watchlist_high = flat_period_data['high'].max()
                        
                        # Mark the flat period (exactly 5 days)
                        for j in range(i, i+5):
                            df.loc[df.index[j], 'watchlist'] = True
                            df.loc[df.index[j], 'highest_high_flat'] = watchlist_high
                            df.loc[df.index[j], 'flat_period_start'] = str(df.index[i])
                            df.loc[df.index[j], 'flat_period_end'] = str(df.index[i+4])
                        
                        # Activate watchlist for future candles
                        current_watchlist_high = watchlist_high
                        watchlist_active = True
                        green_period_watchlist_added = True  # Prevent multiple watchlists in same GREEN period
                        
                        print(f"✓ Watchlist added: Flat period from {df.index[i].date()} to {df.index[i+4].date()} (5 days)")
                        print(f"  Highest High = {watchlist_high:.2f}")
                        
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
                df.loc[df.index[i], 'signal_date'] = str(df.index[i])
                
                print(f"🎯 BUY SIGNAL on {df.index[i].date()}: Close {df['close'].iloc[i]:.2f} > Highest High {current_watchlist_high:.2f}")
                
                # Deactivate watchlist after buy signal
                watchlist_active = False
                current_watchlist_high = None
            
            # Deactivate watchlist if Supertrend turns RED
            elif current_direction == -1:
                print(f"⊗ Watchlist deactivated on {df.index[i].date()} - Supertrend turned RED")
                watchlist_active = False
                current_watchlist_high = None
                green_period_watchlist_added = False  # Reset for next GREEN period
        
        i += 1
    
    return df


def process_single_stock(symbol, stock_data, tolerance=0.001):
    """
    Process a single stock for strategy signals
    
    Parameters:
    symbol: Stock symbol
    stock_data: DataFrame with OHLC data
    tolerance: Flatness tolerance
    
    Returns:
    DataFrame with signals or None if error
    """
    try:
        # Prepare data
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.set_index('date').sort_index()
        
        # Process using detect_strategy_signals
        result_df = detect_strategy_signals(stock_data, tolerance=tolerance)
        
        # Add symbol column
        result_df['symbol'] = symbol
        
        # Count signals
        buy_count = len(result_df[result_df['buy_signal'] == True])
        watchlist_count = len(result_df[result_df['watchlist'] == True])
        
        print(f"✓ {symbol}: {watchlist_count // 5} flat periods found (5 days each), {buy_count} buy signals generated")
        
        return result_df
        
    except Exception as e:
        print(f"✗ Error processing {symbol}: {str(e)}")
        return None


# Example integration with your existing code structure
def process_nse_stocks_updated(sector, nifty_list, start_date, tolerance=0.001):
    """
    Process all stocks in NSE_DATA for strategy signals
    
    Parameters:
    sector: Sector name
    nifty_list: List of stock symbols
    start_date: Start date for data
    tolerance: Flatness tolerance (fixed 5-day periods)
    
    Returns:
    List of DataFrames with signals for each stock
    """
    # Read data (use your existing read_nse_data function)
    print("=" * 80)
    print("STEP 1: Reading NSE Data")
    print("=" * 80)
    
    stocks_df = read_nse_data(nifty_list, start_date)
    stocks_df = stocks_df.rename(columns={
        'trade_dt': 'date',
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'closing_price': 'close',
        'total_trading_volume': 'volume'
    })
    stocks_df = stocks_df[['date', 'tckr_symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    # Ensure High is highest and Low is lowest
    stocks_df['high'] = stocks_df[['open', 'high', 'close']].max(axis=1)
    stocks_df['low'] = stocks_df[['open', 'low', 'close']].min(axis=1)
    
    print("=" * 80)
    print("STEP 2: Processing Stocks for Flat Period Detection")
    print("=" * 80)
    
    all_signals_df = []
    
    for idx, symbol in enumerate(nifty_list, 1):
        print(f"\n[{idx}/{len(nifty_list)}] Processing {symbol}...")
        
        stock_data = stocks_df[stocks_df['tckr_symbol'] == symbol].copy()
        
        if stock_data.empty:
            print(f"  ⊗ No data available for {symbol}")
            continue
        
        # Process stock
        result_df = process_single_stock(symbol, stock_data, flat_days, tolerance)
        
        if result_df is not None:
            all_signals_df.append(result_df)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)
        total_buy_signals = len(combined_df[combined_df['buy_signal'] == True])
        total_watchlists = len(combined_df[combined_df['watchlist'] == True])
        
        print(f"Total stocks processed: {len(all_signals_df)}")
        print(f"Total flat periods found: {total_watchlists}")
        print(f"Total buy signals generated: {total_buy_signals}")
    
    return all_signals_df