
#####
stock_id = ["tckr_symbol", "trade_dt"]
volume_20day_period = 20
volume_26day_period = 26
adx_threshold = 25
rsi_bullish_threshold = 30
rsi_bearish_threshold = 70
 
#####
holding_period=7
volatility_threshold = 0.75 
buy_signal_threshold = 12
sell_signal_threshold= 12

initial_capital=1_000_000 
risk_per_trade=0.02
top_n=2

# Change ranking_priority to test different selection strategies
ranking_priority = ['filter_score', 'atr_pct', 'vol_spike']  # Default
    
    # ['filter_score', 'atr_pct', 'vol_spike'] → Trend confirmation first
    # ['atr_pct', 'filter_score', 'vol_spike'] → Volatility potential first
    # ['vol_spike', 'filter_score', 'atr_pct'] → Volume surge priority 
#ranking_priority=None

custom_weights = {
    # Ichimoku - Strong trend structure
    'kumo_twist_bullish': 3,                     # Early trend reversal
    'price_kumo_bullish_breakout': 4,            # Strong breakout
    'tenkan_kijun_bullish_crossover': 3,         # Strong continuation
    'price_kijun_bullish_crossover': 3,          # Key support test
    'price_kumo_bearish_breakout': 4,            # Strong breakdown
    'price_kumo_bearish_breakdown': 3,           # Follow-through
    'tenkan_below_kijun': 2,                              # Bearish warning
    'chikou_above': 2,                                    # Confirmation of bullish trend
    'chikou_below': 2,                                    # Confirmation of bearish trend

    # EMA structure
    'ema_bullish': 3,                                     # Strong uptrend alignment
    'ema_bearish': 3,                                     # Strong downtrend alignment

    # Momentum Indicators
    'rsi_bullish': 2,                                     # Momentum confirmation
    'rsi_bearish': 2,
    'macd_bullish': 3,                                    # Strong momentum alignment
    'macd_bearish': 3,

    # Price action (yesterday close as minor filter)
    'close_above_yday': 1,                                
    'close_below_yday': 1,

    # Breakouts
    'bollinger_breakout': 3,                              # High-probability expansion
    'bollinger_breakdown': 3,
    'donchian_breakout_buy': 2,                           # Trend continuation
    'donchian_breakout_sell': 2,

    # Overbought/Oversold
    'williams_oversold': 1,                               # Weak standalone, good confirmation
    'williams_overbought': 1,
    'stochrsi_buy': 1,
    'stochrsi_sell': 1,

    # Trend continuation
    'supertrend_buy': 3,                                  # Strong confirmation
    'supertrend_sell': 3,
    'heikin_ashi_buy': 1,
    'heikin_Ashi_sell': 1,

    # Volume/OBV/CCI
    'obv_trend': 2,                                       # Volume confirmation
    'cci_buy': 1,
    'cci_sell': 1,
    'vol_spike': 2,                                       # Strong participation by institutions

    # Strength Indicators
    'adx_strong': 3,                                      # Trend strength key factor
    'atr_high': 1,                                        # Volatility consideration

    # Pivot points
    'pivot_above': 1,
    'pivot_below': 1
}


"""
round_decimals = {'stop_loss_buy' : 0, 'stop_loss_sell' : 0,  
                                'open_price' : 2, 'high_price' : 2,	'low_price' : 2,	
                                'last_price' : 2, 'closing_price' : 2,	
                                'prev_closing_price' : 2, 'total_transfer_value' : 2,
                                'tenkan' : 2,	'kijun' : 2,	
                                'senkou_a' : 2, 'senkou_b' : 2,	
                                'chikou_span' : 2, 'ema5' : 2,	
                                'ema13' : 2, 'ema26' : 2,	
                                'rsi' : 2, 'macd' : 2,	
                                'macd_signal' : 2, 'adx' : 2,	
                                'tr' : 2,	'atr' : 2,	
                                'atr14' : 2, 'entry_price' : 2,	
                                'stop_loss_atr' : 2, 'stop_loss_kijun' : 2,	
                                'entry_price_sell' : 2, 'stop_loss_atr_sell' : 2, 'stop_loss_kijun_sell' : 2,
                                'sma20' : 2, 'stddev20' : 2,'upper_band' : 2, 'lower_band' : 2,
                                'williamsr' : 2, 'donchian_upper' : 2, 'donchian_lower' : 2,
                                'pivot' : 2, 'r1' : 2, 's1' : 2, 
                                'ha_open' : 2, 'ha_close' : 2,
                                'obv' : 2, 'cci' : 2, 'return_std' : 2, 'std_rank' : 2, 'vol_rank' : 2, 'atr_rank' : 2, 
                                'price_range' : 2, 	'range_rank' : 2, 
                                'bb_width' : 2, 'bb_rank' : 2, 
                                'volatility_score' : 2, 
                                'entry_price' : 2, 'stop_loss_atr' : 2, 'stop_loss_kijun' : 2, 'entry_price_sell' : 2, 
                                'stop_loss_atr_sell' : 2, 'stop_loss_kijun_sell' : 2,
                                'ema_alignment_score' : 2, 'obv_ema' : 2, 'ema_trend_strength' : 2
                                }
"""
"""
round_ema_num_vars = {'ema5' : 2, 'ema13' : 2, 'ema26' : 2, 
                      'ema_ribbon_width' : 2, 'ema_ribbon_width_pct' : 0, 'ema_ribbon_avg' : 2, 'ema_alignment_score' : 2,
                      'ema_slope_5' : 2, 'ema_slope_13' : 2, 'ema_slope_26' : 2, 'ema_avg_slope' : 2, 'ema_slope_alignment' : 0, 'ema_trend_strength' : 2,
                      'rsi' : 2, 'stoch_k' : 2, 'stoch_d' : 2, 'macd' : 2, 'macd_signal' : 2, 'macd_histogram' : 2,
                      'volume_ratio' : 2, 'atr' : 2, 'atr_pct' : 2, 
                      'bb_middle' : 2, 'bb_std' : 2, 'bb_upper' : 2, 'bb_lower' : 2, 'bb_width' : 2, 'bb_width_pct' : 2, 'bb_position' : 2,
                      'adx' : 2, 'plus_di' : 2, 'minus_di' : 2, 'pivot' : 2, 
                      'resistance_1' : 2, 'support_1' : 2, 'resistance_2' : 2, 'support_2' : 2, 'signal_quality_enhanced' : 2,
                      'buy_score_normalized' : 2, 'sell_score_normalized' : 2}
"""

key_cols = ['tckr_symbol', 'trade_dt']

ichimoku_cols = ['tenkan', 'kijun','senkou_a', 'senkou_b', 'chikou_span']
ema_cols = ['ema5', 'ema13', 'ema26']
rsi_cols = ['rsi']               
macd_cols= ['macd']
adx_cols = ['tr', 'atr', 'adx']
vol_cols = ['avg_volume']        
bollinger_cols =['sma20', 'stddev20','upper_band', 'lower_band']     
supertrend_cols = ['atr14', 'supertrend']
stockrsi_cols = ['stoch_rsi'] 
donchian_cols = ['donchian_upper', 'donchian_lower']
pivot_cols = ['pivot', 'r1', 's1']
heikin_ashi_cols =['ha_open', 'ha_close']
volatility_cols = ['return_std', 'std_rank' , 'vol_rank', 'atr_rank', 'price_range', 'range_rank', 'bb_width','bb_rank']
obv_cols =['obv']
cci_cols =['cci']
stop_loss_cols = ['stop_loss_atr','stop_loss_kijun', 'stop_loss_atr_sell' , 'stop_loss_kijun_sell']


'''        
price_change_cols = ["price_after_3_days", "price_diff_after_3_days"
                ,"price_after_5_days", "price_diff_after_5_days"
                ,"price_after_7_days", "price_diff_after_7_days"
                ,"price_after_10_days", "price_diff_after_10_days"]
'''




#########################################################################
#                        Common Signals                                 #
#########################################################################
#action_signals = ['action_summary','option_strategy']
action_signals = ['action_summary']                   
volatility_signals = ['high_volatility','volatility_score'] 
macd_signals = ['macd_signal']
vol_signals = ['vol_spike']
williamsr_signals = ['williamsr']        
obv_signals =['obv_trend']

#########################################################################
#                        Buy Signals                                   #
#########################################################################
buy_signals = ['signal_score_buy','buy_action','buy_signal_age','resistance','entry_price','stop_loss_buy']
ichimoku_buy_signals = ['kumo_twist_bullish',
        'price_kumo_bullish_breakout',
        'tenkan_kijun_bullish_crossover',
        'price_kijun_bullish_crossover',
        'price_kumo_breakout_strategy_buy']
bollinger_buy_signals = ['bollinger_breakout_buy']
supertrend_buy_signals = ['supertrend_buy']
stockrsi_buy_signals = ['stochrsi_buy']
donchian_buy_signals = ['donchian_breakout_buy']
pivot_bug_signals = ['above_pivot']
heikin_ashi_buy_signals = ['heikin_ashi_buy']
cci_buy_signals =['cci_buy']

boost_signal = ['signal_score_buy_base']
simulate_exit_signal = ['exit_signal', 'tp_level', 'sl_level']

other_buy_signals = bollinger_buy_signals + supertrend_buy_signals + stockrsi_buy_signals + donchian_buy_signals + pivot_bug_signals + heikin_ashi_buy_signals + cci_buy_signals + volatility_signals + macd_signals + williamsr_signals + obv_signals + boost_signal + simulate_exit_signal + simulate_exit_signal

#########################################################################
#                        Sell Signals                                   #
#########################################################################
sell_signals = ['signal_score_sell','sell_action','sell_signal_age','support','entry_price_sell','stop_loss_sell']
ichiomku_sell_signals = ['price_kumo_bearish_breakout',
        'price_kumo_bearish_breakdown',
        'price_kumo_strategy_sell']       
bollinger_sell_signals = ['bollinger_breakout_sell']
supertrend_sell_signals = ['supertrend_sell']
stockrsi_sell_signals = ['stochrsi_sell']
donchian_sell_signals = ['donchian_breakout_sell']
pivot_sell_signals = ['below_pivot']
heikin_ashi_sell_signals =['heikin_ashi_sell']
cci_sell_signals =['cci_sell']

other_sell_signals = bollinger_sell_signals + supertrend_sell_signals + stockrsi_sell_signals + donchian_sell_signals + pivot_sell_signals + heikin_ashi_sell_signals + cci_sell_signals + volatility_signals + macd_signals + williamsr_signals + obv_signals
            
# Three Dataframe Lists
final_df_list = key_cols + action_signals + volatility_signals + buy_signals + sell_signals + ichimoku_buy_signals + bollinger_buy_signals + supertrend_buy_signals + stockrsi_buy_signals + donchian_buy_signals + pivot_bug_signals + heikin_ashi_buy_signals + cci_buy_signals + ichiomku_sell_signals + bollinger_sell_signals + supertrend_sell_signals + stockrsi_sell_signals + donchian_sell_signals + pivot_sell_signals + heikin_ashi_sell_signals + cci_sell_signals + macd_signals + vol_signals + williamsr_signals + obv_signals + cci_cols + stop_loss_cols + raw_cols + ichimoku_cols + ema_cols + rsi_cols + macd_cols + adx_cols + vol_cols + bollinger_cols + supertrend_cols + stockrsi_cols + donchian_cols + pivot_cols + heikin_ashi_cols + volatility_cols + obv_cols
buy_df_list = key_cols + action_signals + volatility_signals + buy_signals + ichimoku_buy_signals + bollinger_buy_signals + supertrend_buy_signals + stockrsi_buy_signals + donchian_buy_signals + pivot_bug_signals + heikin_ashi_buy_signals + cci_buy_signals + macd_signals + vol_signals + williamsr_signals + obv_signals + cci_cols + stop_loss_cols + raw_cols + ichimoku_cols + ema_cols + rsi_cols + macd_cols + adx_cols + vol_cols + bollinger_cols + supertrend_cols + stockrsi_cols + donchian_cols + pivot_cols + heikin_ashi_cols + volatility_cols + obv_cols
sell_df_list = key_cols + action_signals+ volatility_signals + sell_signals + ichiomku_sell_signals + bollinger_sell_signals + supertrend_sell_signals + stockrsi_sell_signals + donchian_sell_signals + pivot_sell_signals + heikin_ashi_sell_signals + cci_sell_signals + macd_signals + vol_signals + williamsr_signals + obv_signals + cci_cols + stop_loss_cols + raw_cols + ichimoku_cols + ema_cols + rsi_cols + macd_cols + adx_cols + vol_cols + bollinger_cols + supertrend_cols + stockrsi_cols + donchian_cols + pivot_cols + heikin_ashi_cols + volatility_cols + obv_cols
