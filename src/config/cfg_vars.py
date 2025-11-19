#----------------------------------------------------------------------------------------------------------------------
#                                               FILE PATHS
#----------------------------------------------------------------------------------------------------------------------

nifty_file = 'D:/myCourses/shree_dwarkadhish_v3/data/input/nifty.xlsx'
input_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/daily_stocks_data/'
db_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/db/'

json_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/json/'
ema_results_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/scanner/ema/'
ich_results_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/scanner/ich/'
results_backtest_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest_input/'
scanner_log_path = "D:/myCourses/shree_dwarkadhish_v3/data/output/logs/stock_recommender.log"
backtester_log_path = "D:/myCourses/shree_dwarkadhish_v3/data/output/logs/backtrader.log"

backtest_ema_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest/ema/'
backtest_ich_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest/ich/'

## Database and Table Names
db_name = "shree_dwarkadhish.db"
inp_table_name = "historical_stocks"
output_table_name = "signal_results"

#----------------------------------------------------------------------------------------------------------------------
#                                               LOADING VARIABLES
#----------------------------------------------------------------------------------------------------------------------
selected_columns= ['tckr_symbol','trade_dt','scty_series','open_price',
                   'high_price','low_price','closing_price','last_price',
                   'prev_closing_price','total_trading_volume','total_transfer_value']

basic_cols = ["scty_series", "open_price", "high_price"
                ,"low_price", "last_price", "closing_price", "prev_closing_price"
                ,"total_trading_volume", "total_transfer_value"]

raw_cols = ['scty_series', 'open_price', 'high_price','low_price', 'last_price', 
            'closing_price','prev_closing_price',
            'total_trading_volume', 'total_transfer_value']        

column_mapping = {
                'TckrSymb' : 'tckr_symbol',
                'TradDt' : 'trade_dt',  
                'SctySrs' : 'scty_series',  
                'OpnPric' : 'open_price',
                'HghPric' : 'high_price',
                'LwPric'  : 'low_price', 
                'ClsPric' : 'closing_price',     
                'LastPric': 'last_price',
                'PrvsClsgPric' : 'prev_closing_price',
                'TtlTradgVol' : 'total_trading_volume',
                'TtlTrfVal' : 'total_transfer_value'
            }

select_hist_query = "SELECT tckr_symbol, trade_dt, scty_series, open_price, high_price, low_price, closing_price,last_price, prev_closing_price, " \
"total_trading_volume, total_transfer_value FROM historical_stocks"

insert_query = """
            INSERT INTO historical_stocks (
                tckr_symbol, trade_dt, scty_series, open_price, high_price,
                low_price, closing_price, last_price, prev_closing_price,
                total_trading_volume, total_transfer_value
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tckr_symbol, trade_dt) DO NOTHING;          """


#----------------------------------------------------------------------------------------------------------------------
#                                               EMA VARIABLES
#----------------------------------------------------------------------------------------------------------------------

warning_indicators = ['ema_sideways_market', 
                      'ema_whipsaw_risk', 
                      'ema_ribbon_width']

only_signals = ['profit_n_loss_pct', 
                'signal_action', 
                'signal_warning' , 
                'signal_limiter', 
                'signal_reason', 
                'signal_confidence', 
                'signal_type'] 

bullish_indicators = ['ema_golden_cross_filtered',
                      'ema_golden_cross', 
                      'ema_slopes_aligned_buy', 
                      'rsi_bullish', 
                      'macd_bullish',
                      'adx_bullish', 
                      'bullish_volume',
                      'ema_trend_strength',
                      'ema_bullish_alignment', 
                      'ema_alignment_score', 
                      'ema_slope_5', 'ema_slope_13', 'ema_slope_26', 'ema_avg_slope']

bearish_indicators = ['ema_death_cross_filtered', 
                      'ema_death_cross', 
                      'ema_slopes_aligned_sell', 
                      'rsi_bearish', 
                      'macd_bearish',
                      'adx_bearish', 
                      'bearish_volume', 
                      'ema_trend_strength',
                      'ema_bearish_alignment', 
                      'ema_alignment_score', 
                      'ema_slope_5', 
                      'ema_slope_13', 
                      'ema_slope_26', 
                      'ema_avg_slope']

key_fields = ['tckr_symbol', 'trade_dt']

input_fields = ['scty_series', 'open_price', 'high_price', 'low_price', 
                'closing_price', 'last_price', 'prev_closing_price', 
                'total_trading_volume', 'total_transfer_value', 
                'ema5', 'ema13', 'ema26', 'rsi', 'adx', 'atr', 'volatility_regime']

ema_ordered_cols = key_fields + warning_indicators + only_signals + bullish_indicators + bearish_indicators + input_fields


###. EMA_BUY_VARS
ema_buy_vars = key_fields + warning_indicators + only_signals + bullish_indicators + input_fields
ema_sell_vars = key_fields + warning_indicators + only_signals + bearish_indicators + input_fields
ema_noaction_vars = ema_ordered_cols

round_ema_num_vars = {
    'profit_n_loss_pct':2, 'ema5':2, 'ema13':2, 
    'ema26':2, 'ema_ribbon_width':2, 'ema_ribbon_width_pct':0, 
    'ema_ribbon_avg':2, 'ema_slope_5':2, 'ema_slope_13':2, 
    'ema_slope_26':2, 'ema_avg_slope':2, 'ema_alignment_score':2, 
    'ema_trend_strength':2
}
#----------------------------------------------------------------------------------------------------------------------
#                                               ICHIMOKU VARIABLES
#----------------------------------------------------------------------------------------------------------------------


ich_warning_indicators = ['tradeable', 'trend_quality']

ich_warning_indicators_additional = ['sideways_market', 'weak_trend', 'choppy_market', 'false_breakout_risk'] 

only_signals = ['signal_action', 
                'signal_warning', 
                'signal_limiter', 
                'signal_reason', 
                'signal_confidence', 
                'signal_type'] 

ich_bullish_indicators = ['kumo_twist_bullish_filtered','price_kumo_bullish_breakout_filtered', 
                          'tenkan_kijun_bullish_crossover_filtered','price_kijun_bullish_crossover_filtered']

ich_bearish_indicators = ['price_kumo_bearish_breakout_filtered', 'price_kumo_bearish_breakdown_filtered']

filtering_metrics = ['kumo_thickness', 'kumo_thickness_pct', 'tk_separation', 'tk_separation_pct',
                     'price_kumo_distance', 'price_kumo_distance_pct', 'chikou_clear', 'price_range_pct', 'avg_range_pct']

key_fields = ['tckr_symbol', 'trade_dt']

ich_input_fields = ['tenkan', 'kijun','senkou_a', 'senkou_b', 'chikou_span', 'kumo_top', 'kumo_bottom', 'prev_closing_price',
                    'scty_series', 'open_price', 'high_price', 'low_price', 'closing_price', 'last_price', 'prev_closing_price', 
                    'total_trading_volume', 'total_transfer_value']


ich_ordered_cols = key_fields + ich_warning_indicators + only_signals + ich_bullish_indicators + ich_bearish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional

ich_buy_vars = key_fields + ich_warning_indicators + only_signals + ich_bullish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional
ich_sell_vars = key_fields + ich_warning_indicators + only_signals + ich_bearish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional
ich_noaction_vars = ich_ordered_cols

round_ich_num_vars = {'price_kumo_distance' : 2, 'price_kumo_distance_pct' : 2, 'price_range_pct' : 2, 'avg_range_pct' : 2, 
                      'trend_quality' : 2, 'tenkan' : 2, 'kijun' : 2, 'senkou_a' : 2,	'senkou_b' : 2, 'chikou_span' : 2,
                      'kumo_top' : 2, 'kumo_bottom' : 2,	'kumo_thickness' : 2, 'kumo_thickness_pct' : 2,	
                      'tk_separation' : 2, 'tk_separation_pct' : 2, 'trend_quality' : 2}