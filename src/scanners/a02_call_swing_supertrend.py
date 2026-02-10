#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V5')  # Add project root to path
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta, datetime
import concurrent.futures
import yfinance as yf
import pretty_errors
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from enum import Enum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_SuperTrend_Strategy import (calculate_supertrend, 
    check_supertrend_flat,
    detect_buy_signals,
    read_nse_data,
    process_nse_stocks,
    summarize_signals,
    clamp,
    trend_quality_score,
    get_weekly_trend,
    backtest_signals)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error


#---------------------------------------------------------------------------------------------------------------------
#                                            MAIN WORKFLOW
#--------------------------------------------------------------------------------------------------------------------- 
# Example usage
if __name__ == "__main__":
    sector ='all'  # 'test' / 'nifty100' / 'fno'/ 'fno_movers' / 'small_mid' / 'all'
    start_date = '2024-01-01'
    #filter_date = '2025-11-01'
    filter_date = start_date

    # Before saving the files, generate timestamp:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    swing_path1 = cfg_vars.swing_model_dir1 + f'{sector}_data_{timestamp}.xlsx'
    swing_path2 = cfg_vars.swing_model_dir1 + f'{sector}_signals_{timestamp}.xlsx'
    summary_path = cfg_vars.swing_model_dir1 + f'{sector}_summary_{timestamp}.xlsx'
    backtest_path = cfg_vars.swing_model_dir1 + f'{sector}_backtest_results_{timestamp}.xlsx'

    all_stocks = cfg_nifty.nifty100 + cfg_nifty.nifty_fno + cfg_nifty.nifty_mid_small_caps + cfg_nifty.mono_duopoly_stocks
    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'nifty100': nifty_list = list(set(cfg_nifty.nifty_50 + cfg_nifty.nifty_next_50))
        case 'fno' : nifty_list = cfg_nifty.nifty_fno
        case 'fno_movers_poly': nifty_list = list(set(cfg_nifty.top_30_fno_swing + cfg_nifty.nifty_movers + cfg_nifty.mono_duopoly_stocks))
        case 'small_mid': nifty_list = cfg_nifty.nifty_mid_small_caps
        case 'all': nifty_list = list(set(all_stocks))  
        case _: print("Invalid Sector")    
        
    
    # Sort the list alphabetically
    nifty_list = sorted(nifty_list)

    ### Step1. Generate all Data points
    all_signals_df = process_nse_stocks(sector, nifty_list, start_date)
    combined_df = pd.concat(all_signals_df, ignore_index=False)
    
    ### Step2. Filter to include only rows where flat_period_start, buy_date, or buyexit_date is populated
    combined_df = combined_df.reset_index()
    combined_low_df = combined_df[['tckr_symbol','date','low','volume']]
    combined_filtd_df = combined_df[
        (combined_df['flat_period_start'] != '') | 
        (combined_df['buy_date'] != '') | 
        (combined_df['buyexit_date'] != '')]
    
    # Reset index to make date a column
    #combined_filtd_df = combined_filtd_df.reset_index()
    comb_ord_cols = ['tckr_symbol',	'date',  'watchlist',	'watchlist_active',	'highest_high_flat',	'flat_period_start',	'flat_period_end',	
    'buy_signal',	'buy_date',	'position_active',	'buyexit_signal',	'buyexit_date', 'trend_score',	'rsi', 'rsi_signal_flag',
    'supertrend',	'supertrend_direction', 'final_ub',	'final_lb', 'atr', 'volatility_signal', 'pivot_signal','gobuy_flag',
    'open',	'high',	'low',	'close', 'volume', 'h-l', 'h-pc', 'l-pc', 'tr',	 'atr14', 'basic_ub', 'basic_lb', 'ema20', 'ema50','symbol','exit_reason',
    'price_move_pct_5d', 'price_move_pct_10d', 'price_move_pct_15d','supertrend_trailing']
    combined_filtd_df = combined_filtd_df[comb_ord_cols]

    # Recalculate totals after filtering
    total_buy_signals = len(combined_filtd_df[combined_filtd_df['buy_signal'] == True])
    total_watchlists = len(combined_filtd_df[combined_filtd_df['watchlist'] == True])
    print(f"Total stocks processed: {len(all_signals_df)}")
    print(f"Total flat periods found (after filtering): {total_watchlists // 5}")  # Assuming 5 rows per period
    print(f"Total buy signals generated (after filtering): {total_buy_signals}")
    
    ###Step3. Summarize Signals
    summary_df = summarize_signals(combined_filtd_df, combined_low_df,nifty_list)
    summary_df['weekly_trend'] = None

    summary_cols = ['tckr_symbol',	'buy_date',	'duration_buy_to_buyexit','buyexit_date',	'buy_signal', 'gobuy_flag',	'buyexit_signal',
	'rsi_signal_flag',	'volatility_signal',	'volume_flag',	'pivot_signal',	'trend_score',	'weekly_trend',	
    'price_move_pct_5d',	'pct_to_lowest_5d',	'flat_period_start',	'flat_period_end',
    'duration_watchlist_to_buy',	'price_diff_watchlist_buy',
    'profit_or_loss_percent',	'highest_high_flat',	'exit_reason',	'buy_close',	'buyexit_close',
    'price_move_pct_10d',	'price_move_pct_15d',	'lowest_low_5d',	'lowest_low_10d',	
    'pct_to_lowest_10d',	'lowest_low_15d',	'pct_to_lowest_15d', 'profit_or_loss','supertrend_trailing']  # Added buy_close and buyexit_close
    summary_df = summary_df[summary_cols]
    #Filterout all records where duration_watchlist_to_buy is more than 20 days
    summary_df = summary_df[summary_df['duration_watchlist_to_buy'] <=20]

    ###Step4. Get Weekly Trend Data for the Identified Buy Signals Only
    #weekly_data_path = cfg_vars.weekly_data_dir + f'stocks_weekly_data.xlsx'
    #summary_df1 = get_weekly_trend(summary_df,weekly_data_path)    

    ###Step4 Save Results
    summary_df.to_excel(summary_path, index=False)
    
    #combined_df.reset_index().to_excel(swing_path1, index=False)
    #combined_filtd_df.reset_index().to_excel(swing_path2, index=False)  # Include date as a column
    #Backtest Output Summary
    #results_df = backtest_signals(summary_df)
    #results_df.to_excel(backtest_path, index=False)
    
    print(f"Summary DataFrame created with {len(summary_df)} rows!")