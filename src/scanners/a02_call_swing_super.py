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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_SuperTrend_Strategy import (calculate_supertrend, 
    check_supertrend_flat,
    detect_strategy_signals,
    read_nse_data,
    process_nse_stocks,
    summarize_signals,
    clamp,
    trend_quality_score,
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
    sector ='fno_sects'  # 'test' / 'movers' / 'fno_sects' / 'small_mid'
    start_date = '2025-01-01'
    #filter_date = '2025-11-01'
    filter_date = start_date

    swing_path1 = cfg_vars.swing_model_dir1 + f'{sector}_data.xlsx'
    swing_path2 = cfg_vars.swing_model_dir1 + f'{sector}_signals.xlsx'
    summary_path = cfg_vars.swing_model_dir1 + f'{sector}_summary.xlsx'
    backtest_path = cfg_vars.swing_model_dir1 + f'{sector}_backtest_results.xlsx'

    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'small_mid': nifty_list = cfg_nifty.nifty_mid_small_caps
        case 'movers': nifty_list = cfg_nifty.nifty_movers
        case 'fno_sects': nifty_list = list(set(cfg_nifty.nifty_fno + cfg_nifty.nifty_sectoral))
        case 'cash': nifty_list = cfg_nifty.nifty_cash  
        case _: print("Invalid Sector")
    
    
    # Sort the list alphabetically
    nifty_list = sorted(nifty_list)

    all_signals_df = process_nse_stocks(sector, nifty_list, start_date)
    combined_df = pd.concat(all_signals_df, ignore_index=False)
    
    # Filter to include only rows where flat_period_start, buy_date, or buyexit_date is populated
    combined_df.reset_index().to_excel(swing_path1, index=False)
    combined_df = combined_df[
        (combined_df['flat_period_start'] != '') | 
        (combined_df['buy_date'] != '') | 
        (combined_df['buyexit_date'] != '')
    ]
    
    # Reset index to make date a column
    combined_df = combined_df.reset_index()
    comb_ord_cols = ['tckr_symbol',	'date',  'watchlist',	'watchlist_active',	'highest_high_flat',	'flat_period_start',	'flat_period_end',	
    'buy_signal',	'buy_date',	'position_active',	'buyexit_signal',	'buyexit_date', 'trend_score',	
    'supertrend',	'supertrend_direction', 'final_ub',	'final_lb', 'atr',
    'open',	'high',	'low',	'close', 'volume', 'h-l', 'h-pc', 'l-pc', 'tr',	 'atr14', 'basic_ub', 'basic_lb', 'ema20', 'ema50','symbol','exit_reason']
    combined_df = combined_df[comb_ord_cols]

    # Recalculate totals after filtering
    total_buy_signals = len(combined_df[combined_df['buy_signal'] == True])
    total_watchlists = len(combined_df[combined_df['watchlist'] == True])
    print(f"Total stocks processed: {len(all_signals_df)}")
    print(f"Total flat periods found (after filtering): {total_watchlists // 5}")  # Assuming 5 rows per period
    print(f"Total buy signals generated (after filtering): {total_buy_signals}")
    
    ### Summarize the Signals
    summary_df = summarize_signals(combined_df, nifty_list)
    #
       
    # Save Results
    combined_df.reset_index().to_excel(swing_path2, index=False)  # Include date as a column
    summary_df.to_excel(summary_path, index=False)

    #Backtest Output Summary
    results_df = backtest_signals(summary_df)
    results_df.to_excel(backtest_path, index=False)
    
    print(f"Summary DataFrame created with {len(summary_df)} rows!")
