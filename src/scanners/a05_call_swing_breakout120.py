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
from src.classes.scanner.cl_Swing_Breakout120 import (read_nse_data,
    calculate_rsi,
    calculate_supertrend,
    calculate_bbw,
    calculate_pivot_points,
    get_weekly_trend,
    apply_breakout_strategy,
    process_nse_stocks,
    gen_basic_breakout_signals,
    gen_trade_group_summary,
    get_monday_date,
    add_weekly_signals,
    apply_hard_structural_exit_with_pnl)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error


#---------------------------------------------------------------------------------------------------------------------
#                                            MAIN WORKFLOW
#--------------------------------------------------------------------------------------------------------------------- 
sector ='all'  # 'test' / 'nifty100' / 'fno'/ 'fno_movers' / 'small_mid' / 'all'
start_date = '2024-01-01'
#filter_date = '2025-11-01'
filter_date = start_date

# Before saving the files, generate timestamp:
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
swing_path1 = cfg_vars.swing_model_dir2 + f'{sector}_data_{timestamp}.xlsx'
swing_path2 = cfg_vars.swing_model_dir2 + f'{sector}_summary_{timestamp}.xlsx'
final_path = cfg_vars.swing_model_dir2 + f'{sector}_final_{timestamp}.xlsx'
weekly_data_path = cfg_vars.weekly_data_dir + f'stocks_weekly_data.xlsx'

if __name__ == "__main__":    

    all_stocks = cfg_nifty.nifty100 + cfg_nifty.nifty_fno + cfg_nifty.nifty_mid_small_caps + cfg_nifty.mono_duopoly_stocks
    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_test
        case 'nifty100': nifty_list = cfg_nifty.nifty_50 + cfg_nifty.nifty_next_50
        case 'fno' : nifty_list = cfg_nifty.nifty_fno
        case 'fno_movers_poly': nifty_list = cfg_nifty.top_30_fno_swing + cfg_nifty.nifty_movers + cfg_nifty.mono_duopoly_stocks
        case 'small_mid': nifty_list = cfg_nifty.nifty_mid_small_caps
        case 'all': nifty_list = all_stocks
        case _: print("Invalid Sector")    
    # Sort the list alphabetically
    nifty_list = sorted(list(set(nifty_list)))
    
    ### Step1. Generate all Data points
    combined_df, close_df = gen_basic_breakout_signals(sector, nifty_list, start_date)    
    combined_df.to_excel(swing_path1, index=False)

    ### Step2 Save Results
    summary_df = gen_trade_group_summary(combined_df)
    summary_df.to_excel(swing_path2, index=False)   
    #print(f"Summary DataFrame created with {len(summary_df)} rows!")
     
    ### Step3. Append the Weekly Signals
    all_df = add_weekly_signals(summary_df, weekly_data_path)
    all_df = apply_hard_structural_exit_with_pnl(all_df,close_df)
    all_df.to_excel(final_path, index=False)

    print(f"Final DataFrame created with {len(all_df)} rows!")