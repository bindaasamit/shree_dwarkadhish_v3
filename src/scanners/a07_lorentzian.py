#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V5')  # Add project root to path
import math
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
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator, ADXIndicator

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Lorentzian import (process_nse_stocks,
    read_nse_data,
    rsi, cci, adx, wt,
    lorentzian_distance,
    rational_quadratic,
    gaussian,
    run_exact_lorentzian)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

#---------------------------------------------------------------------------------------------------------------------
#                                            MAIN WORKFLOW
#--------------------------------------------------------------------------------------------------------------------- 
if __name__ == "__main__":
    sector ='test'  # 'test' / 'nifty100' / 'fno'/ 'fno_movers' / 'small_mid' / 'all'
    start_date = '2024-01-01'
    #filter_date = '2025-11-01'
    filter_date = start_date

    # Before saving the files, generate timestamp:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_path = cfg_vars.swing_reversal_rsi_model_dir + f'{sector}_lorentzian_summary_{timestamp}.xlsx'
    
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
    if all_signals_df:
        combined_df = pd.concat(all_signals_df, ignore_index=False)

        combined_df['parity_checksum'] = (
            combined_df['prediction'] * 100000 +
            combined_df['signal'] * 10000 +
            combined_df['startLong'].astype(int) * 1000 +
            combined_df['startShort'].astype(int) * 100 +
            combined_df['endLong'].astype(int) * 10 +
            combined_df['endShort'].astype(int)
        )

        """
        ord_cols = ['tckr_symbol',  'date',
            'prediction', 'signal', 'signal_change', 
            'wasBullishRate', 'isBullishRate', 'isBullishChange', 'isBullishSmooth', 'isBullish', 
            'wasBearishRate', 'isBearishRate', 'isBearishChange', 'isBearishSmooth', 'isBearish',
            'startLong', 'endLong', 'endLongStrict', 'endLongDynamic',
            'startShort','endShort', 'endShortStrict',  'endShortDynamic', 
            'open', 'high', 'low', 'close', 'volume', 
            'f1', 'f2', 'f3', 'f4', 'f5', 'barsHeld','yhat1', 'yhat2']
        combined_df = combined_df[ord_cols]
        """
    else:
        combined_df = pd.DataFrame()
        
    ###Step2 Save Results
    combined_df.to_excel(summary_path, index=False)
    
    print(f"Summary DataFrame created with {len(combined_df)} rows!")





