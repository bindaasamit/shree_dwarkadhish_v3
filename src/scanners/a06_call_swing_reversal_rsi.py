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
from src.classes.scanner.cl_Swing_Reversal_RSI import (process_nse_stocks,
    read_nse_data,
    apply_hard_gates,
    calculate_pivot_points,
    get_monday_date,
    add_weekly_signals)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

#---------------------------------------------------------------------------------------------------------------------
#                                            MAIN WORKFLOW
#--------------------------------------------------------------------------------------------------------------------- 
weekly_data_path = cfg_vars.weekly_data_dir + f'stocks_weekly_data.xlsx'

# Example usage
def transform(df):
    ### Reset index to make date a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    ### Filter on pct_change > 2
    df = df[df['pct_change'].gt(2)]
    # Generate ceiling of pct_change
    df['pct_close_change'] = np.ceil(df['pct_change'])

    ### Sort for future calculations & Calculate future pct change after 5 candles
    df = df.sort_values(['tckr_symbol', 'date'])
    df['future_close_5'] = df.groupby('tckr_symbol')['close'].shift(-5)
    df['future_pct_5'] = (df['future_close_5'] - df['close']) / df['close'] * 100
    df['future_pct_5days_change'] = np.ceil(df['future_pct_5'])

    # Add ACTIVE_FLAG for rows without future data
    df['active_flag'] = np.where(df['future_pct_5'].isna(), 'Yes', 'No')

    # Add Weekly Signals
    #Get Start Date of the Week for each date
    df['w_start_date'] = df['date'].apply(get_monday_date)
    #df.to_excel(cfg_vars.swing_reversal_rsi_model_dir + 'debug_daily_data_with_daily_data.xlsx', index=False)
    df = add_weekly_signals(df, weekly_data_path)
    
    #Rearrange the columns
    ord_cols = ['tckr_symbol','date', 'pct_close_change', 'future_pct_5days_change', 'active_flag', 
                'hard_gate_score', 'quality_score', 'hard_gate_buy', 'weekly_structure_score', 
                'pp', 'r1', 's1', 'r2', 's2', 'flag_weekly_trend_stack', 'flag_weekly_range_strength', 
                'flag_weekly_atr_ok', 'flag_weekly_bbw_compression', 'volume',  'pivot_signal','close', 'future_close_5',	]
    df = df[ord_cols]
    return df

if __name__ == "__main__":
    sector ='all'  # 'test' / 'nifty100' / 'fno'/ 'fno_movers' / 'small_mid' / 'all'
    start_date = '2024-01-01'
    #filter_date = '2025-11-01'
    filter_date = start_date

    # Before saving the files, generate timestamp:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_path = cfg_vars.swing_reversal_rsi_model_dir + f'{sector}_summary_{timestamp}.xlsx'
    
    all_stocks = cfg_nifty.nifty100 + cfg_nifty.nifty_fno + cfg_nifty.nifty_mid_small_caps + cfg_nifty.mono_duopoly_stocks
    match sector:
        case 'test': nifty_list = cfg_nifty.nifty_reversal
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
        combined_df = transform(combined_df)
    else:
        combined_df = pd.DataFrame()
        
    ###Step2 Save Results
    combined_df.to_excel(summary_path, index=False)
    
    print(f"Summary DataFrame created with {len(combined_df)} rows!")