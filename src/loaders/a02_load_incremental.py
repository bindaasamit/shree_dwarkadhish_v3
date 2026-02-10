#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V5')  # Add project root to path
import pandas as pd
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta
import concurrent.futures
import yfinance as yf
import pretty_errors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger
#------------------------------------------------------------------------------
###            Import Modules
#------------------------------------------------------------------------------
from config import cfg_nifty 
from config import cfg_vars 
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_Weekly import (read_nse_data,
        load_incremental_data,
        daily_to_weekly,
        add_weekly_ema_stack,
        add_weekly_range_position,
        add_weekly_atr_filter,
        add_weekly_bbw_filter,
        add_weekly_structure_score)


logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error
#------------------------------------------------------------------------------
###            Get Historical Data from yfinance and transform to required format
#------------------------------------------------------------------------------
def gen_weekly_indicators(df):
    df = add_weekly_ema_stack(df)
    df = add_weekly_range_position(df)
    df = add_weekly_atr_filter(df)
    df = add_weekly_bbw_filter(df)
    df = add_weekly_structure_score(df)
    return df

def main():
    incremental_dir = 'C:/Users/Amit/Downloads/nse_data'
    start_date = '2020-01-01'
    weekly_data_path = cfg_vars.weekly_data_dir + f'stocks_weekly_data.xlsx'
    
    nifty_list = sorted(list(set(cfg_nifty.nifty100 + cfg_nifty.nifty_fno + cfg_nifty.nifty_mid_small_caps + cfg_nifty.mono_duopoly_stocks)))
    #nifty_list = sorted(list(set(cfg_nifty.nifty_test)))

    task = 'gen_week_inds'  # 'load_only' / 'gen_week_inds' 
    
    match task:
        case 'load_only': 
            load_incremental_data(incremental_dir)    
        case 'gen_week_inds': 
            # Generate weekly data for all stocks
            stocks_df = read_nse_data(nifty_list, start_date)    
            weekly_data = daily_to_weekly(stocks_df)

            ### Generate all other Weekly Indicators
            weekly_data = gen_weekly_indicators(weekly_data)
            
            # Save to Excel
            weekly_data.to_excel(weekly_data_path, index=False)
            print(f"Weekly data saved to {weekly_data_path}")        

        case _: print("Invalid Task")
    
if __name__ == "__main__":
    main()