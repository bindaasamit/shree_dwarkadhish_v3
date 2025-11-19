import sys, os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
#------------------------------------------------------------------------------
###            Common Libraries that are required across notebooks
#------------------------------------------------------------------------------
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import concurrent.futures
from loguru import logger
import warnings
import pretty_errors
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import Configurations
from src.config import cfg_nifty as cfg_nifty
from src.config import cfg_vars as cfg_vars

# Import Utility Functions
from src.utils.util_functions import (export_file, get_hist_data, load_to_sqlite_db, scan_multiple_stocks)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

# Add a safe wrapper to call the utility and handle possible UnboundLocalError
def safe_load_to_sqlite_db(filename, truncate_flag):
    try:
        load_to_sqlite_db(filename, truncate_flag)
    except UnboundLocalError as e:
        # This indicates a bug inside load_to_sqlite_db where 'conn' (or similar) wasn't set.
        logger.exception(f"UnboundLocalError while loading to sqlite DB: {e}")
        return False
    except Exception as e:
        # Catch-all to avoid crashing the whole script; log details for debugging.
        logger.exception(f"Unexpected error while loading to sqlite DB: {e}")
        return False
    return True

#------------------------------------------------------------------------------
###            Main Definition
#------------------------------------------------------------------------------

def main():
    start_trade_dt = '2023-01-01'
    end_trade_dt = date.today().strftime('%Y-%m-%d')

    #-------------------------------------------------#
    #------Tweak these values before every run -------#
    #-------------------------------------------------#
    just_load_flag = "no"
    truncate_and_load_flag ='no'
    signal_span=365

    use_ind = 'ema'
    sector ='test'

    if just_load_flag == "yes":
        # Step1: Load the data to SQLite DB
        #     If "yes" "truncate the sqllite table and reload the table"
        #    If "no"  "append the new records to the sqllite table"
        #load_to_sqlite_db('NSE_20230101_20250805.xlsx',truncate_and_load_flag)
        # use wrapper to avoid UnboundLocalError crashing the script

        #data_file = 'NSE_20230101_20250901.xlsx'
        #data_file = 'NSE_20250901_20251112.xlsx'
        data_file = 'NSE_20251113_tilldate.xlsx'
        ok = safe_load_to_sqlite_db(data_file, truncate_and_load_flag)
        if not ok:
            logger.error("Loading to sqlite DB failed. Exiting main early to avoid further errors.")
            return
        #load_to_sqlite_db(,truncate_and_load_flag)
    else:
        logger.info("Skipping Step1: Load the data to SQLite DB")
        
        # Step2. Just get the stocks for which the scanner will run
        if sector =='test':
            nifty_list = cfg_nifty.nifty_test_list
        elif sector =='cash':   
            nifty_list = cfg_nifty.nifty_cash_list
        else:
            nifty_list = cfg_nifty.nifty_fno_list
        hist_df = get_hist_data(start_trade_dt,end_trade_dt,nifty_list)

        # Step3. Generate Insights for all stocks
        results_dict = scan_multiple_stocks(hist_df,nifty_list,use_ind=use_ind,scanner_type='basic')
        
        if use_ind == 'ema':  # Set to 'ich' for Ichimoku, 'ema' for EMA
            ema_list = [] # List to hold all EMA results for all stocks
            for symbol, result in results_dict.items():
                logger.info(f"Processing results for stock: {symbol}")
                ema_df = result["dataframe1"]
                if ema_df is not None:
                    ema_list.append(ema_df) # Append to list only if not None
                else:
                    logger.warning(f"Skipping stock {symbol} - DataFrame is None")

            if ema_list:
                ema_all_df = pd.concat(ema_list, ignore_index=True)
            else:
                logger.error("No valid EMA DataFrames to concatenate. All stocks failed to scan.")
                return
         
            export_file(ema_all_df, 'ema', sector, use_ind, signal_span)


        elif use_ind == 'ich':
            ich_list = [] # List to hold all Ichimoku results for all stocks
            for symbol, result in results_dict.items():
                logger.info(f"Processing results for stock: {symbol}")
                ich_df = result["dataframe1"]
                if ich_df is not None:
                    ich_list.append(ich_df) # Append to list only if not None
                else:
                    logger.warning(f"Skipping stock {symbol} - DataFrame is None")

            if ich_list:
                ich_all_df = pd.concat(ich_list, ignore_index=True)
            else:
                logger.error("No valid Ichimoku DataFrames to concatenate. All stocks failed to scan.")
                return
        
            export_file(ich_all_df, 'ich', sector, use_ind, signal_span)

        else:
            #-------------- Generate EMA Results ----------------#
            ema_list = [] # List to hold all EMA results for all stocks
            ich_list = [] # List to hold all Ichimoku results for all stocks
            
            for symbol, result in results_dict.items():
                logger.info(f"Processing results for stock: {symbol}")
                
                ema_df = result["dataframe1"]
                if ema_df is not None:
                    ema_list.append(ema_df) # Append to list only if not None
                else:
                    logger.warning(f"Skipping stock {symbol} - DataFrame is None")
                
                ich_df = result["dataframe2"]
                if ich_df is not None:
                    ich_list.append(ich_df) # Append to list only if not None
                else:
                    logger.warning(f"Skipping stock {symbol} - DataFrame is None")

            if ema_list:
                ema_all_df = pd.concat(ema_list, ignore_index=True)
            else:
                logger.error("No valid EMA DataFrames to concatenate. All stocks failed to scan.")
                return

            # Prep to export              
            export_file(ema_all_df, 'ema', sector,  use_ind, signal_span)

            #-------------- Generate ICH Results ----------------#               
            
            if ich_list:
                ich_all_df = pd.concat(ich_list, ignore_index=True)
            else:
                logger.error("No valid Ichimoku DataFrames to concatenate. All stocks failed to scan.")
                return

            # Prep to export              
            export_file(ich_all_df, 'ich', sector, use_ind, signal_span)
    
if __name__ == "__main__":
    main()