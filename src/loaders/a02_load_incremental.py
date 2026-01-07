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


logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error
#------------------------------------------------------------------------------
###            Get Historical Data from yfinance and transform to required format
#------------------------------------------------------------------------------
def load_incremental_data(incremental_dir,series_name):
    files = os.listdir(incremental_dir)
    print(f"Found {len(files)} files in {incremental_dir} to process.")
    for file in files:
        full_file_name = os.path.join(incremental_dir, file)
        print(f".Processing: {full_file_name}...")
        # Read the data
        try:
            # Determine file type and read accordingly
            if file.endswith('.csv'):
                df = pd.read_csv(full_file_name)
            elif file.endswith('.xlsx') or file.endswith('.xls'):
                df = pd.read_excel(full_file_name)
            else:
                print(f"Unsupported file type for {file}. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading {file}: {e}. Skipping.")
            continue
        
        #Preprocess the data before loading to DB
        # Select the required columns
        df = df[cfg_vars.required_columns]
        # Rename the columns
        df = df.rename(columns=cfg_vars.column_mapping)        

        #Add the 3 new columns
        df['series_start_dt'] = ''
        df['series_end_dt']  = ''
        df['series_name'] = series_name

        #Remove other series
        print("Count of records before filtering series:", len(df))
        df = df[df['scty_series'] == 'EQ']
        print("Count of records after filtering series:", len(df))
        
        #Load the incremental Data to DB
        db_path = cfg_vars.db_dir + cfg_vars.db_name
        table_name='nse_data'       

        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()

        logger.info(f"Loaded with {len(df)} rows.")         
        logger.info("DataFrame loaded to sqlite DB successfully.")

def main():
    """
    Load Incremental Data from all files in the directory to Sqlite DB
    """
    incremental_dir = 'C:/Users/Amit/Downloads/nse_latest_data'
    series_name = 'series_2025_12'  
    load_incremental_data(incremental_dir,series_name)

    """
    Update the series_name column in the sqlite DB for all records with the new series_name
    """
    
    
    

if __name__ == "__main__":
    main()