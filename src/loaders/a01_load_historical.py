#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V3')  # Add project root to path
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
from src.utils import util_functions

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error
#------------------------------------------------------------------------------
###            Get Historical Data from yfinance and transform to required format
#------------------------------------------------------------------------------

def get_historical_data(stock_list,historical_dir):
    
    successful_stocks = []
    
    for stock in stock_list:
        print(f"Downloading data for {stock}...")
        try:
            yf_df = yf.download(stock + '.NS', start='2000-01-01', end='2025-11-21')
            yf_df = yf_df.reset_index()   # Reset index to make Date a column
            
            if yf_df.empty:
                print(f"No data available for {stock}. Skipping.")
                continue
            
            stock_df = pd.DataFrame()
            stock_df['trade_dt'] = yf_df['Date']  # Date becomes trade_dt
            stock_df['scty_series'] = 'EQ'  # Default series for NSE stocks (can be customized)
            stock_df['open_price'] = yf_df['Open']
            stock_df['high_price'] = yf_df['High']
            stock_df['low_price'] = yf_df['Low']
            stock_df['closing_price'] = yf_df['Close']
            stock_df['last_price'] = yf_df['Open']  # Mapped to Open as specified
            stock_df['prev_closing_price'] = yf_df['Close'].shift(1)  # Previous day's close
            stock_df['total_trading_volume'] = yf_df['Volume']
            stock_df['total_transfer_value'] = 0
            stock_df.loc[0, 'prev_closing_price'] = 0
            stock_df = stock_df.assign(tckr_symbol = stock)
            stock_df = stock_df[['tckr_symbol', 'trade_dt', 'scty_series', 'open_price', 'high_price', 'low_price', 'closing_price', 'last_price', 'prev_closing_price', 'total_trading_volume', 'total_transfer_value']]
            stock_df = stock_df.round(decimals = {'open_price' : 2, 'high_price' : 2,	'low_price' : 2,'last_price' : 2, 'closing_price' : 2, 'prev_closing_price' : 2, 'total_transfer_value' : 2})
            
            # Save to separate file
            output_file = cfg_vars.historical_dir + f"{stock}_historical.csv"
            stock_df.to_csv(output_file, index=False)
            print(f"Saved data for {stock} to {output_file}")
            successful_stocks.append(stock)
            
        except Exception as e:
            print(f"Error downloading {stock}: {e}")
            continue
    
    return successful_stocks

def load_file_to_sqlite_db(full_file_name,append_flag):
    logger.info("Start Loading the file....")
    
    if full_file_name.lower().endswith(('.csv')):
        df = pd.read_csv(full_file_name)
        df = df[cfg_vars.hist_file_columns]
        if 'scty_series' in df.columns:
            df = df[df['scty_series'] == 'EQ']
    else:
        logger.error(f"Unsupported file type for {full_file_name}")
        return


def load_historical_data(historical_dir):
    # Load Hisotorical Data from all files in the directory
    files = os.listdir(historical_dir)
    print(f"Found {len(files)} files in {historical_dir} to process.")
    for file in files:
        full_file_name = os.path.join(historical_dir, file)
        print(f".Processing: {full_file_name}...")
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
        
        #Remove Duplicates 
        df_no_duplicates = df.drop_duplicates(subset=['tckr_symbol', 'trade_dt'])
        df_no_duplicates['trade_dt'] = pd.to_datetime(df_no_duplicates['trade_dt'])
        df_no_duplicates = df_no_duplicates[df_no_duplicates['scty_series'] == 'EQ']
        
        #Load the Historical Data to DB
        db_path = cfg_vars.db_dir + cfg_vars.db_name
        table_name='nse_data'       

        conn = sqlite3.connect(db_path)
        #replace - does a truncate and load
        df_no_duplicates.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()

        logger.info(f"Loaded with {len(df_no_duplicates)} rows.") 

def main():
    """
    print("## 1. First Download all the Data to Historical_Dir directory")
    nifty_list = cfg_nifty.nse_cash_list
    nifty_list = ['CUPID','INFY','HDFCLIFE','GAIL']  # For testing purposes, limit to a few stocks
    successful_stocks = get_historical_data(nifty_list,cfg_vars.historical_dir)
    print(f"   Successfully processed {len(successful_stocks)} stocks: {successful_stocks}")
    print("    Individual CSV files saved in the results directory.")
    """
    print("2. Load all the data from Historical_Dir to Sqlite DB")
    load_historical_data(cfg_vars.historical_dir)
    print("All Files Loaded to Sqlite DB successfully.")


# Example usage
if __name__ == "__main__":
    main()