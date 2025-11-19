#------------------------------------------------------------------------------
###            Common Libraries that are required across notebooks
#------------------------------------------------------------------------------
import pandas as pd
import yfinance as yf
import numpy as np
import sqlite3
import nbformat
import pickle
from datetime import date, timedelta
import concurrent.futures
from loguru import logger
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#------------------------------------------------------------------------------
###            Import Modules
#------------------------------------------------------------------------------

from classes.cl_SQLLite import SQLLiteManager
from config import cfg_nifty as cfg_nifty
from config import cfg_vars as cfg_vars

#Import Classes
from classes.cl_Signal_EMA import Basic_EMA_Scanner
from classes.cl_Signal_Ichimoku import Basic_Ichimoku_Scanner

def SQLiteTableManager(*args, **kwargs):
    raise NotImplementedError

def load_to_sqlite_db(file_name, truncate_and_load_flag):
    logger.info("Start Loading the Excel file.")
    file_path = cfg_vars.input_dir + file_name
    if file_name.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
        df = df[cfg_vars.selected_columns]
        if 'scty_series' in df.columns:
            df = df[df['scty_series'] == 'EQ']

        #Remove Duplicates
        df_no_duplicates = df.drop_duplicates(subset=['tckr_symbol', 'trade_dt'])
        df_no_duplicates['trade_dt'] = pd.to_datetime(df_no_duplicates['trade_dt'])

        db_path = cfg_vars.db_dir + cfg_vars.db_name
        
        loader= SQLLiteManager(db_path=db_path,table_name='historical_stocks')
        loader.write_data(df_no_duplicates,truncate_and_load_flag)
                          
        logger.info(f"Loaded Excel file: {file_path} with {len(df_no_duplicates)} rows.") 

    else:
        logger.error(f"Unsupported file type for {file_name}")
        return

def get_hist_data(start_date,end_date,nifty_list):
    db_path = cfg_vars.db_dir + cfg_vars.db_name
    logger.info(f"Retrieving Historical Data from {db_path}.")
    reader = SQLLiteManager(db_path=db_path, table_name=cfg_vars.inp_table_name)
    df = reader.read_data(cfg_vars.select_hist_query)
    logger.info(f"No of records retrived from SQLLite Table is {df.shape}.")

    #Remove Duplicates
    df_no_dups = df.drop_duplicates(subset=['tckr_symbol', 'trade_dt'])

    filtered_df = df_no_dups[
            df_no_dups['tckr_symbol'].isin(nifty_list) & 
            (df_no_dups['trade_dt'] >= start_date) & 
            (df_no_dups['trade_dt'] < end_date)
        ]
    #print(filtered_df.tail(2))
    logger.info(f"No of records after Filtering for Training: {len(filtered_df)} rows.")
    return df

def run_scanner_for_stock(hist_df, stock_symbol, use_ind, scanner_type):
    stock_df = hist_df[hist_df['tckr_symbol'] == stock_symbol]
    stock_df = stock_df.sort_values(by=['tckr_symbol', 'trade_dt'])
    
    # Check if we have enough data for the stock
    if len(stock_df) < 50:  # Minimum 50 trading days for reliable indicators
        logger.warning(f"Insufficient data for {stock_symbol}: only {len(stock_df)} records. Skipping.")
        return stock_symbol, None
    
    # Check for required columns
    required_columns = ['closing_price', 'high_price', 'low_price', 'total_trading_volume']
    missing_columns = [col for col in required_columns if col not in stock_df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns for {stock_symbol}: {missing_columns}. Skipping.")
        return stock_symbol, None
    
    # Check for excessive NaN values
    nan_ratio = stock_df[required_columns].isnull().mean().mean()
    if nan_ratio > 0.1:  # More than 10% NaN values
        logger.warning(f"Too many NaN values for {stock_symbol}: {nan_ratio:.1%}. Skipping.")
        return stock_symbol, None
    
    stock_df['profit_n_loss_pct'] = ((stock_df['closing_price'].shift(-7).fillna(0) - stock_df['closing_price']) / stock_df['closing_price']) * 100

    #print("Check Data is Read for Stock Symbol:", stock_symbol)
    #print(stock_df.head(2))
    
    try:
        # Initialize scanners
        logger.info(f"Scanning stock: {stock_symbol} with {len(stock_df)} records")
        #scanner = SignalScanner(stock_df)
        #signal_results = signal_scanner.run_all()

        if use_ind =='ema':
            ema_scanner = Basic_EMA_Scanner(stock_df)
            ema_df = ema_scanner.run_complete_analysis(strategy='full')
            if ema_df is None or len(ema_df) == 0:
                logger.warning(f"Scanner returned empty DataFrame for {stock_symbol}")
                return stock_symbol, None
            
            logger.info(f"No of Records returned for {stock_symbol}: {ema_df.shape}")
            return stock_symbol, ema_df
        elif use_ind == 'ich':
            ichimoku_scanner = Basic_Ichimoku_Scanner(stock_df)
            ich_df = ichimoku_scanner.run_complete_analysis(strategy='full')
            if ich_df is None or len(ich_df) == 0:
                logger.warning(f"Ichimoku scanner returned empty DataFrame for {stock_symbol}")
                return stock_symbol, None
           
            logger.info(f"No of Records returned for {stock_symbol}: {ich_df.shape}")
            return stock_symbol, ich_df
        else:
            # For both indicators
            ema_scanner = Basic_EMA_Scanner(stock_df)
            ema_df = ema_scanner.run_complete_analysis(strategy='full')
            if ema_df is None or len(ema_df) == 0:
                logger.warning(f"EMA scanner returned empty DataFrame for {stock_symbol}")
                return stock_symbol, None, None

            ichimoku_scanner = Basic_Ichimoku_Scanner(stock_df)
            ich_df = ichimoku_scanner.run_complete_analysis(strategy='full')
            if ich_df is None or len(ich_df) == 0:
                logger.warning(f"Ichimoku scanner returned empty DataFrame for {stock_symbol}")
                return stock_symbol, None
            
            logger.info(f"No of Records returned for {stock_symbol}: EMA {ema_df.shape}, Ichimoku {ich_df.shape}")
            return stock_symbol, ema_df, ich_df
                         
    except Exception as e:
        logger.error(f"Error scanning {stock_symbol}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return stock_symbol, None

#This function runs your Trading scanner in parallel for a list of stock symbols using concurrent.futures.ThreadPoolExecutor.
def scan_multiple_stocks(hist_df,stock_symbols,use_ind,scanner_type,pmax_workers=4):
    """
    Run scanners for multiple stocks in parallel and collect results.
    Args:
        hist_df (pd.DataFrame): Historical data for all stocks.
        stock_symbols (list): List of stock symbols to scan.
        pmax_workers (int): Maximum number of threads to use.

    Returns:
        dict: A dictionary where the key is the stock symbol, and the value is a dictionary of results.
    """
    all_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=pmax_workers) as executor:
        # For each symbol, it submits a job (a call to run_scanner_for_stock(symbol)) to the thread pool.
        # It creates a dictionary mapping each Future (a placeholder for a result) to the corresponding symbol
        future_to_stock = {
            executor.submit(run_scanner_for_stock, hist_df, symbol, use_ind,scanner_type): symbol
            for symbol in stock_symbols
        }
        for future in concurrent.futures.as_completed(future_to_stock):
            symbol = future_to_stock[future]
            try:
                if use_ind == 'ema':
                    symbol_name, ema_df = future.result()
                    if all([ema_df is not None]):
                        all_results[symbol_name] = {"dataframe1": ema_df}
                    else:
                        logger.warning(f"One or more dataframes are None for stock: {symbol_name}")             
                elif use_ind == 'ich':
                    symbol_name, ich_df = future.result()
                    if all([ich_df is not None]):
                        all_results[symbol_name] = {"dataframe1": ich_df}
                    else:
                        logger.warning(f"One or more dataframes are None for stock: {symbol_name}")             
                else:
                    symbol_name, ema_df, ich_df = future.result()
                    if all([ema_df is not None,
                            ich_df is not None]):
                        all_results[symbol_name] = {"dataframe1": ema_df,
                                                    "dataframe2": ich_df}
                    else:
                        logger.warning(f"One or more dataframes are None for stock: {symbol_name}")                 
            except Exception as e:
                logger.error(f"Error retrieving results for {symbol}: {e}")

    return all_results

def export_file(df, signal, sector, use_ind, signal_span):
    logger.info(" Exporting the Results Start.")
    #-----------------------------------------------------------------------------
    # Generate the buy and sell files for the last 7 days only
    #-----------------------------------------------------------------------------
    
    today = date.today()
    seven_days_ago = today - timedelta(days=signal_span)        
    logger.info(f'Seven Days Ago Date: {seven_days_ago}')
    
    if use_ind == 'ema':
        export_all_file_path= cfg_vars.ema_results_dir + sector + '_' + signal + '_all_results.csv'
        export_buy_file_path= cfg_vars.ema_results_dir + sector + '_' + signal + '_buy_results.csv'
        export_sell_file_path= cfg_vars.ema_results_dir + sector + '_' + signal + '_sell_results.csv'
        export_noaction_file_path= cfg_vars.ema_results_dir + sector + '_' + signal + '_noaction_results.csv'
    else:
        export_all_file_path= cfg_vars.ich_results_dir + sector + '_' + signal + '_all_results.csv'
        export_buy_file_path= cfg_vars.ich_results_dir + sector + '_' + signal + '_buy_results.csv'
        export_sell_file_path= cfg_vars.ich_results_dir + sector + '_' + signal + '_sell_results.csv'
        export_noaction_file_path= cfg_vars.ich_results_dir + sector + '_' + signal + '_noaction_results.csv'
        
    df['trade_dt'] = pd.to_datetime(df['trade_dt'])
    df = df.sort_values(['tckr_symbol', 'trade_dt'])
    
    # Convert boundary dates to pandas Timestamps for proper comparison
    start_date = pd.Timestamp(seven_days_ago)
    end_date = pd.Timestamp(today)
    df = df[df['trade_dt'].between(start_date, end_date)]
    
    # Convert to YYYY-MM-DD format for export
    df['trade_dt'] = df['trade_dt'].dt.strftime('%Y-%m-%d')

    ### Write the Full File File
    if use_ind == 'ema':
        df = df.round(decimals = cfg_vars.round_ema_num_vars)
        df = df[cfg_vars.ema_ordered_cols]
    else:
        df = df.round(decimals = cfg_vars.round_ich_num_vars)
        df = df[cfg_vars.ich_ordered_cols]
    df.to_csv(export_all_file_path, index=False)
    logger.success(f"Exported All results")

    ### Write the Buy File
    if use_ind == 'ema':
        buy_df = df[df['signal_type'].isin(['BUY'])]
        buy_df = buy_df[cfg_vars.ema_buy_vars]
    else:
        buy_df = df[df['signal_type'].isin(['BUY'])]
        buy_df = buy_df[cfg_vars.ich_buy_vars]

    buy_df.to_csv(export_buy_file_path, index=False)
    logger.success(f"Exported Buy results")

    ### Write the Sell File
    if use_ind == 'ema':
        sell_df = df[df['signal_type'].isin(['SELL'])]
        sell_df = sell_df[cfg_vars.ema_sell_vars]
    else:
        sell_df = df[df['signal_type'].isin(['SELL'])]
        sell_df = sell_df[cfg_vars.ich_sell_vars]

    sell_df.to_csv(export_sell_file_path, index=False)
    logger.success(f"Exported Sell results")

    ### Write the Noaction File
    if use_ind == 'ema':
        noaction_df = df[df['signal_type'].isin(['HOLD','WAIT'])]
        noaction_df = noaction_df[cfg_vars.ema_noaction_vars]
    else:
        noaction_df = df[df['signal_type'].isin(['HOLD','WAIT'])]
        noaction_df = noaction_df[cfg_vars.ich_noaction_vars]
    noaction_df.to_csv(export_noaction_file_path, index=False)
    logger.success(f"Exported Hold/Wait results")



