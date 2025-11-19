import sys, os

# Add the project root directory to sys.path
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(project_root)
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


scanner_file_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/test_ema_all_results.csv'
backtester_file_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest_output/backtest_outcomes.csv'
#------------------------------------------------------------------------------
##   Read Scanner Output
#------------------------------------------------------------------------------

logger.info("Start Loading the Scanner Excel file.")
scanner_df = pd.read_csv(scanner_file_path)

#Remove Duplicates
scanner_df = scanner_df.drop_duplicates(subset=['tckr_symbol', 'trade_dt'])
# Convert to datetime first, then format
scanner_df['trade_dt'] = pd.to_datetime(scanner_df['trade_dt'], errors='coerce').dt.strftime('%d-%m-%Y')

#get only those rows that have signal_Action in 'STRONG_BUY'
scanner_df = scanner_df[scanner_df['signal_action'].isin(['STRONG BUY'])]



logger.info(f"Filtered Scanner records are - {len(scanner_df)} rows.")


#------------------------------------------------------------------------------
##   Read Backtester Output
#------------------------------------------------------------------------------

logger.info("Start Loading the Backtester Excel file.")
backtester_df = pd.read_csv(backtester_file_path)

#Remove Duplicates
backtester_df = backtester_df.drop_duplicates(subset=['tckr_symbol', 'trade_dt'])
# Convert to datetime first, then format
backtester_df['trade_dt'] = pd.to_datetime(backtester_df['trade_dt'], errors='coerce').dt.strftime('%d-%m-%Y')

#get only those rows that have signal_Action in 'STRONG_BUY' & outcome is 'FALSE_POSITIVE'
fp_backtester_df = backtester_df[backtester_df['signal_action'].isin(['STRONG BUY']) & backtester_df['outcome'].isin(['FALSE_POSITIVE'])]
logger.info(f"Filtered FP records are - {len(fp_backtester_df)} rows.")


#------------------------------------------------------------------------------
##   Join the Two Dataframes on tckr_symbol and trade_dt    
#------------------------------------------------------------------------------
merged_df = pd.merge(
    left=fp_backtester_df,
    right=scanner_df,
    on=['tckr_symbol', 'trade_dt'],
    suffixes=('_backtester', '_scanner'),
    how='inner'
)

merged_df.to_excel('D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest_output/fp_detailed_review.xlsx', index=False)
logger.info(f"Merged FP records are - {len(merged_df)} rows.")