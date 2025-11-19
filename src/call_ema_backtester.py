#------------------------------------------------------------------------------
###            Common Libraries that are required across notebooks
#------------------------------------------------------------------------------
import pandas as pd
from datetime import date
from loguru import logger
import warnings
import pretty_errors


#------------------------------------------------------------------------------
###            Import Modules
#------------------------------------------------------------------------------

# Import Classes
from config import cfg_nifty as cfg_nifty
from config import cfg_vars as cfg_vars
from classes.cl_Backtester_EMA import EMA_Signal_Backtester

#------------------------------------------------------------------------------------------------------------
#                               Backtesting the EMAPlus Scanner
#        Confusion matrix (counts):
# True Positive (TP): predicted +, actually +
# True Negative (TN): predicted −, actually −

# False Positive (FP): predicted +, actually −
# False Negative (FN): predicted −, actually +
#------------------------------------------------------------------------------------------------------------

## Read the Backtesting File
backtesting_file_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/test_ema_all_results.csv'
scanner_df = pd.read_csv(backtesting_file_path)
# parse dates in mixed format (handles both YYYY-MM-DD and MM-DD-YYYY)
scanner_df['trade_dt'] = pd.to_datetime(scanner_df['trade_dt'], format='mixed', errors="coerce")
parsed = scanner_df['trade_dt'].notna().sum()
logger.info(f"Read Records: {len(scanner_df)} rows. Parsed trade_dt valid dates: {parsed}")

backtester = EMA_Signal_Backtester(
        df=scanner_df,
        profit_threshold=2.0  # 2% profit target for 7 days
    )
    
# Run complete analysis
results = backtester.run_analysis()
    
# Export results
backtester.export_results(output_dir=cfg_vars.backtest_ema_path)

# Access specific results
print(f"\nPrecision: {results['performance_metrics']['precision']*100:.2f}%")
print(f"Recall: {results['performance_metrics']['recall']*100:.2f}%")
print(f"Win Rate: {results['performance_metrics']['win_rate']*100:.2f}%")
    
# Get false positives for detailed review
fp_df = results['false_positives']
if len(fp_df) > 0:
    print(f"\nTop 5 worst false positives:")
    print(fp_df.nsmallest(5, 'pnl_pct')[['trade_dt', 'tckr_symbol', 'pnl_pct', 'signal_action']])
