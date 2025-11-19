#------------------------------------------------------------------------------
###            Common Libraries that are required across notebooks
#------------------------------------------------------------------------------
import pandas as pd
from datetime import date
from loguru import logger
import warnings
import pretty_errors

# Import Classes
from config import cfg_nifty as cfg_nifty
from config import cfg_vars as cfg_vars
from classes.cl_Signal_Ichimoku import Basic_Ichimoku_Scanner
from classes.cl_Backtester_Ichimoku import Ichimoku_Backtester

#------------------------------------------------------------------------------
#                        Ichimoku Backtesting
#------------------------------------------------------------------------------
# Step 1: Get the Ichimoku Scanner signals

backtesting_file_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/scanner/ich/fno_ich_all_results.csv'
scanner_df = pd.read_csv(backtesting_file_path)
# parse dates in mixed format (handles both YYYY-MM-DD and MM-DD-YYYY)
scanner_df['trade_dt'] = pd.to_datetime(scanner_df['trade_dt'], format='mixed', errors="coerce")
parsed = scanner_df['trade_dt'].notna().sum()
logger.info(f"Read Records: {len(scanner_df)} rows. Parsed trade_dt valid dates: {parsed}")

# Step 2: Run backtest
backtester = Ichimoku_Backtester(
    df=scanner_df,
    initial_capital=100000,
    commission=0.001,
    slippage=0.001
)

# Step 3: Run complete analysis
report = backtester.run_complete_backtest(
    holding_period=5,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    profit_threshold_pct=2.0
)

# Step 4: Export results
backtestfile = "D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest/ich/ichimoku_backtest_"
backtester.export_results(base_filename=backtestfile)

# Step 5: Access specific results
false_positives = backtester.false_positives
false_negatives = backtester.false_negatives
recommendations = backtester.recommendations

# Step 6: Review top issues
print("\nTop False Positive Issues:")
for reason, count in recommendations['summary']['false_positive_patterns'].items():
    print(f"  {reason}: {count} times")

print("\nTop False Negative Issues:")
for reason, count in recommendations['summary']['false_negative_patterns'].items():
    print(f"  {reason}: {count} times")