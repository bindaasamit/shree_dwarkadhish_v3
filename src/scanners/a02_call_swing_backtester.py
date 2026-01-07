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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_Strategy import SwingTradingStrategyV2

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

from src.classes.scanner.cl_Swing_Strategy import SwingTradingStrategyV2
from src.classes.backtester.cl_Swing_Backtester import SwingStrategyBacktester, BacktestConfig

# ============================================================================
#                         USAGE EXAMPLE
# ============================================================================

def run_backtest(backtest_input_path) -> Dict:
    """
    Args:
        strategy_instance: Configured instance of SwingTradingStrategyV2
        swing_results_path: Optional path to save swing results
    
    Returns:
        Complete backtest results and diagnostics
    """
    # Prepare data (ensure OHLCV columns and DatetimeIndex)
    df = pd.read_excel(backtest_input_path, parse_dates=True)
    df = df.set_index(['symbol', 'date']).sort_index()
    
    # Create strategy
    strategy = SwingTradingStrategyV2(data=df)

    # Create config
    config = BacktestConfig(
        initial_capital=100000,
        base_position_size=0.02,
        slippage_pct=0.001,
        commission_pct=0.0005)

    # Run backtest
    backtester = SwingStrategyBacktester(strategy, config)
    results = backtester.run_backtest()
    
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    res = results['results']
    print(f"\nPerformance:")
    print(f"  Total Trades: {res['total_trades']}")
    print(f"  Win Rate: {res['win_rate']:.1f}%")
    print(f"  Total R: {res['total_r']:.2f}R")
    print(f"  Average R: {res['avg_r']:.2f}R")
    print(f"  Expectancy: {res['expectancy']:.2f}R")
    print(f"  Profit Factor: {res['profit_factor']:.2f}")
    
    print(f"\nReturns:")
    print(f"  Initial Capital: ${res['initial_capital']:,.2f}")
    print(f"  Final Capital: ${res['final_capital']:,.2f}")
    print(f"  Total Return: {res['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {res['max_drawdown_pct']:.2f}%")
    
    print(f"\nTrade Characteristics:")
    print(f"  Avg Bars Held: {res['avg_bars_held']:.1f}")
    print(f"  Avg Winner Bars: {res['avg_bars_winner']:.1f}")
    print(f"  Avg Loser Bars: {res['avg_bars_loser']:.1f}")
    print(f"  Max Win Streak: {res['max_win_streak']}")
    print(f"  Max Loss Streak: {res['max_loss_streak']}")
    
    print(f"\nCosts:")
    print(f"  Total Commission: ${res['total_commission']:,.2f}")
    print(f"  Total Slippage: ${res['total_slippage']:,.2f}")
    print(f"  Total Costs: ${res['total_costs']:,.2f}")

    # Print top recommendations
    print("\n" + "="*80)
    print("KEY RECOMMENDATIONS")
    print("="*80)
    
    recs = results['diagnostics']['recommendations']
    
    if recs['REMOVE']:
        print("\n  REMOVE (Eliminate):")
        for rec in recs['REMOVE'][:3]:
            print(f"  • {rec}")
    
    if recs['SCALE']:
        print("\n SCALE (Increase Position Size):")
        for rec in recs['SCALE'][:3]:
            print(f"  • {rec}")
    
    if recs['NEVER_TRADE']:
        print("\n NEVER TRADE AGAIN:")
        for rec in recs['NEVER_TRADE'][:3]:
            print(f"  • {rec}")
    
    if recs['TIGHTEN']:
        print("\n TIGHTEN (Reduce Risk):")
        for rec in recs['TIGHTEN'][:3]:
            print(f"  • {rec}")
    
    print("\n" + "="*50)
    print("WIN RATE IMPROVEMENT TIPS")
    print("="*50)
    for tip in results['diagnostics']['win_rate_improvement_tips']:
        print(f"• {tip}")

    print("\n" + "="*50)
    print("LOSS REDUCTION TIPS")
    print("="*50)
    for tip in results['diagnostics']['loss_reduction_tips']:
        print(f"• {tip}")
    
    return results

if __name__ == "__main__":
    backtest_input_path = cfg_vars.swing_model_dir + "fno_sects_data.xlsx"
    backtest_results_path = cfg_vars.swing_model_dir + "fno_sects_backtest_results.xlsx"
    results = run_backtest(backtest_input_path)
    pd.DataFrame([results['results']]).to_excel(backtest_results_path, index=False)

