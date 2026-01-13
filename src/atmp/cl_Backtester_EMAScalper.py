#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V3')  # Add project root to path
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta, datetime
import concurrent.futures
import yfinance as yf
import pretty_errors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

from typing import List, Dict, Optional
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs, util_bear, util_bull

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

ema_scalper_path = cfg_vars.emascalpmodel_dir + 'ema_scalper_signals_all.xlsx'
backtest_results_path = cfg_vars.emascalpmodel_dir + 'backtest_results.xlsx'
loss_summary_path = cfg_vars.emascalpmodel_dir + 'loss_analysis.xlsx'

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------


class BacktestEngine:
    """
    Backtests trading signals against historical data.
    Now accepts a DataFrame of signals and a dictionary of OHLC data per symbol.
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], signals_df: pd.DataFrame,stop_loss_adjustment):
        """
        Initialize backtesting engine.
        
        Args:
            data_dict: Dictionary with {symbol: DataFrame} pairs containing OHLC data
            signals_df: DataFrame with columns: symbol, date, type, entry_price, stop_loss, profit_target, risk, reward, risk_reward_ratio, index
        """
        self.data_dict = data_dict
        self.signals_df = signals_df
        self.results = []
        self.stats = {}
        self.stop_loss_adjustment = stop_loss_adjustment  # e.g., 0.1 to widen by 10%
        
    def backtest_signals(self, max_hold_bars: int = 50) -> List[Dict]:
        """
        Backtest all signals and determine outcomes.
        
        Args:
            max_hold_bars: Maximum number of bars to hold a position
            
        Returns:
            List of trade results
        """
        
        self.results = []
        
        for _, signal in self.signals_df.iterrows():
            symbol = signal['symbol']
            if symbol not in self.data_dict:
                continue  # Skip if no data for symbol
            data = self.data_dict[symbol]
            
            signal_date = signal['date']
            if signal_date not in data.index:
                continue  # Skip if date not in data index
            signal_idx = data.index.get_loc(signal_date)
            
            # Look forward to see if stop loss or profit target was hit
            end_idx = min(signal_idx + max_hold_bars, len(data))
            future_data = data.iloc[signal_idx:end_idx]
            
            if len(future_data) < 2:  # Not enough data to evaluate
                continue
            
            type_ = signal['type']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            profit_target = signal['profit_target']
            risk = signal['risk']  # Risk amount
            reward = signal['reward']  # Reward amount
            ratio = signal['risk_reward_ratio']  # Reward / Risk ratio

            # Adjust stop loss distance  -#Amit
            original_risk_distance = abs(entry_price - stop_loss)
            adjusted_risk_distance = original_risk_distance * (1 + self.stop_loss_adjustment)

            if type_ == 'LONG':
                stop_loss = entry_price - adjusted_risk_distance
            else:  # SHORT
                stop_loss = entry_price + adjusted_risk_distance

            # Adjust risk proportionally (reward stays the same to improve risk-reward ratio)
            risk = risk * (1 + self.stop_loss_adjustment)
            ratio = reward / risk  # Update ratio

            # Profit target remains unchanged

            if 'LONG' in type_:
                # Check if profit target hit
                profit_hit_mask = future_data['High'] >= profit_target
                # Check if stop loss hit
                stop_hit_mask = future_data['Low'] <= stop_loss
                
                profit_hit = profit_hit_mask.any()
                stop_hit = stop_hit_mask.any()
                
                if profit_hit and stop_hit:
                    # Check which came first
                    profit_idx = future_data[profit_hit_mask].index[0]
                    stop_idx = future_data[stop_hit_mask].index[0]
                    
                    if profit_idx < stop_idx:
                        outcome = 'WIN'
                        pnl = reward
                        exit_date = profit_idx
                        exit_price = profit_target
                    else:
                        outcome = 'LOSS'
                        pnl = -risk
                        exit_date = stop_idx
                        exit_price = stop_loss
                elif profit_hit:
                    outcome = 'WIN'
                    pnl = reward
                    exit_date = future_data[profit_hit_mask].index[0]
                    exit_price = profit_target
                elif stop_hit:
                    outcome = 'LOSS'
                    pnl = -risk
                    exit_date = future_data[stop_hit_mask].index[0]
                    exit_price = stop_loss
                else:
                    outcome = 'OPEN'
                    pnl = 0
                    exit_date = None
                    exit_price = None
                    
            else:  # SHORT
                # Check if profit target hit
                profit_hit_mask = future_data['Low'] <= profit_target
                # Check if stop loss hit
                stop_hit_mask = future_data['High'] >= stop_loss
                
                profit_hit = profit_hit_mask.any()
                stop_hit = stop_hit_mask.any()
                
                if profit_hit and stop_hit:
                    # Check which came first
                    profit_idx = future_data[profit_hit_mask].index[0]
                    stop_idx = future_data[stop_hit_mask].index[0]
                    
                    if profit_idx < stop_idx:
                        outcome = 'WIN'
                        pnl = reward
                        exit_date = profit_idx
                        exit_price = profit_target
                    else:
                        outcome = 'LOSS'
                        pnl = -risk
                        exit_date = stop_idx
                        exit_price = stop_loss
                elif profit_hit:
                    outcome = 'WIN'
                    pnl = reward
                    exit_date = future_data[profit_hit_mask].index[0]
                    exit_price = profit_target
                elif stop_hit:
                    outcome = 'LOSS'
                    pnl = -risk
                    exit_date = future_data[stop_hit_mask].index[0]
                    exit_price = stop_loss
                else:
                    outcome = 'OPEN'
                    pnl = 0
                    exit_date = None
                    exit_price = None
            
            # Adjust risk proportionally (reward stays the same to improve risk-reward ratio) - Amit
            risk = risk * (1 + self.stop_loss_adjustment)
            ratio = reward / risk  # Update ratio

            # Calculate holding period
            if exit_date:
                holding_bars = data.index.get_loc(exit_date) - signal_idx
            else:
                holding_bars = None
            
            trend_confirmation = signal.get('Trend_Confirmation', '')
            
            self.results.append({
                'symbol': symbol,
                'entry_date': signal_date,
                'exit_date': exit_date,
                'type': type_,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'outcome': outcome,
                'pnl': pnl,
                'risk': risk,
                'reward': reward,
                'risk_reward_ratio': ratio,
                'holding_bars': holding_bars,
                'trend_confirmation': trend_confirmation
            })
        
        return self.results
    
    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics from backtest results."""
        
        def compute_stats(trades_df):
            if len(trades_df) == 0:
                return {
                    'closed_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'profit_factor': 0,
                    'expectancy': 0,
                }
            wins = trades_df[trades_df['outcome'] == 'WIN']
            losses = trades_df[trades_df['outcome'] == 'LOSS']
            return {
                'closed_trades': len(trades_df),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                'total_pnl': trades_df['pnl'].sum(),
                'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
                'largest_win': wins['pnl'].max() if len(wins) > 0 else 0,
                'largest_loss': losses['pnl'].min() if len(losses) > 0 else 0,
                'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
                'expectancy': trades_df['pnl'].mean() if len(trades_df) > 0 else 0,
            }
        
        if not self.results:
            print("[All Symbols] No results to calculate statistics")
            return {}
        
        results_df = pd.DataFrame(self.results)
        closed_trades = results_df[results_df['outcome'] != 'OPEN']
        open_trades = results_df[results_df['outcome'] == 'OPEN']
        
        # Overall stats
        overall_stats = compute_stats(closed_trades)
        overall_stats['total_signals'] = len(self.results)
        overall_stats['open_trades'] = len(open_trades)
        
        # Calculate consecutive wins/losses for overall
        outcomes = closed_trades['outcome'].values
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive = 1
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                current_consecutive += 1
            else:
                if outcomes[i-1] == 'WIN':
                    max_consecutive_wins = max(max_consecutive_wins, current_consecutive)
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
                current_consecutive = 1
        if len(outcomes) > 0:
            if outcomes[-1] == 'WIN':
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive)
            else:
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        overall_stats['max_consecutive_wins'] = max_consecutive_wins
        overall_stats['max_consecutive_losses'] = max_consecutive_losses
        overall_stats['avg_holding_bars'] = closed_trades['holding_bars'].mean() if 'holding_bars' in closed_trades.columns and len(closed_trades) > 0 else 0
        
        self.stats = {'overall': overall_stats, 'by_type': {}}
        
        # Stats by type
        types = closed_trades['type'].unique()
        for type_ in types:
            type_closed = closed_trades[closed_trades['type'] == type_]
            type_open = open_trades[open_trades['type'] == type_]
            type_stats = compute_stats(type_closed)
            type_stats['total_signals'] = len(results_df[results_df['type'] == type_])
            type_stats['open_trades'] = len(type_open)
            type_stats['max_consecutive_wins'] = 0  # Can be calculated if needed
            type_stats['max_consecutive_losses'] = 0
            type_stats['avg_holding_bars'] = type_closed['holding_bars'].mean() if 'holding_bars' in type_closed.columns and len(type_closed) > 0 else 0
            self.stats['by_type'][type_] = type_stats
        
        self.stats['results_df'] = results_df
        return self.stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        
        if not self.stats:
            self.calculate_statistics()
        
        # Print overall
        print("\n" + "="*60)
        print("BACKTEST RESULTS - Overall")
        print("="*60)
        stats = self.stats['overall']
        print(f"Total Signals Generated: {stats['total_signals']}")
        print(f"Closed Trades: {stats['closed_trades']}")
        print(f"Open Trades: {stats['open_trades']}")
        print(f"Wins: {stats['wins']}")
        print(f"Losses: {stats['losses']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Total P&L: ${stats['total_pnl']:.2f}")
        print(f"Average Win: ${stats['avg_win']:.2f}")
        print(f"Average Loss: ${stats['avg_loss']:.2f}")
        print(f"Largest Win: ${stats['largest_win']:.2f}")
        print(f"Largest Loss: ${stats['largest_loss']:.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Expectancy: ${stats['expectancy']:.2f}")
        print(f"Max Consecutive Wins: {stats['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {stats['max_consecutive_losses']}")
        print(f"Avg Holding Period: {stats['avg_holding_bars']:.1f} bars")
        print("="*60)
        
        # Print by type
        for type_, stats in self.stats['by_type'].items():
            print(f"\nBACKTEST RESULTS - {type_}")
            print("="*60)
            print(f"Total Signals Generated: {stats['total_signals']}")
            print(f"Closed Trades: {stats['closed_trades']}")
            print(f"Open Trades: {stats['open_trades']}")
            print(f"Wins: {stats['wins']}")
            print(f"Losses: {stats['losses']}")
            print(f"Win Rate: {stats['win_rate']:.2f}%")
            print(f"Total P&L: ${stats['total_pnl']:.2f}")
            print(f"Average Win: ${stats['avg_win']:.2f}")
            print(f"Average Loss: ${stats['avg_loss']:.2f}")
            print(f"Largest Win: ${stats['largest_win']:.2f}")
            print(f"Largest Loss: ${stats['largest_loss']:.2f}")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Expectancy: ${stats['expectancy']:.2f}")
            print(f"Avg Holding Period: {stats['avg_holding_bars']:.1f} bars")
            print("="*60)
    
    def analyze_losses(self) -> Dict:
        """
        Analyze all losses to identify patterns and suggest fixes.
        
        Returns:
            Dictionary with loss analysis and suggestions
        """
        if not self.results:
            print("No results to analyze")
            return {}
        
        results_df = pd.DataFrame(self.results)
        losses = results_df[results_df['outcome'] == 'LOSS'].copy()
        
        if losses.empty:
            print("No losses to analyze")
            return {}
        
        # Accumulate losses DataFrame
        loss_analysis = {
            'losses_df': losses,
            'total_losses': len(losses),
            'avg_holding_bars_losses': losses['holding_bars'].mean(),
            'avg_loss_amount': losses['pnl'].mean(),
            'largest_loss': losses['pnl'].min(),
            'patterns': {},
            'suggestions': []
        }
        
        # Identify patterns
        # Pattern 1: Losses with very short holding period (stop hit immediately)
        quick_losses = losses[losses['holding_bars'] <= 1]
        if not quick_losses.empty:
            loss_analysis['patterns']['quick_losses'] = {
                'count': len(quick_losses),
                'percentage': len(quick_losses) / len(losses) * 100,
                'avg_holding': quick_losses['holding_bars'].mean(),
                'suggestion': "Stop loss hit too quickly. Consider widening stop loss or using trailing stop."
            }
            loss_analysis['suggestions'].append({
                'pattern': 'quick_losses',
                'fix': "Widen stop loss by 10%",
                'code_snippet': "",
                'perc_affected': loss_analysis['patterns']['quick_losses']['percentage']
            })
        
        # Pattern 2: Losses with long holding period (didn't hit target within max_bars)
        long_losses = losses[losses['holding_bars'] == 50]  # Assuming max_hold_bars=50
        if not long_losses.empty:
            loss_analysis['patterns']['long_losses'] = {
                'count': len(long_losses),
                'percentage': len(long_losses) / len(losses) * 100,
                'avg_holding': long_losses['holding_bars'].mean(),
                'suggestion': "Profit target not reached within max holding period. Consider increasing max_hold_bars or adjusting profit target."
            }
            loss_analysis['suggestions'].append({
                'pattern': 'long_losses',
                'fix': "Increase max_hold_bars to 75",
                'code_snippet': """
# In backtest call
backtest.backtest_signals(max_hold_bars=75)
""",
                'perc_affected': loss_analysis['patterns']['long_losses']['percentage']
            })
        
        # Pattern 3: Losses by symbol
        symbol_losses = losses.groupby('symbol').size().sort_values(ascending=False)
        if not symbol_losses.empty:
            loss_analysis['patterns']['symbol_losses'] = symbol_losses.to_dict()
            top_symbol_count = symbol_losses.iloc[0]
            perc_affected = top_symbol_count / len(losses) * 100
            loss_analysis['suggestions'].append({
                'pattern': 'symbol_losses',
                'fix': f"Avoid or adjust strategy for symbols with high losses: {symbol_losses.index[0]}",
                'code_snippet': "",
                'perc_affected': perc_affected
            })
        
        # Pattern 4: Losses by type (LONG/SHORT)
        type_losses = losses.groupby('type').size()
        if not type_losses.empty:
            loss_analysis['patterns']['type_losses'] = type_losses.to_dict()
            max_type = type_losses.idxmax()
            max_count = type_losses.max()
            perc_affected = max_count / len(losses) * 100
            if max_type == 'LONG':
                loss_analysis['suggestions'].append({
                    'pattern': 'type_losses',
                    'fix': "More LONG losses. Consider adjusting LONG entry conditions.",
                    'code_snippet': "",
                    'perc_affected': perc_affected
                })
        
        # Print analysis
        print("\n" + "="*60)
        print("LOSS ANALYSIS")
        print("="*60)
        print(f"Total Losses: {loss_analysis['total_losses']}")
        print(f"Average Holding Bars for Losses: {loss_analysis['avg_holding_bars_losses']:.1f}")
        print(f"Average Loss Amount: ${loss_analysis['avg_loss_amount']:.2f}")
        print(f"Largest Loss: ${loss_analysis['largest_loss']:.2f}")
        
        for pattern, data in loss_analysis['patterns'].items():
            if isinstance(data, dict) and 'count' in data:
                print(f"\n{pattern.upper()}: {data['count']} ({data['percentage']:.1f}%)")
                print(f"Suggestion: {data['suggestion']}")
        
        print("\nSUGGESTIONS:")
        for i, sugg in enumerate(loss_analysis['suggestions'], 1):
            print(f"{i}. {sugg['fix']}")
            print(f"   Code Snippet: {sugg['code_snippet']}")
        
        return loss_analysis
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame."""
        return pd.DataFrame(self.results)
    
    def plot_equity_curve(self, figsize: tuple = (12, 6)):
        """Plot the equity curve over time."""
        
        if not self.results:
            print("No results to plot")
            return
        
        results_df = pd.DataFrame(self.results)
        closed_trades = results_df[results_df['outcome'] != 'OPEN'].copy()
        
        if len(closed_trades) == 0:
            print("No closed trades to plot")
            return
        
        closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Equity curve
        ax1.plot(closed_trades['entry_date'], closed_trades['cumulative_pnl'], 
                 linewidth=2, color='blue')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax1.set_title('All Symbols - Equity Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Individual trade P&L
        colors = ['green' if x > 0 else 'red' for x in closed_trades['pnl']]
        ax2.bar(closed_trades['entry_date'], closed_trades['pnl'], color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Trade P&L ($)', fontsize=12)
        ax2.set_title('Individual Trade Results', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def read_nse_data(nifty_list,start_date):
    '''
    Read NSE_DATA table from SQLite DB and filter based on start_date and nifty_list
    '''
    db_path = cfg_vars.db_dir + cfg_vars.db_name
    table_name='nse_data'       
    query = cfg_vars.nse_data_read_query
    
    all_stocks_df = util_funcs.read_data(db_path, table_name, query) 
    print(f"No of Records read from NSE_DATA: {len(all_stocks_df)}.")
 
    # Filter for records after 2021-01-01 and where tckr_symbol is in nifty_list
    all_stocks_df['trade_dt'] = pd.to_datetime(all_stocks_df['trade_dt'], errors='coerce')  # Ensure datetime format
    all_stocks_df = all_stocks_df[
        (all_stocks_df['trade_dt'] >= pd.to_datetime(start_date)) & 
        (all_stocks_df['trade_dt'] <= pd.to_datetime(date.today())) & 
        (all_stocks_df['tckr_symbol'].isin(nifty_list))]
    print(f"No of Records post 2001-01-01 and in nifty_list from NSE_DATA: {len(all_stocks_df)}.")
    #all_stocks_df.to_excel('D:/myCourses/shree_dwarkadhish_v3/data/output/results/sdmodel/all_stocks_filtered.xlsx', index=False)
    all_stocks_df['trade_dt'] = all_stocks_df['trade_dt'].dt.strftime('%Y-%m-%d')
    print(min(all_stocks_df['trade_dt']), max(all_stocks_df['trade_dt']))
    return(all_stocks_df)

# ============================================================================
#                                MAIN WORKFLOW
# ============================================================================
if __name__ == "__main__":    
    # Import BacktestEngine from the backtester module
    from src.classes.backtester.cl_Backtester_EMAScalper import BacktestEngine
    
    # Read the Excel file with signals
    signals_df = pd.read_excel(ema_scalper_path)
    
    # Convert date to datetime
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # Get unique symbols
    symbols = signals_df['symbol'].unique()
    
    # Read NSE data for these symbols
    start_date = '2025-01-01'
    start_date = signals_df['date'].min().strftime('%Y-%m-%d')
    stocks_df = read_nse_data(list(symbols), start_date)
    
    # Prepare data_dict
    data_dict = {}
    for symbol in symbols:
        stock_data = stocks_df[stocks_df['tckr_symbol'] == symbol].copy()
        
        if stock_data.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        
        # Prepare data for BacktestEngine
        stock_data['trade_dt'] = pd.to_datetime(stock_data['trade_dt'])
        stock_data = stock_data.set_index('trade_dt').sort_index()
        stock_data = stock_data.rename(columns={
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'closing_price': 'Close',
            'total_trading_volume': 'Volume'
        })
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure High is highest and Low is lowest
        stock_data['High'] = stock_data[['Open', 'High', 'Close']].max(axis=1) + 1
        stock_data['Low'] = stock_data[['Open', 'Low', 'Close']].min(axis=1) - 1
        
        data_dict[symbol] = stock_data
    
    # Create BacktestEngine instance
    backtest = BacktestEngine(data_dict, signals_df,stop_loss_adjustment=0.1) #Amit

    # Run backtest
    results = backtest.backtest_signals(max_hold_bars=50)
    
    # Calculate and print statistics
    stats = backtest.calculate_statistics()
    backtest.print_statistics()
    
    # Analyze losses 
    loss_analysis = backtest.analyze_losses()

    # Save loss analysis to Excel
    loss_summary_path = cfg_vars.emascalpmodel_dir + 'loss_analysis.xlsx'
    sheets_to_write = []
    if 'losses_df' in loss_analysis and not loss_analysis['losses_df'].empty:
        sheets_to_write.append('Losses')
    if 'suggestions' in loss_analysis and loss_analysis['suggestions']:
        sheets_to_write.append('Suggestions')
    if 'patterns' in loss_analysis and loss_analysis['patterns']:
        sheets_to_write.append('Patterns')
    
    if sheets_to_write:
        with pd.ExcelWriter(loss_summary_path) as writer:
            if 'Losses' in sheets_to_write:
                loss_analysis['losses_df'].to_excel(writer, sheet_name='Losses', index=False)
            if 'Suggestions' in sheets_to_write:
                pd.DataFrame(loss_analysis['suggestions']).to_excel(writer, sheet_name='Suggestions', index=False)
            if 'Patterns' in sheets_to_write:
                patterns_list = []
                for pattern, data in loss_analysis['patterns'].items():
                    if isinstance(data, dict):
                        if 'count' in data:
                            patterns_list.append({
                                'pattern': pattern,
                                'count': data['count'],
                                'percentage': data['percentage'],
                                'avg_holding': data.get('avg_holding', None),
                                'suggestion': data.get('suggestion', '')
                            })
                        else:
                            # For groupby dicts like symbol_losses, type_losses
                            patterns_list.append({
                                'pattern': pattern,
                                'details': str(data)
                            })
                patterns_df = pd.DataFrame(patterns_list)
                patterns_df.to_excel(writer, sheet_name='Patterns', index=False)
        print("Loss analysis saved to loss_analysis.xlsx")
    else:
        print("No loss analysis data to save.")
    
    # Optionally, save results
    results_df = backtest.get_results_dataframe()
    results_df['entry_date'] = results_df['entry_date'].dt.strftime('%Y-%m-%d')
    results_df['exit_date'] = results_df['exit_date'].dt.strftime('%Y-%m-%d')
    results_df.to_excel(backtest_results_path, index=False)
    print("Backtest results saved to backtest_results.xlsx")
