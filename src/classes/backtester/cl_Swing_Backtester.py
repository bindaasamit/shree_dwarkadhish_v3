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


# ============================================================================
#                         CONFIGURATION & MODELS
# ============================================================================

@dataclass
class BacktestConfig:
    """Backtesting execution parameters."""
    
    # Capital & Position Sizing
    initial_capital: float = 100000.0
    base_position_size: float = 0.02  # 2% of capital per trade
    max_position_size: float = 0.10   # 10% cap
    
    # Execution Realism
    slippage_pct: float = 0.001       # 0.1% slippage
    commission_pct: float = 0.0005    # 0.05% commission per side
    entry_execution: str = "NEXT_OPEN"  # Entry at next bar open after signal
    
    # Position Sizing Multipliers
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'TRENDING': 1.25,
        'CONSOLIDATING': 0.75,
        'CHOPPY': 0.0  # Don't trade choppy markets
    })
    
    quality_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 0.8,
        2: 1.0,
        3: 1.2,
        4: 1.4
    })
    
    # Risk Management
    max_bars_held_long: int = 30
    max_bars_held_short: int = 30
    breakeven_r: float = 1.0  # Move to breakeven after 1R gain
    profit_lock_r: float = 1.0  # Start trailing after 1R
    
    # Pattern Priority (Higher number = higher priority)
    # Used when multiple patterns trigger on same bar
    pattern_priority: Dict[str, int] = field(default_factory=lambda: {
        'MOMENTUM_BURST': 4,
        'BREAKOUT': 3,
        'CONTINUATION': 2,
        'PULLBACK': 1,
        'MOMENTUM_CRASH': 4,
        'BREAKDOWN': 3,
        'CONTINUATION_SHORT': 2,
        'RALLY_FADE': 1
    })


@dataclass
class Trade:
    """Complete trade record with lifecycle tracking."""
    
    # Entry Information
    trade_id: int
    symbol: str
    signal_type: str  # 'LONG' or 'SHORT'
    pattern: str
    regime: str
    
    entry_date: datetime
    entry_bar: int
    entry_price: float
    entry_quality_score: int
    entry_impulse_score: int
    
    # Position Sizing
    position_size: float  # Dollars allocated
    position_multiplier: float  # Regime * Quality multiplier
    
    # Risk Parameters
    initial_stop: float
    initial_risk: float  # entry_price - initial_stop (absolute)
    initial_risk_r: float = 1.0  # Always 1R at entry
    
    # Exit Information
    exit_date: Optional[datetime] = None
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_details: Optional[str] = None
    
    # Performance Metrics
    bars_held: int = 0
    r_multiple: float = 0.0
    pnl_pct: float = 0.0
    pnl_dollars: float = 0.0
    
    # Trade Journey Tracking
    max_favorable_excursion: float = 0.0  # MFE in R
    max_adverse_excursion: float = 0.0    # MAE in R
    profit_locked_bar: Optional[int] = None
    stop_moved_to_breakeven_bar: Optional[int] = None
    
    # Pattern Failure Tracking
    pattern_failed: bool = False
    pattern_failure_bar: Optional[int] = None
    pattern_failure_reason: Optional[str] = None
    
    # Commission & Slippage
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    total_costs: float = 0.0


# ============================================================================
#                         BACKTESTING ENGINE
# ============================================================================

class SwingStrategyBacktester:
    """
    Production-grade backtesting engine with perfect live-trading replication.
    
    Key Design Principles:
    1. Zero lookahead bias - all decisions use only past data
    2. Bar-by-bar state machine - simulates real-time execution
    3. Strategy-faithful - uses exact same logic as live trading
    4. Complete audit trail - tracks every decision and state change
    """
    
    def __init__(self, 
                 strategy_instance,
                 config: BacktestConfig = None):
        """
        Initialize backtester with a configured strategy instance.
        
        Args:
            strategy_instance: Instance of SwingTradingStrategyV2 (already initialized with data)
            config: Backtesting configuration parameters
        """
        self.strategy = strategy_instance
        self.config = config or BacktestConfig()
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        
        # Portfolio tracking
        self.capital = self.config.initial_capital
        self.equity_curve: List[Dict] = []
        self.peak_equity = self.config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Performance tracking
        self.trade_counter = 0
        
    # ========================================================================
    #                         CORE EXECUTION LOGIC
    # ========================================================================
    
    def run_backtest(self) -> Dict:
        """
        Execute complete backtest with bar-by-bar simulation.
        
        Returns:
            Dictionary with complete backtest results and diagnostics
        """
        print("="*80)
        print("BACKTESTING ENGINE - LIVE TRADING REPLICATION MODE")
        print("="*80)
        print(f"Symbol: {self.strategy.symbol}")
        print(f"Data Range: {self.strategy.data.index[0]} to {self.strategy.data.index[-1]}")
        print(f"Total Bars: {len(self.strategy.data)}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print("="*80)
        
        # PHASE 1: Generate all entry signals (uses strategy's logic)
        print("\n[PHASE 1] Generating entry signals using live strategy logic...")
        entry_signals = self.strategy.generate_entry_signals()
        print(f"Total signals generated: {len(entry_signals)}")
        
        # PHASE 2: Simulate trade lifecycle bar-by-bar
        print("\n[PHASE 2] Simulating trade lifecycles...")
        self._simulate_trades(entry_signals)
        
        # PHASE 3: Calculate performance metrics
        print("\n[PHASE 3] Calculating performance metrics...")
        results = self._calculate_results()
        
        # PHASE 4: Generate diagnostics
        print("\n[PHASE 4] Generating deep diagnostics...")
        diagnostics = self._generate_diagnostics()
        
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        
        return {
            'config': self.config,
            'results': results,
            'diagnostics': diagnostics,
            'trades': self.closed_trades,
            'equity_curve': pd.DataFrame(self.equity_curve)
        }
    
    def _simulate_trades(self, entry_signals: List[Dict]) -> None:
        """
        Simulate complete trade lifecycle for all signals.
        
        This is the heart of the backtesting engine - it processes signals
        in chronological order and manages open positions bar-by-bar.
        """
        # Sort signals by date (chronological processing)
        sorted_signals = sorted(entry_signals, key=lambda x: x['index'])
        
        # Filter out choppy market signals (per config)
        filtered_signals = [
            sig for sig in sorted_signals 
            if self.config.regime_multipliers.get(sig['regime'], 0) > 0
        ]
        
        print(f"Filtered signals (removed choppy): {len(filtered_signals)}")
        
        # Handle multiple patterns on same bar (priority system)
        filtered_signals = self._apply_pattern_priority(filtered_signals)
        
        print(f"After pattern priority filtering: {len(filtered_signals)}")
        
        # Process each signal
        for signal in filtered_signals:
            # Check if we can enter (capital available, max positions, etc.)
            if not self._can_enter_trade(signal):
                continue
            
            # Create trade object
            trade = self._create_trade_from_signal(signal)
            
            # Execute entry
            self._execute_entry(trade, signal)
            
            # Add to open trades
            self.open_trades.append(trade)
            self.trades.append(trade)
            
            # Manage trade lifecycle bar-by-bar until exit
            self._manage_trade_lifecycle(trade, signal['index'] + 1)
        
        print(f"\nTotal trades executed: {len(self.closed_trades)}")
    
    def _apply_pattern_priority(self, signals: List[Dict]) -> List[Dict]:
        """
        When multiple patterns trigger on same bar, keep only highest priority.
        
        Priority order (from strategy): D > A > C > B for both LONG and SHORT
        """
        # Group by bar index
        by_bar = {}
        for sig in signals:
            bar = sig['index']
            if bar not in by_bar:
                by_bar[bar] = []
            by_bar[bar].append(sig)
        
        # For each bar, keep only highest priority signal
        filtered = []
        for bar, sigs in by_bar.items():
            if len(sigs) == 1:
                filtered.append(sigs[0])
            else:
                # Find highest priority
                best = max(sigs, key=lambda s: self.config.pattern_priority.get(s['pattern'], 0))
                filtered.append(best)
        
        return filtered
    
    def _can_enter_trade(self, signal: Dict) -> bool:
        """Check if we can enter a new trade."""
        # For now, simple check: do we have capital?
        # Can add: max concurrent positions, daily trade limits, etc.
        required_capital = self._calculate_position_size(signal)
        return self.capital >= required_capital
    
    def _create_trade_from_signal(self, signal: Dict) -> Trade:
        """Create Trade object from entry signal."""
        self.trade_counter += 1
        
        # Calculate position sizing
        position_size = self._calculate_position_size(signal)
        regime_mult = self.config.regime_multipliers.get(signal['regime'], 1.0)
        quality_mult = self.config.quality_multipliers.get(signal.get('quality_score', 2), 1.0)
        total_multiplier = regime_mult * quality_mult
        
        # Get entry bar data
        entry_bar = signal['index']
        entry_date = self.strategy.data.index[entry_bar]
        
        # Entry price: NEXT bar open (as per config)
        if entry_bar + 1 >= len(self.strategy.data):
            # Can't enter - signal at last bar
            return None
        
        next_bar_data = self.strategy.data.iloc[entry_bar + 1]
        entry_price_raw = next_bar_data['Open']
        
        # Apply slippage
        if signal['signal'] == 'LONG':
            entry_price = entry_price_raw * (1 + self.config.slippage_pct)
        else:
            entry_price = entry_price_raw * (1 - self.config.slippage_pct)
        
        # Calculate initial risk
        initial_stop = signal['stop_loss']
        initial_risk = abs(entry_price - initial_stop)
        
        # Calculate commissions
        entry_commission = position_size * self.config.commission_pct
        entry_slippage = abs(entry_price - entry_price_raw) * (position_size / entry_price)
        
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=self.strategy.symbol,
            signal_type=signal['signal'],
            pattern=signal['pattern'],
            regime=signal['regime'],
            entry_date=next_bar_data.name,  # Actual entry date (next bar)
            entry_bar=entry_bar + 1,  # Actual entry bar
            entry_price=entry_price,
            entry_quality_score=signal.get('quality_score', 0),
            entry_impulse_score=signal.get('impulse_score', 0),
            position_size=position_size,
            position_multiplier=total_multiplier,
            initial_stop=initial_stop,
            initial_risk=initial_risk,
            entry_commission=entry_commission,
            entry_slippage=entry_slippage
        )
        
        return trade
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """
        Calculate position size based on regime, quality, and risk.
        
        Position Size = base_size * regime_mult * quality_mult * capital
        Capped at max_position_size * capital
        """
        regime_mult = self.config.regime_multipliers.get(signal['regime'], 1.0)
        quality_mult = self.config.quality_multipliers.get(signal.get('quality_score', 2), 1.0)
        
        base_allocation = self.config.base_position_size * self.capital
        adjusted_allocation = base_allocation * regime_mult * quality_mult
        
        # Apply cap
        max_allocation = self.config.max_position_size * self.capital
        final_size = min(adjusted_allocation, max_allocation)
        
        return final_size
    
    def _execute_entry(self, trade: Trade, signal: Dict) -> None:
        """Execute trade entry and update capital."""
        # Deduct capital
        self.capital -= trade.position_size
        self.capital -= trade.entry_commission
        self.capital -= trade.entry_slippage
    
    def _manage_trade_lifecycle(self, trade: Trade, start_bar: int) -> None:
        """
        Manage trade bar-by-bar until exit.
        
        This simulates real-time trade management:
        1. Check pattern failure every bar
        2. Update dynamic stops
        3. Check stop loss
        4. Monitor profit protection
        5. Apply time-based exits
        
        Exit priority (first match wins):
        1. Volatility failure (bars 3-6 for momentum patterns)
        2. Pattern failure (structure breakdown)
        3. Hard stop loss
        4. Profit protection trailing stop
        5. EMA cross (last resort)
        6. Time exit
        """
        for current_bar in range(start_bar, len(self.strategy.data)):
            current_data = self.strategy.data.iloc[current_bar]
            bars_held = current_bar - trade.entry_bar
            
            # Update MFE/MAE
            self._update_excursions(trade, current_data)
            
            # --- EXIT LOGIC (Priority Order) ---
            
            # 1. VOLATILITY FAILURE (Early bars, momentum patterns only)
            exit_result = self._check_volatility_failure(trade, current_bar, bars_held, current_data)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # 2. PATTERN FAILURE (Structure breakdown)
            exit_result = self._check_pattern_failure_exit(trade, current_bar, bars_held)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # 3. HARD STOP LOSS (Dynamic stop)
            exit_result = self._check_stop_loss_exit(trade, current_bar, bars_held, current_data)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # 4. PROFIT PROTECTION (After profit locked)
            exit_result = self._check_profit_protection_exit(trade, current_bar, bars_held, current_data)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # 5. EMA CROSS (Last resort)
            exit_result = self._check_ema_cross_exit(trade, current_bar, bars_held, current_data)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # 6. TIME EXIT (Max bars held)
            exit_result = self._check_time_exit(trade, current_bar, bars_held, current_data)
            if exit_result:
                self._execute_exit(trade, current_bar, current_data, *exit_result)
                return
            
            # Update profit lock and breakeven stops
            self._update_trade_management(trade, current_bar, bars_held, current_data)
        
        # If we reach here, position still open at end of data
        self._execute_exit(
            trade, 
            len(self.strategy.data) - 1,
            self.strategy.data.iloc[-1],
            'STILL_HOLDING',
            'End of backtest data'
        )
    
    # ========================================================================
    #                         EXIT CONDITION CHECKS
    # ========================================================================
    
    def _check_volatility_failure_exit(self, trade: Trade, current_bar: int, 
                                       bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """
        Check volatility failure for momentum patterns (bars 3-6).
        
        Exit if:
        - Last 3 bars range < 1.1 * ATR
        - ATR contracting below 85% of entry ATR
        - No follow-through (close unfavorable)
        - Not in TRENDING regime
        """
        # Only for momentum patterns
        VOLATILITY_PATTERNS = {'BREAKOUT', 'MOMENTUM_BURST', 'BREAKDOWN', 'MOMENTUM_CRASH'}
        if trade.pattern not in VOLATILITY_PATTERNS:
            return None
        
        # Only check in bars 3-6
        if not (3 <= bars_held <= 6):
            return None
        
        # Not in trending regime
        if trade.regime == 'TRENDING':
            return None
        
        # Get entry ATR
        entry_data = self.strategy.data.iloc[trade.entry_bar]
        entry_atr = entry_data['ATR']
        
        # Check last 3 bars range
        recent_high = self.strategy.data['High'].iloc[current_bar-2:current_bar+1].max()
        recent_low = self.strategy.data['Low'].iloc[current_bar-2:current_bar+1].min()
        last_3_range = recent_high - recent_low
        
        # Check conditions
        atr_contracting = current_data['ATR'] < 0.85 * entry_atr
        no_follow_through = (
            current_data['Close'] < trade.entry_price if trade.signal_type == 'LONG'
            else current_data['Close'] > trade.entry_price
        )
        
        if last_3_range < 1.1 * current_data['ATR'] and atr_contracting and no_follow_through:
            return ('VOLATILITY_FAILURE', 
                    f'Volatility expansion failed - Range: {last_3_range:.2f}, ATR: {current_data["ATR"]:.2f}')
        
        return None
    
    def _check_pattern_failure_exit(self, trade: Trade, current_bar: int, 
                                    bars_held: int) -> Optional[Tuple[str, str]]:
        """
        Check if pattern structure has failed.
        
        Uses strategy's check_pattern_failure method - EXACT same logic as live trading.
        """
        if bars_held < 3:
            return None
        
        # Create entry signal dict for failure check
        entry_signal = {
            'index': trade.entry_bar - 1,  # Signal bar (before entry)
            'pattern': trade.pattern,
            'signal': trade.signal_type
        }
        
        failed, reason = self.strategy.check_pattern_failure(entry_signal, current_bar)
        
        if failed:
            trade.pattern_failed = True
            trade.pattern_failure_bar = current_bar
            trade.pattern_failure_reason = reason
            return ('PATTERN_FAILURE', reason)
        
        return None
    
    def _check_stop_loss_exit(self, trade: Trade, current_bar: int, 
                              bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """
        Check dynamic stop loss.
        
        Uses strategy's calculate_dynamic_stop and check_stop_loss methods.
        """
        # Get entry ATR
        entry_data = self.strategy.data.iloc[trade.entry_bar]
        entry_atr = entry_data['ATR']
        
        # Calculate dynamic stop
        stop_loss = self.strategy.calculate_dynamic_stop(
            trade.signal_type,
            trade.pattern,
            trade.entry_price,
            bars_held,
            current_data,
            entry_atr
        )
        
        # Check if stop hit
        stop_hit, exit_price = self.strategy.check_stop_loss(
            trade.signal_type,
            trade.entry_price,
            stop_loss,
            bars_held,
            current_data
        )
        
        if stop_hit:
            # Determine stop type
            if abs(stop_loss - current_data['SuperTrend']) < 0.01:
                stop_type = 'STOP_SUPERTREND'
            elif abs(stop_loss - current_data['EMA20']) < 0.01:
                stop_type = 'STOP_EMA20'
            else:
                stop_type = 'STOP_ATR'
            
            return (stop_type, f'Stop hit at {exit_price:.2f}')
        
        return None
    
    def _check_profit_protection_exit(self, trade: Trade, current_bar: int,
                                      bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """
        Check profit protection trailing stop.
        
        After locking profit (1R gain), trail stop to protect gains.
        """
        # Check if profit locked
        if trade.profit_locked_bar is None:
            # Check if we should lock profit
            if trade.signal_type == 'LONG':
                if current_data['High'] >= trade.entry_price + trade.initial_risk:
                    trade.profit_locked_bar = current_bar
            else:
                if current_data['Low'] <= trade.entry_price - trade.initial_risk:
                    trade.profit_locked_bar = current_bar
        
        # If profit locked, check trailing stop
        if trade.profit_locked_bar is not None:
            if trade.signal_type == 'LONG':
                trail_stop = max(
                    trade.entry_price + 0.5 * trade.initial_risk,
                    current_data['EMA20']
                )
                if current_data['Close'] < trail_stop:
                    return ('PROFIT_PROTECT', 
                            f'Profit locked at bar {trade.profit_locked_bar}, trailing stop hit')
            else:
                trail_stop = min(
                    trade.entry_price - 0.5 * trade.initial_risk,
                    current_data['EMA20']
                )
                if current_data['Close'] > trail_stop:
                    return ('PROFIT_PROTECT',
                            f'Profit locked at bar {trade.profit_locked_bar}, trailing stop hit')
        
        return None
    
    def _check_ema_cross_exit(self, trade: Trade, current_bar: int,
                              bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """
        Check EMA cross exit (last resort).
        
        Only after 3 bars and if not in profit.
        """
        if bars_held < 3:
            return None
        
        # Don't use if profit locked
        if trade.profit_locked_bar is not None:
            return None
        
        # Get EMA column names from strategy
        fast_close = self.strategy.fast_close_col
        slow_close = self.strategy.slow_close_col
        
        if trade.signal_type == 'LONG':
            if (current_data[fast_close] < current_data[slow_close] and 
                current_data['Close'] < trade.entry_price):
                return ('EMA_CROSS_DOWN', 'Fast EMA crossed below slow EMA')
        else:
            if (current_data[fast_close] > current_data[slow_close] and
                current_data['Close'] > trade.entry_price):
                return ('EMA_CROSS_UP', 'Fast EMA crossed above slow EMA')
        
        return None
    
    def _check_time_exit(self, trade: Trade, current_bar: int,
                        bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """
        Check time-based exit.
        
        LONG: Exit after 10 bars if R < 0.5 and below EMA20
        SHORT: Exit after 30 bars unconditionally
        """
        if trade.signal_type == 'LONG' and bars_held >= 10:
            current_r = (current_data['Close'] - trade.entry_price) / trade.initial_risk
            candle_range = current_data['High'] - current_data['Low']
            
            if (current_r < 0.5 and 
                current_data['Close'] <= current_data['EMA20'] and
                candle_range < 0.9 * current_data['ATR']):
                return ('TIME_EXIT', f'Low progress after 10 bars (R={current_r:.2f})')
        
        elif trade.signal_type == 'SHORT' and bars_held >= self.config.max_bars_held_short:
            return ('TIME_EXIT', f'Max {self.config.max_bars_held_short} bars held for SHORT')
        
        return None
    
    def _check_volatility_failure(self, trade: Trade, current_bar: int,
                                  bars_held: int, current_data: pd.Series) -> Optional[Tuple[str, str]]:
        """Wrapper for volatility failure check."""
        return self._check_volatility_failure_exit(trade, current_bar, bars_held, current_data)
    
    # ========================================================================
    #                         TRADE MANAGEMENT
    # ========================================================================
    
    def _update_excursions(self, trade: Trade, current_data: pd.Series) -> None:
        """Update MFE and MAE for trade."""
        if trade.signal_type == 'LONG':
            favorable = (current_data['High'] - trade.entry_price) / trade.initial_risk
            adverse = (trade.entry_price - current_data['Low']) / trade.initial_risk
        else:
            favorable = (trade.entry_price - current_data['Low']) / trade.initial_risk
            adverse = (current_data['High'] - trade.entry_price) / trade.initial_risk
        
        trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable)
        trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse)
    
    def _update_trade_management(self, trade: Trade, current_bar: int,
                                 bars_held: int, current_data: pd.Series) -> None:
        """Update trade management flags (breakeven, profit lock, etc.)."""
        # Move stop to breakeven after breakeven_r gain
        if trade.stop_moved_to_breakeven_bar is None:
            current_r = (
                (current_data['Close'] - trade.entry_price) / trade.initial_risk
                if trade.signal_type == 'LONG'
                else (trade.entry_price - current_data['Close']) / trade.initial_risk
            )
            
            if current_r >= self.config.breakeven_r:
                trade.stop_moved_to_breakeven_bar = current_bar
    
    def _execute_exit(self, trade: Trade, current_bar: int, current_data: pd.Series,
                     exit_reason: str, exit_details: str) -> None:
        """
        Execute trade exit and update all metrics.
        """
        # Exit price with slippage
        exit_price_raw = current_data['Close']
        
        if trade.signal_type == 'LONG':
            exit_price = exit_price_raw * (1 - self.config.slippage_pct)
        else:
            exit_price = exit_price_raw * (1 + self.config.slippage_pct)
        
        # Calculate performance
        bars_held = current_bar - trade.entry_bar
        
        if trade.signal_type == 'LONG':
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
            r_multiple = (exit_price - trade.entry_price) / trade.initial_risk
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
            r_multiple = (trade.entry_price - exit_price) / trade.initial_risk
        
        pnl_dollars = (pnl_pct / 100) * trade.position_size
        
        # Calculate costs
        exit_commission = trade.position_size * self.config.commission_pct
        exit_slippage = abs(exit_price - exit_price_raw) * (trade.position_size / exit_price)
        total_costs = trade.entry_commission + trade.entry_slippage + exit_commission + exit_slippage
        
        # Net PnL
        net_pnl = pnl_dollars - total_costs
        
        # Update trade
        trade.exit_date = current_data.name
        trade.exit_bar = current_bar
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.exit_details = exit_details
        trade.bars_held = bars_held
        trade.r_multiple = r_multiple
        trade.pnl_pct = pnl_pct
        trade.pnl_dollars = net_pnl
        trade.exit_commission = exit_commission
        trade.exit_slippage = exit_slippage
        trade.total_costs = total_costs
        
        # Update capital
        self.capital += trade.position_size  # Return position
        self.capital += net_pnl  # Add/subtract PnL
        
        # Move to closed trades
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        # Update equity curve
        self._update_equity_curve(current_bar, current_data)
    
    def _update_equity_curve(self, current_bar: int, current_data: pd.Series) -> None:
        """Update equity curve and drawdown tracking."""
        current_equity = self.capital
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Record
        self.equity_curve.append({
            'bar': current_bar,
            'date': current_data.name,
            'equity': current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': self.current_drawdown,
            'open_trades': len(self.open_trades)
        })
    
    # ========================================================================
    #                         RESULTS CALCULATION
    # ========================================================================
    
    def _calculate_results(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.closed_trades:
            return {'error': 'No closed trades'}
        
        trades_df = pd.DataFrame([vars(t) for t in self.closed_trades])
        
        # Basic metrics
        total_trades = len(trades_df)
        winners = trades_df[trades_df['r_multiple'] > 0]
        losers = trades_df[trades_df['r_multiple'] <= 0]
        
        win_rate = len(winners) / total_trades * 100
        
        # R metrics
        total_r = trades_df['r_multiple'].sum()
        avg_r = trades_df['r_multiple'].mean()
        avg_r_winner = winners['r_multiple'].mean() if len(winners) > 0 else 0
        avg_r_loser = losers['r_multiple'].mean() if len(losers) > 0 else 0
        
        # Dollar metrics
        total_pnl = trades_df['pnl_dollars'].sum()
        final_capital = self.config.initial_capital + total_pnl
        total_return_pct = (final_capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Risk metrics
        expectancy = avg_r
        profit_factor = abs(winners['pnl_dollars'].sum() / losers['pnl_dollars'].sum()) if len(losers) > 0 and losers['pnl_dollars'].sum() != 0 else float('inf')
        
        # Trade characteristics
        avg_bars_held = trades_df['bars_held'].mean()
        avg_bars_winner = winners['bars_held'].mean() if len(winners) > 0 else 0
        avg_bars_loser = losers['bars_held'].mean() if len(losers) > 0 else 0
        
        # Streak analysis
        trades_df['win'] = trades_df['r_multiple'] > 0
        max_win_streak = self._calculate_max_streak(trades_df['win'].tolist(), True)
        max_loss_streak = self._calculate_max_streak(trades_df['win'].tolist(), False)
        
        results = {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_r': total_r,
            'avg_r': avg_r,
            'avg_r_winner': avg_r_winner,
            'avg_r_loser': avg_r_loser,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'initial_capital': self.config.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown,
            'avg_bars_held': avg_bars_held,
            'avg_bars_winner': avg_bars_winner,
            'avg_bars_loser': avg_bars_loser,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'total_commission': trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum(),
            'total_slippage': trades_df['entry_slippage'].sum() + trades_df['exit_slippage'].sum(),
            'total_costs': trades_df['total_costs'].sum()
        }
        
        return results
    
    def _calculate_max_streak(self, win_list: List[bool], target: bool) -> int:
        """Calculate maximum consecutive streak."""
        max_streak = 0
        current_streak = 0
        
        for win in win_list:
            if win == target:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    # ========================================================================
    #                         DIAGNOSTICS ENGINE
    # ========================================================================
    
    def _generate_diagnostics(self) -> Dict:
        """
        Generate deep diagnostic analysis.
        
        Returns comprehensive insights for strategy optimization.
        """
        if not self.closed_trades:
            return {'error': 'No trades to analyze'}
        
        trades_df = pd.DataFrame([vars(t) for t in self.closed_trades])
        
        diagnostics = {
            'edge_decomposition': self._analyze_edge_decomposition(trades_df),
            'loss_analysis': self._analyze_losses(trades_df),
            'win_analysis': self._analyze_wins(trades_df),
            'pattern_performance': self._analyze_pattern_performance(trades_df),
            'regime_performance': self._analyze_regime_performance(trades_df),
            'exit_reason_analysis': self._analyze_exit_reasons(trades_df),
            'quality_score_analysis': self._analyze_quality_scores(trades_df),
            'big_loss_diagnosis': self._diagnose_big_losses(trades_df),
            'recommendations': self._generate_recommendations(trades_df),
            'win_rate_improvement_tips': self._generate_win_rate_tips(trades_df),
            'loss_reduction_tips': self._generate_loss_reduction_tips(trades_df)
        }
        
        return diagnostics
    
    def _analyze_edge_decomposition(self, df: pd.DataFrame) -> Dict:
        """Analyze which patterns contribute most to edge."""
        pattern_stats = []
        
        for pattern in df['pattern'].unique():
            pattern_trades = df[df['pattern'] == pattern]
            
            winners = pattern_trades[pattern_trades['r_multiple'] > 0]
            losers = pattern_trades[pattern_trades['r_multiple'] <= 0]
            
            stats = {
                'pattern': pattern,
                'total_trades': len(pattern_trades),
                'total_r': pattern_trades['r_multiple'].sum(),
                'avg_r': pattern_trades['r_multiple'].mean(),
                'win_rate': len(winners) / len(pattern_trades) * 100 if len(pattern_trades) > 0 else 0,
                'avg_bars_held': pattern_trades['bars_held'].mean(),
                'avg_r_winner': winners['r_multiple'].mean() if len(winners) > 0 else 0,
                'avg_r_loser': losers['r_multiple'].mean() if len(losers) > 0 else 0,
                'contribution_to_total_r': pattern_trades['r_multiple'].sum() / df['r_multiple'].sum() * 100 if df['r_multiple'].sum() != 0 else 0
            }
            
            pattern_stats.append(stats)
        
        # Sort by total_r contribution
        pattern_stats = sorted(pattern_stats, key=lambda x: x['total_r'], reverse=True)
        
        return {
            'by_pattern': pattern_stats,
            'best_pattern': pattern_stats[0]['pattern'] if pattern_stats else None,
            'worst_pattern': pattern_stats[-1]['pattern'] if pattern_stats else None
        }
    
    def _analyze_losses(self, df: pd.DataFrame) -> Dict:
        """Deep analysis of losing trades."""
        losers = df[df['r_multiple'] <= 0]
        
        if len(losers) == 0:
            return {'message': 'No losing trades'}
        
        # Exit reason distribution
        exit_reason_counts = losers['exit_reason'].value_counts()
        
        # Bars held distribution
        bars_held_stats = {
            'mean': losers['bars_held'].mean(),
            'median': losers['bars_held'].median(),
            'min': losers['bars_held'].min(),
            'max': losers['bars_held'].max()
        }
        
        # Fast vs slow losses
        fast_losses = losers[losers['bars_held'] <= 5]
        slow_losses = losers[losers['bars_held'] > 5]
        
        # Loss clustering
        losers_sorted = losers.sort_values('entry_bar')
        consecutive_losses = self._detect_loss_clusters(losers_sorted)
        
        return {
            'total_losers': len(losers),
            'avg_r_loser': losers['r_multiple'].mean(),
            'worst_r': losers['r_multiple'].min(),
            'exit_reasons': exit_reason_counts.to_dict(),
            'bars_held_stats': bars_held_stats,
            'fast_losses': {
                'count': len(fast_losses),
                'pct': len(fast_losses) / len(losers) * 100,
                'avg_r': fast_losses['r_multiple'].mean() if len(fast_losses) > 0 else 0
            },
            'slow_losses': {
                'count': len(slow_losses),
                'pct': len(slow_losses) / len(losers) * 100,
                'avg_r': slow_losses['r_multiple'].mean() if len(slow_losses) > 0 else 0
            },
            'loss_clusters': consecutive_losses
        }
    
    def _analyze_wins(self, df: pd.DataFrame) -> Dict:
        """Deep analysis of winning trades."""
        winners = df[df['r_multiple'] > 0]
        
        if len(winners) == 0:
            return {'message': 'No winning trades'}
        
        # Top performers
        top_20pct = winners.nlargest(int(len(winners) * 0.2), 'r_multiple')
        
        # Common characteristics
        top_patterns = top_20pct['pattern'].value_counts()
        top_regimes = top_20pct['regime'].value_counts()
        top_quality = top_20pct['entry_quality_score'].value_counts()
        
        # MFE analysis (profit left on table)
        profit_left = (winners['max_favorable_excursion'] - winners['r_multiple']).mean()
        
        return {
            'total_winners': len(winners),
            'avg_r_winner': winners['r_multiple'].mean(),
            'best_r': winners['r_multiple'].max(),
            'top_20pct': {
                'count': len(top_20pct),
                'avg_r': top_20pct['r_multiple'].mean(),
                'patterns': top_patterns.to_dict(),
                'regimes': top_regimes.to_dict(),
                'quality_scores': top_quality.to_dict()
            },
            'mfe_analysis': {
                'avg_mfe': winners['max_favorable_excursion'].mean(),
                'avg_realized_r': winners['r_multiple'].mean(),
                'avg_profit_left': profit_left,
                'pct_profit_left': profit_left / winners['max_favorable_excursion'].mean() * 100 if winners['max_favorable_excursion'].mean() > 0 else 0
            }
        }
    
    def _analyze_pattern_performance(self, df: pd.DataFrame) -> Dict:
        """Comprehensive pattern-by-pattern analysis."""
        analysis = {}
        
        for pattern in df['pattern'].unique():
            pattern_df = df[df['pattern'] == pattern]
            winners = pattern_df[pattern_df['r_multiple'] > 0]
            losers = pattern_df[pattern_df['r_multiple'] <= 0]
            
            analysis[pattern] = {
                'total_trades': len(pattern_df),
                'win_rate': len(winners) / len(pattern_df) * 100,
                'total_r': pattern_df['r_multiple'].sum(),
                'avg_r': pattern_df['r_multiple'].mean(),
                'expectancy': pattern_df['r_multiple'].mean(),
                'avg_bars_held': pattern_df['bars_held'].mean(),
                'avg_mfe': pattern_df['max_favorable_excursion'].mean(),
                'avg_mae': pattern_df['max_adverse_excursion'].mean(),
                'pattern_failures': len(pattern_df[pattern_df['pattern_failed'] == True])
            }
        
        return analysis
    
    def _analyze_regime_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by market regime."""
        analysis = {}
        
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            winners = regime_df[regime_df['r_multiple'] > 0]
            
            analysis[regime] = {
                'total_trades': len(regime_df),
                'win_rate': len(winners) / len(regime_df) * 100,
                'total_r': regime_df['r_multiple'].sum(),
                'avg_r': regime_df['r_multiple'].mean(),
                'expectancy': regime_df['r_multiple'].mean()
            }
        
        return analysis
    
    def _analyze_exit_reasons(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by exit reason."""
        analysis = {}
        
        for reason in df['exit_reason'].unique():
            reason_df = df[df['exit_reason'] == reason]
            
            analysis[reason] = {
                'count': len(reason_df),
                'pct_of_total': len(reason_df) / len(df) * 100,
                'avg_r': reason_df['r_multiple'].mean(),
                'avg_bars_held': reason_df['bars_held'].mean()
            }
        
        return analysis
    
    def _analyze_quality_scores(self, df: pd.DataFrame) -> Dict:
        """Test if quality scores predict performance."""
        analysis = {}
        
        for score in sorted(df['entry_quality_score'].unique()):
            score_df = df[df['entry_quality_score'] == score]
            winners = score_df[score_df['r_multiple'] > 0]
            
            analysis[f'quality_{score}'] = {
                'total_trades': len(score_df),
                'win_rate': len(winners) / len(score_df) * 100 if len(score_df) > 0 else 0,
                'avg_r': score_df['r_multiple'].mean(),
                'expectancy': score_df['r_multiple'].mean()
            }
        
        return analysis
    
    def _diagnose_big_losses(self, df: pd.DataFrame) -> Dict:
        """Root cause analysis of big losses (R < -2)."""
        big_losers = df[df['r_multiple'] < -2]
        
        if len(big_losers) == 0:
            return {'message': 'No big losses (R < -2)'}
        
        # Categorize by exit reason
        reasons = big_losers['exit_reason'].value_counts()
        
        # Regime analysis
        regimes = big_losers['regime'].value_counts()
        
        # Pattern analysis
        patterns = big_losers['pattern'].value_counts()
        
        # Timing analysis
        avg_bars = big_losers['bars_held'].mean()
        fast_crashes = len(big_losers[big_losers['bars_held'] <= 5])
        slow_bleeds = len(big_losers[big_losers['bars_held'] > 10])
        
        # Pattern failure detection
        pattern_failures_missed = len(big_losers[big_losers['pattern_failed'] == False])
        
        return {
            'total_big_losses': len(big_losers),
            'avg_r': big_losers['r_multiple'].mean(),
            'worst_r': big_losers['r_multiple'].min(),
            'exit_reasons': reasons.to_dict(),
            'regimes': regimes.to_dict(),
            'patterns': patterns.to_dict(),
            'timing': {
                'avg_bars_held': avg_bars,
                'fast_crashes': fast_crashes,
                'slow_bleeds': slow_bleeds
            },
            'pattern_failures_missed': pattern_failures_missed
        }
    
    def _detect_loss_clusters(self, losers_sorted: pd.DataFrame) -> List[Dict]:
        print("Detect consecutive losing streaks.")
        clusters = []
        current_cluster = []
        
        for idx, row in losers_sorted.iterrows():
            if not current_cluster:
                current_cluster.append(row)
            else:
                # Check if consecutive (within 5 bars)
                last_bar = current_cluster[-1]['exit_bar']
                if row['entry_bar'] - last_bar <= 5:
                    current_cluster.append(row)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append({
                            'size': len(current_cluster),
                            'start_date': current_cluster[0]['entry_date'],
                            'end_date': current_cluster[-1]['exit_date'],
                            'total_r': sum(t['r_multiple'] for t in current_cluster)
                        })
                    current_cluster = [row]
        
        # Check last cluster
        if len(current_cluster) >= 3:
            clusters.append({
                'size': len(current_cluster),
                'start_date': current_cluster[0]['entry_date'],
                'end_date': current_cluster[-1]['exit_date'],
                'total_r': sum(t['r_multiple'] for t in current_cluster)
            })
        
        return clusters
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict:
        print("Generate actionable recommendations based on all analysis.")
        recommendations = {
            'REMOVE': [],
            'TIGHTEN': [],
            'SCALE': [],
            'NEVER_TRADE': [],
            'RESEARCH_FURTHER': []
        }
        
        # Pattern-level analysis
        for pattern in df['pattern'].unique():
            pattern_df = df[df['pattern'] == pattern]
            avg_r = pattern_df['r_multiple'].mean()
            win_rate = len(pattern_df[pattern_df['r_multiple'] > 0]) / len(pattern_df) * 100
            total_trades = len(pattern_df)
            
            # REMOVE: Negative expectancy
            if avg_r < -0.2 and total_trades >= 10:
                recommendations['REMOVE'].append(
                    f"Remove {pattern} - negative expectancy: avg R={avg_r:.2f}, win rate={win_rate:.1f}%, {total_trades} trades"
                )
            
            # SCALE: High performance
            elif avg_r > 0.8 and win_rate > 60 and total_trades >= 10:
                recommendations['SCALE'].append(
                    f"Scale {pattern} - strong edge: avg R={avg_r:.2f}, win rate={win_rate:.1f}%, {total_trades} trades"
                )
            
            # RESEARCH: Insufficient sample
            elif total_trades < 10:
                recommendations['RESEARCH_FURTHER'].append(
                    f"Insufficient data for {pattern} - only {total_trades} trades, avg R={avg_r:.2f}"
                )
        
        # Pattern-Regime combinations
        for pattern in df['pattern'].unique():
            for regime in df['regime'].unique():
                combo_df = df[(df['pattern'] == pattern) & (df['regime'] == regime)]
                
                if len(combo_df) >= 5:
                    avg_r = combo_df['r_multiple'].mean()
                    win_rate = len(combo_df[combo_df['r_multiple'] > 0]) / len(combo_df) * 100
                    
                    # NEVER TRADE: Toxic combinations
                    if avg_r < -1.0 and win_rate < 30:
                        recommendations['NEVER_TRADE'].append(
                            f"NEVER trade {pattern} in {regime} regime - avg R={avg_r:.2f}, win rate={win_rate:.1f}%, {len(combo_df)} trades"
                        )
        
        # Quality score effectiveness
        for score in sorted(df['entry_quality_score'].unique()):
            score_df = df[df['entry_quality_score'] == score]
            if len(score_df) >= 10:
                avg_r = score_df['r_multiple'].mean()
                
                if avg_r < 0:
                    recommendations['TIGHTEN'].append(
                        f"Tighten quality threshold - Quality {score} has negative expectancy: avg R={avg_r:.2f}"
                    )
        
        # Exit reason analysis
        exit_analysis = df.groupby('exit_reason').agg({
            'r_multiple': ['count', 'mean'],
            'bars_held': 'mean'
        })
        
        for exit_reason, row in exit_analysis.iterrows():
            count = row[('r_multiple', 'count')]
            avg_r = row[('r_multiple', 'mean')]
            
            if 'STOP' in exit_reason and avg_r < -1.5 and count >= 10:
                recommendations['TIGHTEN'].append(
                    f"Tighten stops for {exit_reason} - avg loss = {avg_r:.2f}R from {int(count)} trades"
                )
        
        return recommendations
    
    def _generate_win_rate_tips(self, df: pd.DataFrame) -> List[str]:
        print("Generate specific tips to improve win rate.")
        tips = []
        
        # Analyze patterns with high win rate
        pattern_win_rates = {}
        for pattern in df['pattern'].unique():
            pattern_df = df[df['pattern'] == pattern]
            win_rate = len(pattern_df[pattern_df['r_multiple'] > 0]) / len(pattern_df) * 100
            pattern_win_rates[pattern] = win_rate
        
        high_win_patterns = [p for p, wr in pattern_win_rates.items() if wr > 60]
        if high_win_patterns:
            tips.append(f"Focus on patterns with high win rates: {', '.join(high_win_patterns)}")
        
        # Quality scores
        quality_win_rates = {}
        for score in df['entry_quality_score'].unique():
            score_df = df[df['entry_quality_score'] == score]
            win_rate = len(score_df[score_df['r_multiple'] > 0]) / len(score_df) * 100
            quality_win_rates[score] = win_rate
        
        if quality_win_rates:
            best_quality = max(quality_win_rates, key=quality_win_rates.get)
            tips.append(f"Prioritize high-quality entries (score {best_quality}) with {quality_win_rates[best_quality]:.1f}% win rate")
        
        # Regimes
        regime_win_rates = {}
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            win_rate = len(regime_df[regime_df['r_multiple'] > 0]) / len(regime_df) * 100
            regime_win_rates[regime] = win_rate
        
        if regime_win_rates:
            best_regime = max(regime_win_rates, key=regime_win_rates.get)
            tips.append(f"Trade primarily in {best_regime} regime ({regime_win_rates[best_regime]:.1f}% win rate)")
        
        return tips
    
    def _generate_loss_reduction_tips(self, df: pd.DataFrame) -> List[str]:
        print("Generate specific tips to reduce losses.")
        tips = []
        
        # Avoid low win rate patterns
        pattern_win_rates = {}
        for pattern in df['pattern'].unique():
            pattern_df = df[df['pattern'] == pattern]
            win_rate = len(pattern_df[pattern_df['r_multiple'] > 0]) / len(pattern_df) * 100
            pattern_win_rates[pattern] = win_rate
        
        low_win_patterns = [p for p, wr in pattern_win_rates.items() if wr < 40]
        if low_win_patterns:
            tips.append(f"Avoid or improve patterns with low win rates: {', '.join(low_win_patterns)}")
        
        # Tighten stops for bad exit reasons
        exit_avg_r = df.groupby('exit_reason')['r_multiple'].mean()
        bad_exits = exit_avg_r[exit_avg_r < -1].index.tolist()
        if bad_exits:
            tips.append(f"Tighten stops to avoid exits like: {', '.join(bad_exits)}")
        
        # Avoid bad regimes
        regime_avg_r = df.groupby('regime')['r_multiple'].mean()
        bad_regimes = regime_avg_r[regime_avg_r < 0].index.tolist()
        if bad_regimes:
            tips.append(f"Avoid trading in these regimes: {', '.join(bad_regimes)}")
        
        # Reduce position size in high loss scenarios
        big_losses = df[df['r_multiple'] < -2]
        if len(big_losses) > 0:
            tips.append(f"Reduce position size to limit big losses; {len(big_losses)} trades lost >2R")
        
        return tips
