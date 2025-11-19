#------------------------------------------------------------------------------
###     EMA Signal Backtester - 7-Day Profit Analysis (Using Existing P&L)
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#                    Simple EMA Signal Backtester Class
#------------------------------------------------------------------------------

class EMA_Signal_Backtester:
    """
    Simplified backtesting framework that uses pre-calculated 7-day profits.
    
    Focus: Analyze STRONG BUY signals only against actual 7-day profit/loss outcomes
    - Classify outcomes: True Positive, False Positive (only for STRONG BUY signals)
    - Performance metrics for STRONG BUY signals
    - Root cause analysis for losses
    - Recommend fixes for signal generation
    """
    
    def __init__(self, df: pd.DataFrame, 
                 profit_threshold: float = 2.0):  # 2% profit target
        """
        Initialize backtester.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with signals and 'profit_n_loss_pct' column
        profit_threshold : float
            Minimum profit % to consider a signal "successful" (default 2%)
        """
        logger.info("Initializing EMA Signal Backtester (7-Day P&L Analysis)...")
        
        self.df = df.copy().reset_index(drop=True)
        self.profit_threshold = profit_threshold
        
        # Validate required columns
        self._validate_dataframe()
        
        # Results storage
        self.signal_outcomes = []
        self.performance_metrics = {}
    
    def _validate_dataframe(self):
        """Validate required columns exist"""
        required_cols = ['profit_n_loss_pct', 'signal_action', 'signal_type']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check if profit_n_loss_pct has data
        if self.df['profit_n_loss_pct'].isna().all():
            raise ValueError("profit_n_loss_pct column has no data!")
        
        logger.info(f"DataFrame validated: {len(self.df)} rows")
        logger.info(f"Profit data available: {(~self.df['profit_n_loss_pct'].isna()).sum()} rows")
    
    #==========================================================================
    # MAIN ANALYSIS WORKFLOW
    #==========================================================================
    
    def run_analysis(self):
        """
        Main analysis workflow:
        1. Classify signal outcomes (TP/FP/TN/FN)
        2. Calculate performance metrics
        3. Root cause analysis for losses
        4. Generate recommendations
        """
        logger.info("="*70)
        logger.info("STARTING 7-DAY PROFIT ANALYSIS")
        logger.info("="*70)
        
        # Step 1: Classify outcomes
        self._classify_signal_outcomes()
        
        # Step 2: Calculate metrics
        self._calculate_performance_metrics()
        
        # Step 3: Print summary
        self.print_performance_summary()
        
        # Step 4: Detailed analysis
        logger.info("\n" + "="*70)
        logger.info("DETAILED SIGNAL OUTCOME ANALYSIS")
        logger.info("="*70)
        
        self.analyze_true_positives()
        self.analyze_false_positives()
        # Note: Only analyzing STRONG BUY signals, so no FN/TN analysis
        
        # Step 5: Generate improvement roadmap
        self._generate_improvement_roadmap()
        
        return self.get_results()
    
    #==========================================================================
    # SIGNAL OUTCOME CLASSIFICATION
    #==========================================================================
    
    def _classify_signal_outcomes(self):
        """
        Classify STRONG BUY signals as TP or FP based on 7-day profit/loss outcome.
        
        Classification Logic (STRONG BUY only):
        ---------------------
        TRUE POSITIVE (TP):
            - STRONG BUY signal + Profit >= threshold
        
        FALSE POSITIVE (FP):
            - STRONG BUY signal + Profit < threshold (including losses)
        
        Note: Only STRONG BUY signals are analyzed. Other signal types are ignored.
        """
        logger.info("Classifying signal outcomes...")
        
        outcomes = []
        
        for idx, row in self.df.iterrows():
            signal_type = row['signal_type']
            signal_action = row['signal_action']
            pnl_pct = row['profit_n_loss_pct']
            
            # Only process STRONG BUY signals for backtesting
            if signal_action != 'STRONG BUY':
                continue
            
            # Skip rows without profit data
            if pd.isna(pnl_pct):
                continue
            
            outcome = self._classify_single_outcome(
                signal_type, signal_action, pnl_pct, idx, row
            )
            outcomes.append(outcome)
        
        self.signal_outcomes = outcomes
        self.outcomes_df = pd.DataFrame(outcomes)
        
        logger.info(f"Classified {len(outcomes)} signal outcomes")
        
        # Add classification back to main dataframe
        if len(outcomes) > 0:
            outcome_map = {o['idx']: o['outcome'] for o in outcomes}
            self.df['outcome_class'] = self.df.index.map(outcome_map)
    
    def _classify_single_outcome(self, signal_type: str, signal_action: str, 
                                 pnl_pct: float, idx: int, row: pd.Series) -> dict:
        """Classify a STRONG BUY signal outcome (simplified for STRONG BUY only)"""
        
        # Since we only process STRONG BUY signals, all are BUY type
        is_profitable = (pnl_pct >= self.profit_threshold)
        
        # Classification for STRONG BUY signals only
        if is_profitable:
            outcome = 'TRUE_POSITIVE'
            explanation = f"STRONG BUY successful: {pnl_pct:.2f}% profit"
        else:
            outcome = 'FALSE_POSITIVE'
            explanation = f"STRONG BUY failed: {pnl_pct:.2f}% (below {self.profit_threshold}% target)"
        
        return {
            'idx': idx,
            'trade_dt': row.get('trade_dt', idx),
            'tckr_symbol': row.get('tckr_symbol', 'N/A'),
            'signal_type': signal_type,
            'signal_action': signal_action,
            'signal_reason': row.get('signal_reason', 'N/A'),
            'signal_confidence': row.get('signal_confidence', 'N/A'),
            'outcome': outcome,
            'explanation': explanation,
            'pnl_pct': pnl_pct,
            'closing_price': row.get('closing_price', 0),
            # Include key indicators for root cause analysis
            'ema_ribbon_width_pct': row.get('ema_ribbon_width_pct', 0),
            'ema_avg_slope': row.get('ema_avg_slope', 0),
            'ema_sideways_market': row.get('ema_sideways_market', False),
            'ema_whipsaw_risk': row.get('ema_whipsaw_risk', False),
            'ema_trending_market': row.get('ema_trending_market', False),
            'rsi': row.get('rsi', np.nan),
            'adx': row.get('adx', np.nan),
            'volume_ratio': row.get('volume_ratio', np.nan),
            'atr_pct': row.get('atr_pct', np.nan)
        }
    
    #==========================================================================
    # PERFORMANCE METRICS
    #==========================================================================
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        if len(self.outcomes_df) == 0:
            logger.warning("No outcomes to analyze")
            return
        
        outcomes = self.outcomes_df
        
        # Count outcomes (STRONG BUY only analysis)
        tp_count = len(outcomes[outcomes['outcome'] == 'TRUE_POSITIVE'])
        fp_count = len(outcomes[outcomes['outcome'] == 'FALSE_POSITIVE'])
        tn_count = 0  # Not analyzing non-STRONG BUY signals
        fn_count = 0  # Not analyzing missed opportunities
        
        total = len(outcomes)
        
        # Calculate metrics for STRONG BUY signals only
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = 0  # Not applicable for STRONG BUY only analysis
        f1_score = 0  # Not applicable
        
        accuracy = precision  # For STRONG BUY analysis, accuracy = precision
        
        # Profit metrics for STRONG BUY signals
        strong_buy_signals = outcomes[outcomes['signal_action'] == 'STRONG BUY']
        
        if len(strong_buy_signals) > 0:
            total_profit = strong_buy_signals['pnl_pct'].sum()
            avg_profit = strong_buy_signals['pnl_pct'].mean()
            
            winning_trades = strong_buy_signals[strong_buy_signals['pnl_pct'] >= self.profit_threshold]
            losing_trades = strong_buy_signals[strong_buy_signals['pnl_pct'] < self.profit_threshold]
            
            win_rate = len(winning_trades) / len(strong_buy_signals) if len(strong_buy_signals) > 0 else 0
            
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum()) \
                           if len(losing_trades) > 0 and losing_trades['pnl_pct'].sum() != 0 else float('inf')
        else:
            total_profit = 0
            avg_profit = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Not applicable for STRONG BUY only analysis
        missed_profit = 0
        avoided_loss = 0
        
        self.performance_metrics = {
            'total_signals': total,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'tn_count': tn_count,
            'fn_count': fn_count,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'buy_signal_count': len(strong_buy_signals),
            'total_profit_pct': total_profit,
            'avg_profit_pct': avg_profit,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'missed_profit_pct': missed_profit,
            'avoided_loss_pct': avoided_loss
        }
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        m = self.performance_metrics
        
        if not m:
            logger.warning("No metrics to display")
            return
        
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\n📊 STRONG BUY SIGNAL ANALYSIS")
        logger.info(f"   Strong Buy Signals:     {m['buy_signal_count']}")
        logger.info(f"   True Positives (TP):    {m['tp_count']} ({m['tp_count']/m['total_signals']*100:.1f}%)")
        logger.info(f"   False Positives (FP):   {m['fp_count']} ({m['fp_count']/m['total_signals']*100:.1f}%)")
        logger.info(f"   True Negatives (TN):    {m['tn_count']} (Not analyzed)")
        logger.info(f"   False Negatives (FN):   {m['fn_count']} (Not analyzed)")
        
        logger.info(f"\n🎯 STRONG BUY PERFORMANCE METRICS")
        logger.info(f"   Precision: {m['precision']*100:.2f}% (Success Rate for STRONG BUY signals)")
        logger.info(f"   Recall:    N/A (Limited to STRONG BUY analysis)")
        logger.info(f"   F1-Score:  N/A (Limited to STRONG BUY analysis)")
        logger.info(f"   Accuracy:  {m['accuracy']*100:.2f}% (Same as Precision)")
        
        logger.info(f"\n💰 PROFIT METRICS (STRONG BUY Signals Only)")
        logger.info(f"   Total Profit:    {m['total_profit_pct']:.2f}%")
        logger.info(f"   Average Profit:  {m['avg_profit_pct']:.2f}%")
        logger.info(f"   Win Rate:        {m['win_rate']*100:.2f}%")
        logger.info(f"   Avg Win:         {m['avg_win_pct']:.2f}%")
        logger.info(f"   Avg Loss:        {m['avg_loss_pct']:.2f}%")
        logger.info(f"   Profit Factor:   {m['profit_factor']:.2f}")
        
        logger.info(f"\n📈 NOTE: Analysis limited to STRONG BUY signals only")
        logger.info(f"   No opportunity analysis (FN/TN) performed")
        
        # Grade the strategy
        grade = self._grade_strategy()
        logger.info(f"\n{'='*70}")
        logger.info(f"STRATEGY GRADE: {grade['letter']} - {grade['assessment']}")
        logger.info(f"Score: {grade['score']}/100")
        logger.info(f"{'='*70}")
    
    def _grade_strategy(self) -> dict:
        """Grade STRONG BUY strategy based on performance (precision-focused)"""
        m = self.performance_metrics
        score = 0
        
        # Precision (35 points) - Most important for STRONG BUY analysis
        if m['precision'] >= 0.7:
            score += 35
        elif m['precision'] >= 0.6:
            score += 25
        elif m['precision'] >= 0.5:
            score += 15
        elif m['precision'] >= 0.4:
            score += 5
        
        # Win Rate (25 points) - Should align with precision
        if m['win_rate'] >= 0.65:
            score += 25
        elif m['win_rate'] >= 0.55:
            score += 18
        elif m['win_rate'] >= 0.45:
            score += 10
        
        # Profit Factor (20 points)
        if m['profit_factor'] >= 2.0:
            score += 20
        elif m['profit_factor'] >= 1.5:
            score += 15
        elif m['profit_factor'] >= 1.2:
            score += 10
        elif m['profit_factor'] >= 1.0:
            score += 5
        
        # Average Profit (10 points)
        if m['avg_profit_pct'] >= 3.0:
            score += 10
        elif m['avg_profit_pct'] >= 2.0:
            score += 7
        elif m['avg_profit_pct'] >= 1.0:
            score += 4
        
        # Signal Count (10 points) - More signals = more confidence
        if m['buy_signal_count'] >= 50:
            score += 10
        elif m['buy_signal_count'] >= 25:
            score += 7
        elif m['buy_signal_count'] >= 10:
            score += 4
        
        # Assign grade
        if score >= 85:
            letter = 'A'
            assessment = 'Excellent - Strong predictive power'
        elif score >= 70:
            letter = 'B'
            assessment = 'Good - Minor improvements needed'
        elif score >= 55:
            letter = 'C'
            assessment = 'Average - Significant improvements needed'
        elif score >= 40:
            letter = 'D'
            assessment = 'Below Average - Major revision required'
        else:
            letter = 'F'
            assessment = 'Failed - Strategy not viable'
        
        return {'score': score, 'letter': letter, 'assessment': assessment}
    
    #==========================================================================
    # TRUE POSITIVES ANALYSIS
    #==========================================================================
    
    def analyze_true_positives(self):
        """Analyze successful signals to understand what works"""
        tp_df = self.outcomes_df[self.outcomes_df['outcome'] == 'TRUE_POSITIVE'].copy()
        
        if len(tp_df) == 0:
            logger.warning("No true positives found!")
            return
        
        logger.info("\n" + "="*70)
        logger.info("TRUE POSITIVE ANALYSIS (Successful Signals)")
        logger.info("="*70)
        logger.info(f"Total: {len(tp_df)} signals")
        logger.info(f"Average Profit: {tp_df['pnl_pct'].mean():.2f}%")
        logger.info(f"Total Profit: {tp_df['pnl_pct'].sum():.2f}%")
        
        # Analyze characteristics of successful signals
        logger.info("\n🎯 SUCCESS FACTORS:")
        
        # Signal type breakdown
        signal_breakdown = tp_df['signal_action'].value_counts()
        logger.info("\n Signal Type Distribution:")
        for signal, count in signal_breakdown.items():
            avg_profit = tp_df[tp_df['signal_action'] == signal]['pnl_pct'].mean()
            logger.info(f"   {signal}: {count} signals, Avg: {avg_profit:.2f}%")
        
        # Market conditions
        trending_wins = tp_df[tp_df['ema_trending_market'] == True]
        if len(trending_wins) > 0:
            logger.info(f"\n✓ Trending Market: {len(trending_wins)} wins ({len(trending_wins)/len(tp_df)*100:.1f}%)")
            logger.info(f"   Avg Profit: {trending_wins['pnl_pct'].mean():.2f}%")
        
        # RSI analysis
        if 'rsi' in tp_df.columns:
            rsi_valid = tp_df[tp_df['rsi'].notna()]
            if len(rsi_valid) > 0:
                logger.info(f"\n RSI at Entry:")
                logger.info(f"   Average RSI: {rsi_valid['rsi'].mean():.1f}")
                logger.info(f"   RSI Range: {rsi_valid['rsi'].min():.1f} - {rsi_valid['rsi'].max():.1f}")
        
        # EMA characteristics
        logger.info(f"\n EMA Characteristics:")
        logger.info(f"   Avg Ribbon Width: {tp_df['ema_ribbon_width_pct'].mean():.3f}%")
        logger.info(f"   Avg Slope: {tp_df['ema_avg_slope'].mean():.3f}%")
        
        logger.info("\n💡 KEY TAKEAWAY:")
        logger.info("   These signal characteristics produced profits - replicate them!")
        
        return tp_df
    
    #==========================================================================
    # FALSE POSITIVES ANALYSIS (Most Important for Improvement)
    #==========================================================================
    
    def analyze_false_positives(self):
        """
        Analyze failed BUY signals - most critical for improvement.
        These are signals that looked good but resulted in losses/poor returns.
        """
        fp_df = self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_POSITIVE'].copy()
        
        if len(fp_df) == 0:
            logger.info("\n✅ No false positives - perfect signal quality!")
            return
        
        logger.info("\n" + "="*70)
        logger.info("FALSE POSITIVE ANALYSIS (Failed Signals)")
        logger.info("="*70)
        logger.info(f"Total: {len(fp_df)} signals")
        logger.info(f"Average Loss: {fp_df['pnl_pct'].mean():.2f}%")
        logger.info(f"Total Loss: {fp_df['pnl_pct'].sum():.2f}%")
        
        # Root cause analysis
        logger.info("\n🔍 ROOT CAUSE ANALYSIS:")
        
        root_causes = []
        
        for idx, row in fp_df.iterrows():
            cause, fix, priority = self._diagnose_false_positive(row)
            root_causes.append({
                'idx': row['idx'],
                'trade_dt': row['trade_dt'],
                'tckr_symbol': row['tckr_symbol'],
                'signal_action': row['signal_action'],
                'pnl_pct': row['pnl_pct'],
                'root_cause': cause,
                'suggested_fix': fix,
                'priority': priority
            })
        
        fp_analysis_df = pd.DataFrame(root_causes)
        
        # Group by root cause
        cause_summary = fp_analysis_df.groupby('root_cause').agg({
            'pnl_pct': ['count', 'mean', 'sum'],
            'suggested_fix': 'first',
            'priority': 'first'
        }).round(2)
        
        cause_summary.columns = ['Count', 'Avg_Loss', 'Total_Loss', 'Fix', 'Priority']
        cause_summary = cause_summary.sort_values('Priority', ascending=False)
        
        logger.info("\nFalse Positive Root Causes (by priority):")
        logger.info(cause_summary)
        
        # Top fixes
        logger.info("\n" + "="*70)
        logger.info("🔧 RECOMMENDED FIXES (Priority Order):")
        logger.info("="*70)
        
        for idx, (cause, group) in enumerate(fp_analysis_df.groupby('root_cause'), 1):
            count = len(group)
            avg_loss = group['pnl_pct'].mean()
            total_loss = group['pnl_pct'].sum()
            fix = group['suggested_fix'].iloc[0]
            priority = group['priority'].iloc[0]
            
            logger.info(f"\n{idx}. [{priority}] {cause}")
            logger.info(f"   Impact: {count} signals, Avg Loss: {avg_loss:.2f}%, Total: {total_loss:.2f}%")
            logger.info(f"   FIX: {fix}")
        
        return fp_analysis_df
    
    def _diagnose_false_positive(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Diagnose why a signal failed.
        
        Returns:
        --------
        (root_cause, suggested_fix, priority)
        """
        
        # Check market conditions
        if row['ema_sideways_market']:
            return (
                "Sideways market - signal in choppy conditions",
                "Strengthen sideways detection: Lower ribbon threshold to 0.35 or add consolidation filter",
                "HIGH"
            )
        
        if row['ema_whipsaw_risk']:
            return (
                "Whipsaw risk - multiple crossovers caused reversal",
                "Increase whipsaw sensitivity: Require 5+ bars of stable trend after crossover",
                "HIGH"
            )
        
        # Check RSI conditions
        if not pd.isna(row['rsi']):
            if row['rsi'] > 75:
                return (
                    "Overbought entry - RSI > 75 at signal",
                    "Block signals when RSI > 75, or wait for pullback to 60-70 range",
                    "MEDIUM"
                )
            
            if row['rsi'] < 45:
                return (
                    "Weak momentum - RSI < 45 at bullish signal",
                    "Require RSI > 50 AND rising for all buy signals",
                    "HIGH"
                )
        
        # Check trend strength
        if not pd.isna(row['adx']) and row['adx'] < 15:
            return (
                "Weak trend - ADX < 15 indicates no clear direction",
                "Increase minimum ADX threshold to 18 for signal generation",
                "MEDIUM"
            )
        
        # Check volume
        if not pd.isna(row['volume_ratio']) and row['volume_ratio'] < 0.7:
            return (
                "Low volume - insufficient participation",
                "Require volume_ratio > 0.8 or volume_trend_increasing = True",
                "MEDIUM"
            )
        
        # Check volatility
        if not pd.isna(row['atr_pct']) and row['atr_pct'] > 4:
            return (
                "High volatility - ATR > 4% increases risk",
                "Block signals when ATR > 4% or use wider stops in volatile markets",
                "LOW"
            )
        
        # Check EMA characteristics
        if row['ema_ribbon_width_pct'] < 0.5:
            return (
                "Narrow EMA ribbon - EMAs too close together",
                "Require ribbon width > 0.6% for strong signals",
                "MEDIUM"
            )
        
        if abs(row['ema_avg_slope']) < 0.3:
            return (
                "Flat EMAs - insufficient directional bias",
                "Require avg_slope > 0.4% for bullish signals",
                "MEDIUM"
            )
        
        # Generic timing issue
        if row['pnl_pct'] < -2:
            return (
                "Poor timing - entered at local peak",
                "Add confirmation period: Wait 2-3 bars after crossover before entry",
                "HIGH"
            )
        else:
            return (
                "Marginal profit/small loss - signal was borderline",
                "Increase signal quality threshold: Only take signals with quality > 0.7",
                "LOW"
            )
    
    #==========================================================================
    # FALSE NEGATIVES ANALYSIS
    #==========================================================================
    
    def analyze_false_negatives(self):
        """Analyze missed opportunities - where we should have signaled but didn't"""
        fn_df = self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_NEGATIVE'].copy()
        
        if len(fn_df) == 0:
            logger.info("\n✅ No false negatives - capturing all opportunities!")
            return
        
        logger.info("\n" + "="*70)
        logger.info("FALSE NEGATIVE ANALYSIS (Missed Opportunities)")
        logger.info("="*70)
        logger.info(f"Total: {len(fn_df)} missed signals")
        logger.info(f"Average Missed Profit: {fn_df['pnl_pct'].mean():.2f}%")
        logger.info(f"Total Missed Profit: {fn_df['pnl_pct'].sum():.2f}%")
        
        logger.info("\n🔍 ROOT CAUSE ANALYSIS:")
        
        root_causes = []
        
        for idx, row in fn_df.iterrows():
            cause, fix, priority = self._diagnose_false_negative(row)
            root_causes.append({
                'idx': row['idx'],
                'trade_dt': row['trade_dt'],
                'tckr_symbol': row['tckr_symbol'],
                'pnl_pct': row['pnl_pct'],
                'root_cause': cause,
                'suggested_fix': fix,
                'priority': priority
            })
        
        fn_analysis_df = pd.DataFrame(root_causes)
        
        # Group by root cause
        cause_summary = fn_analysis_df.groupby('root_cause').agg({
            'pnl_pct': ['count', 'mean', 'sum'],
            'suggested_fix': 'first',
            'priority': 'first'
        }).round(2)
        
        cause_summary.columns = ['Count', 'Avg_Missed', 'Total_Missed', 'Fix', 'Priority']
        cause_summary = cause_summary.sort_values('Priority', ascending=False)
        
        logger.info("\nFalse Negative Root Causes (by priority):")
        logger.info(cause_summary)
        
        # Top fixes
        logger.info("\n" + "="*70)
        logger.info("🔧 RECOMMENDED FIXES (Priority Order):")
        logger.info("="*70)
        
        for idx, (cause, group) in enumerate(fn_analysis_df.groupby('root_cause'), 1):
            count = len(group)
            avg_missed = group['pnl_pct'].mean()
            total_missed = group['pnl_pct'].sum()
            fix = group['suggested_fix'].iloc[0]
            priority = group['priority'].iloc[0]
            
            logger.info(f"\n{idx}. [{priority}] {cause}")
            logger.info(f"   Impact: {count} opportunities, Avg: {avg_missed:.2f}%, Total: {total_missed:.2f}%")
            logger.info(f"   FIX: {fix}")
        
        return fn_analysis_df
    
    def _diagnose_false_negative(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Diagnose why a profitable opportunity was missed.
        
        Returns:
        --------
        (root_cause, suggested_fix, priority)
        """
        
        # Check if filters blocked the signal
        if row['ema_sideways_market']:
            return (
                "Sideways filter blocked signal - but breakout occurred",
                "Add breakout detection from sideways: Trigger signal when ribbon expands >0.5% after sideways period",
                "HIGH"
            )
        
        if row['ema_whipsaw_risk']:
            return (
                "Whipsaw filter blocked signal - but trend established",
                "Add whipsaw recovery logic: Signal when whipsaw ends and ribbon widens",
                "HIGH"
            )
        
        # Check RSI conditions
        if not pd.isna(row['rsi']):
            if row['rsi'] > 70:
                return (
                    "RSI overbought filter blocked entry - but strong trend continued",
                    "Use RSI zones (70-80) instead of hard cutoff, allow entry in strong uptrends",
                    "MEDIUM"
                )
            
            if row['rsi'] < 30:
                return (
                    "RSI oversold - no signal generated but recovery occurred",
                    "Add oversold bounce pattern: Signal when RSI crosses back above 30 with EMA alignment",
                    "LOW"
                )
        
        # Check for missing alignment
        if not row.get('ema_bullish_alignment', False):
            if row.get('ema_bullish_alignment_partial', False):
                return (
                    "Partial alignment (2/3 EMAs) - waiting for full alignment missed move",
                    "Enable partial alignment: Accept 2/3 EMAs aligned with volume/momentum confirmation",
                    "HIGH"
                )
            else:
                return (
                    "No EMA alignment - trend not detected early enough",
                    "Add early trend detection: Signal on EMA5 x EMA13 golden cross before full alignment",
                    "MEDIUM"
                )
        
        # Check ADX
        if not pd.isna(row['adx']) and row['adx'] < 15:
            return (
                "ADX too low - trend filter rejected weak but profitable move",
                "Lower ADX threshold to 12, or add momentum confirmation as alternative to ADX",
                "MEDIUM"
            )
        
        # Check volume
        if not pd.isna(row['volume_ratio']) and row['volume_ratio'] < 0.7:
            return (
                "Low volume blocked signal - but price moved on institutional accumulation",
                "Use volume trend instead of absolute ratio, or lower threshold to 0.6",
                "LOW"
            )
        
        # Check for consolidation breakout
        if row['ema_ribbon_width_pct'] < 0.5:
            return (
                "Narrow ribbon - signal not generated during consolidation before breakout",
                "Add consolidation breakout pattern: Detect range compression followed by expansion",
                "HIGH"
            )
        
        # Missing pattern recognition
        return (
            "No signal pattern matched - alternative entry pattern needed",
            "Add pullback/bounce patterns: Signal on price returning to EMA in established trend",
            "MEDIUM"
        )
    
    #==========================================================================
    # TRUE NEGATIVES ANALYSIS
    #==========================================================================
    
    def analyze_true_negatives(self):
        """Analyze correctly avoided bad trades"""
        tn_df = self.outcomes_df[self.outcomes_df['outcome'] == 'TRUE_NEGATIVE'].copy()
        
        if len(tn_df) == 0:
            logger.info("\n⚠️  No true negatives - all non-signals would have been profitable!")
            logger.info("   Consider: Filters may be TOO strict, causing false negatives")
            return
        
        logger.info("\n" + "="*70)
        logger.info("TRUE NEGATIVE ANALYSIS (Correctly Avoided)")
        logger.info("="*70)
        logger.info(f"Total: {len(tn_df)} correctly avoided")
        logger.info(f"Average Would-Be Loss: {tn_df['pnl_pct'].mean():.2f}%")
        logger.info(f"Total Loss Avoided: {abs(tn_df['pnl_pct'].sum()):.2f}%")
        
        logger.info("\n✅ PROTECTIVE FILTERS THAT WORKED:")
        
        # Analyze which filters saved us
        protective_filters = []
        
        for idx, row in tn_df.iterrows():
            filter_type = self._identify_protective_filter(row)
            protective_filters.append({
                'filter_type': filter_type,
                'avoided_loss': abs(row['pnl_pct'])
            })
        
        filter_df = pd.DataFrame(protective_filters)
        filter_summary = filter_df.groupby('filter_type').agg({
            'avoided_loss': ['count', 'mean', 'sum']
        }).round(2)
        
        filter_summary.columns = ['Count', 'Avg_Saved', 'Total_Saved']
        filter_summary = filter_summary.sort_values('Total_Saved', ascending=False)
        
        logger.info("\nProtective Filters Performance:")
        logger.info(filter_summary)
        
        logger.info("\n💡 VALIDATION:")
        logger.info("   These filters are working correctly - DO NOT remove or weaken them!")
        logger.info("   They saved you from significant losses.")
        
        return tn_df
    
    def _identify_protective_filter(self, row: pd.Series) -> str:
        """Identify which filter prevented a bad trade"""
        
        if row['ema_sideways_market']:
            return "Sideways market filter"
        
        if row['ema_whipsaw_risk']:
            return "Whipsaw risk filter"
        
        if not pd.isna(row['rsi']):
            if row['rsi'] > 75:
                return "RSI overbought filter (>75)"
            if row['rsi'] < 25:
                return "RSI oversold filter (<25)"
        
        if not pd.isna(row['adx']) and row['adx'] < 15:
            return "Weak trend filter (ADX < 15)"
        
        if not pd.isna(row['volume_ratio']) and row['volume_ratio'] < 0.6:
            return "Low volume filter"
        
        if row['ema_ribbon_width_pct'] < 0.4:
            return "Narrow ribbon filter"
        
        if not row.get('ema_bullish_alignment', False):
            return "EMA alignment filter"
        
        return "Multiple filters"
    
    #==========================================================================
    # IMPROVEMENT ROADMAP
    #==========================================================================
    
    def _generate_improvement_roadmap(self):
        """Generate prioritized improvement roadmap based on all analyses"""
        logger.info("\n" + "="*70)
        logger.info("IMPROVEMENT ROADMAP (Comprehensive)")
        logger.info("="*70)
        
        improvements = []
        
        # From False Positives (reduce losses)
        fp_df = self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_POSITIVE']
        if len(fp_df) > 0:
            fp_impact = abs(fp_df['pnl_pct'].sum())
            improvements.append({
                'category': 'FALSE_POSITIVE',
                'impact_value': fp_impact,
                'count': len(fp_df),
                'recommendation': 'Strengthen filters to reduce bad signals',
                'priority_score': fp_impact * 2  # Losses are 2x important
            })
        
        # From False Negatives (capture more profit)
        fn_df = self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_NEGATIVE']
        if len(fn_df) > 0:
            fn_impact = fn_df['pnl_pct'].sum()
            improvements.append({
                'category': 'FALSE_NEGATIVE',
                'impact_value': fn_impact,
                'count': len(fn_df),
                'recommendation': 'Relax filters or add alternative patterns',
                'priority_score': fn_impact * 1  # Missed profit is 1x important
            })
        
        # Sort by priority
        improvements.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info("\n📋 PRIORITY ACTIONS:")
        for i, item in enumerate(improvements, 1):
            logger.info(f"\n{i}. Address {item['category']}")
            logger.info(f"   Impact: {item['impact_value']:.2f}% across {item['count']} signals")
            logger.info(f"   Action: {item['recommendation']}")
        
        # Specific recommendations
        logger.info("\n" + "="*70)
        logger.info("🎯 SPECIFIC PARAMETER RECOMMENDATIONS:")
        logger.info("="*70)
        
        m = self.performance_metrics
        
        # Based on precision
        if m['precision'] < 0.6:
            logger.info("\n⚠️  LOW PRECISION - Too many false positives")
            logger.info("   → Strengthen sideways detection (threshold: 0.5 → 0.35)")
            logger.info("   → Increase confirmation period (2 bars → 3 bars)")
            logger.info("   → Require RSI > 50 AND rising for all buys")
            logger.info("   → Block signals when ADX < 18")
        
        # Based on recall
        if m['recall'] < 0.5:
            logger.info("\n⚠️  LOW RECALL - Missing too many opportunities")
            logger.info("   → Enable partial alignment (2/3 EMAs)")
            logger.info("   → Add pullback/breakout patterns")
            logger.info("   → Lower ADX threshold (15 → 12)")
            logger.info("   → Use RSI zones (70-80) instead of hard 70 cutoff")
            logger.info("   → Add whipsaw recovery detection")
        
        # Based on win rate
        if m['win_rate'] < 0.55:
            logger.info("\n⚠️  LOW WIN RATE - Signal quality needs improvement")
            logger.info("   → Increase minimum signal quality threshold (0.6 → 0.7)")
            logger.info("   → Require trending_market = True for strong signals")
            logger.info("   → Add volume trend confirmation")
        
        # Balanced approach
        if m['precision'] > 0.65 and m['recall'] < 0.5:
            logger.info("\n✅ PRECISION GOOD, RECALL LOW - Be more aggressive")
            logger.info("   → Current filters are too conservative")
            logger.info("   → Focus on FALSE NEGATIVE fixes to capture more profits")
        
        if m['recall'] > 0.6 and m['precision'] < 0.6:
            logger.info("\n✅ RECALL GOOD, PRECISION LOW - Be more selective")
            logger.info("   → Capturing opportunities but too many fail")
            logger.info("   → Focus on FALSE POSITIVE fixes to improve quality")
        
        if m['precision'] > 0.65 and m['recall'] > 0.6:
            logger.info("\n🎉 EXCELLENT BALANCE - Fine-tune for optimization")
            logger.info("   → Strategy is working well")
            logger.info("   → Make incremental improvements to edge cases")
    
    #==========================================================================
    # RESULTS EXPORT
    #==========================================================================
    
    def get_results(self) -> dict:
        """Get all analysis results"""
        return {
            'performance_metrics': self.performance_metrics,
            'outcomes_df': self.outcomes_df,
            'true_positives': self.outcomes_df[self.outcomes_df['outcome'] == 'TRUE_POSITIVE'],
            'false_positives': self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_POSITIVE'],
            'true_negatives': self.outcomes_df[self.outcomes_df['outcome'] == 'TRUE_NEGATIVE'],
            'false_negatives': self.outcomes_df[self.outcomes_df['outcome'] == 'FALSE_NEGATIVE']
        }
    
    def export_results(self, output_dir: str = '.'):
        """Export all results to CSV files"""
        logger.info(f"\nExporting results to {output_dir}/...")
        
        # Format trade_dt to YYYY-MM-DD for export
        export_df = self.outcomes_df.copy()
        if 'trade_dt' in export_df.columns:
            # Handle datetime conversion safely - check for existing datetime and handle NaT
            if not pd.api.types.is_datetime64_any_dtype(export_df['trade_dt']):
                export_df['trade_dt'] = pd.to_datetime(export_df['trade_dt'], errors='coerce')
            
            # Convert to string format, handling NaT values
            export_df['trade_dt'] = export_df['trade_dt'].dt.strftime('%Y-%m-%d').fillna('INVALID_DATE')
        
        # Export full outcomes
        if len(export_df) > 0:
            export_df.to_csv(f'{output_dir}/backtest_outcomes.csv', index=False)
            logger.info("✓ backtest_outcomes.csv")
        
        # Export by category (only TP and FP for STRONG BUY analysis)
        for outcome_type in ['TRUE_POSITIVE', 'FALSE_POSITIVE']:
            df = export_df[export_df['outcome'] == outcome_type]
            if len(df) > 0:
                filename = f'{output_dir}/{outcome_type.lower()}_analysis.csv'
                df.to_csv(filename, index=False)
                logger.info(f"✓ {outcome_type.lower()}_analysis.csv")
        
        # Export metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(f'{output_dir}/performance_metrics.csv', index=False)
        logger.info("✓ performance_metrics.csv")
        
        logger.info("\n✅ All results exported successfully!")
