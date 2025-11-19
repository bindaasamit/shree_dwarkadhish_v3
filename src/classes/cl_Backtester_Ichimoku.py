#------------------------------------------------------------------------------
###            Ichimoku Strategy Backtesting and Analysis Class
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger
import warnings
from collections import Counter
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Ichimoku_Backtester:
    """
    Backtesting class for Ichimoku Scanner with false positive/negative analysis
    and recommendation generation for strategy improvement.
    """
    
    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000, 
                 commission: float = 0.001, slippage: float = 0.001):
        """
        Initialize the backtester.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with Ichimoku signals (output from Basic_Ichimoku_Scanner)
        initial_capital : float
            Starting capital for backtesting
        commission : float
            Commission rate (default 0.1%)
        slippage : float
            Slippage rate (default 0.1%)
        """
        logger.info("Initializing IchimokuBacktester...")
        self.df = df.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Results storage
        self.trades = []
        self.false_positives = pd.DataFrame()
        self.false_negatives = pd.DataFrame()
        self.recommendations = {}
        
        # Validate required columns
        required_cols = ['signal_action', 'signal_type', 'closing_price', 'trade_dt']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    
    def run_backtest(self, holding_period: int = 5, stop_loss_pct: float = 0.05, 
                     take_profit_pct: float = 0.10):
        """
        Execute simple backtest strategy.
        
        Parameters:
        -----------
        holding_period : int
            Number of days to hold position (default 5)
        stop_loss_pct : float
            Stop loss percentage (default 5%)
        take_profit_pct : float
            Take profit percentage (default 10%)
        
        Returns:
        --------
        pd.DataFrame : Trade log with results
        """
        logger.info(f"Running backtest with holding_period={holding_period}, "
                   f"stop_loss={stop_loss_pct:.1%}, take_profit={take_profit_pct:.1%}")
        
        self.trades = []
        capital = self.initial_capital
        position = None
        
        for idx in range(len(self.df) - holding_period):
            row = self.df.iloc[idx]
            
            # Skip if already in position
            if position is not None:
                continue
            
            # Entry logic - only on STRONG or MODERATE signals
            if row['signal_action'] in ['STRONG BUY', 'MODERATE BUY']:
                entry_price = row['closing_price'] * (1 + self.slippage)
                entry_date = row['trade_dt']
                shares = int((capital * 0.95) / entry_price)  # Use 95% of capital
                
                if shares > 0:
                    position = {
                        'entry_idx': idx,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'signal_action': row['signal_action'],
                        'signal_confidence': row['signal_confidence'],
                        'trend_quality': row.get('trend_quality', 0),
                        'direction': 'LONG'
                    }
                    
            elif row['signal_action'] in ['STRONG SELL', 'MODERATE SELL']:
                entry_price = row['closing_price'] * (1 - self.slippage)
                entry_date = row['trade_dt']
                shares = int((capital * 0.95) / entry_price)
                
                if shares > 0:
                    position = {
                        'entry_idx': idx,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'signal_action': row['signal_action'],
                        'signal_confidence': row['signal_confidence'],
                        'trend_quality': row.get('trend_quality', 0),
                        'direction': 'SHORT'
                    }
            
            # Exit logic - check stop loss, take profit, or holding period
            if position is not None:
                for exit_idx in range(idx + 1, min(idx + holding_period + 1, len(self.df))):
                    exit_row = self.df.iloc[exit_idx]
                    exit_price = exit_row['closing_price']
                    
                    exit_triggered = False
                    exit_reason = 'HOLDING_PERIOD'
                    
                    if position['direction'] == 'LONG':
                        # Long position exits
                        pct_change = (exit_price - position['entry_price']) / position['entry_price']
                        
                        if pct_change <= -stop_loss_pct:
                            exit_price = exit_price * (1 - self.slippage)
                            exit_triggered = True
                            exit_reason = 'STOP_LOSS'
                        elif pct_change >= take_profit_pct:
                            exit_price = exit_price * (1 - self.slippage)
                            exit_triggered = True
                            exit_reason = 'TAKE_PROFIT'
                        elif exit_idx == idx + holding_period:
                            exit_price = exit_price * (1 - self.slippage)
                            exit_triggered = True
                    
                    else:  # SHORT position
                        pct_change = (position['entry_price'] - exit_price) / position['entry_price']
                        
                        if pct_change <= -stop_loss_pct:
                            exit_price = exit_price * (1 + self.slippage)
                            exit_triggered = True
                            exit_reason = 'STOP_LOSS'
                        elif pct_change >= take_profit_pct:
                            exit_price = exit_price * (1 + self.slippage)
                            exit_triggered = True
                            exit_reason = 'TAKE_PROFIT'
                        elif exit_idx == idx + holding_period:
                            exit_price = exit_price * (1 + self.slippage)
                            exit_triggered = True
                    
                    if exit_triggered:
                        # Calculate P&L
                        if position['direction'] == 'LONG':
                            gross_pnl = (exit_price - position['entry_price']) * position['shares']
                        else:
                            gross_pnl = (position['entry_price'] - exit_price) * position['shares']
                        
                        commission_cost = (position['entry_price'] + exit_price) * position['shares'] * self.commission
                        net_pnl = gross_pnl - commission_cost
                        
                        capital += net_pnl
                        
                        # Record trade
                        trade_record = {
                            'entry_idx': position['entry_idx'],
                            'exit_idx': exit_idx,
                            'entry_date': position['entry_date'],
                            'exit_date': exit_row['trade_dt'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'direction': position['direction'],
                            'signal_action': position['signal_action'],
                            'signal_confidence': position['signal_confidence'],
                            'trend_quality': position['trend_quality'],
                            'gross_pnl': gross_pnl,
                            'commission': commission_cost,
                            'net_pnl': net_pnl,
                            'return_pct': (net_pnl / (position['entry_price'] * position['shares'])) * 100,
                            'exit_reason': exit_reason,
                            'holding_days': exit_idx - position['entry_idx']
                        }
                        
                        self.trades.append(trade_record)
                        position = None
                        break
        
        self.trades_df = pd.DataFrame(self.trades)
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        
        return self.trades_df
    
    
    def calculate_performance_metrics(self):
        """
        Calculate key performance metrics including Profit Factor and Max Drawdown.
        
        Returns:
        --------
        dict : Performance metrics
        """
        if len(self.trades) == 0:
            logger.warning("No trades to analyze.")
            return {}
        
        trades_df = self.trades_df
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit metrics
        total_gross_profit = trades_df[trades_df['gross_pnl'] > 0]['gross_pnl'].sum()
        total_gross_loss = abs(trades_df[trades_df['gross_pnl'] < 0]['gross_pnl'].sum())
        
        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf
        
        total_net_profit = trades_df['net_pnl'].sum()
        
        # Returns
        avg_return_pct = trades_df['return_pct'].mean()
        avg_win_pct = trades_df[trades_df['net_pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = trades_df[trades_df['net_pnl'] < 0]['return_pct'].mean() if losing_trades > 0 else 0
        
        # Drawdown calculation
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        # Risk-adjusted metrics
        sharpe_ratio = (avg_return_pct / trades_df['return_pct'].std()) if trades_df['return_pct'].std() > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'total_gross_profit': total_gross_profit,
            'total_gross_loss': total_gross_loss,
            'profit_factor': profit_factor,
            'total_net_profit': total_net_profit,
            'avg_return_pct': avg_return_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.initial_capital + total_net_profit
        }
        
        logger.info(f"Performance Metrics: Profit Factor={profit_factor:.2f}, "
                   f"Win Rate={win_rate:.1f}%, Max DD={max_drawdown_pct:.2f}%")
        
        return metrics
    
    
    def identify_false_positives_negatives(self, profit_threshold_pct: float = 2.0):
        """
        Identify false positives (signals that lost money) and 
        false negatives (missed opportunities).
        
        Parameters:
        -----------
        profit_threshold_pct : float
            Minimum profit % to consider signal as true positive (default 2%)
        
        Returns:
        --------
        tuple : (false_positives_df, false_negatives_df)
        """
        logger.info("Identifying false positives and false negatives...")
        
        # FALSE POSITIVES: Signals that resulted in losing trades
        false_positives = self.trades_df[self.trades_df['return_pct'] < -profit_threshold_pct].copy()
        
        # FALSE NEGATIVES: Periods with good price moves but no signal
        # Look for periods where price moved significantly without a signal
        false_negatives_list = []
        
        for idx in range(len(self.df) - 5):
            row = self.df.iloc[idx]
            
            # Skip if there was a signal
            if row['signal_action'] in ['STRONG BUY', 'MODERATE BUY', 'STRONG SELL', 'MODERATE SELL']:
                continue
            
            # Check if price moved significantly in next 5 days
            future_prices = self.df.iloc[idx+1:idx+6]['closing_price']
            if len(future_prices) == 0:
                continue
            
            max_gain = ((future_prices.max() - row['closing_price']) / row['closing_price']) * 100
            max_loss = ((future_prices.min() - row['closing_price']) / row['closing_price']) * 100
            
            # Identify missed bullish opportunities
            if max_gain > profit_threshold_pct * 2:
                false_negatives_list.append({
                    'idx': idx,
                    'trade_dt': row['trade_dt'],
                    'signal_action': row['signal_action'],
                    'signal_type': row['signal_type'],
                    'closing_price': row['closing_price'],
                    'missed_direction': 'BULLISH',
                    'missed_gain_pct': max_gain,
                    'trend_quality': row.get('trend_quality', 0),
                    'tradeable': row.get('tradeable', False),
                    'sideways_market': row.get('sideways_market', False),
                    'choppy_market': row.get('choppy_market', False),
                    'weak_trend': row.get('weak_trend', False)
                })
            
            # Identify missed bearish opportunities
            if max_loss < -profit_threshold_pct * 2:
                false_negatives_list.append({
                    'idx': idx,
                    'trade_dt': row['trade_dt'],
                    'signal_action': row['signal_action'],
                    'signal_type': row['signal_type'],
                    'closing_price': row['closing_price'],
                    'missed_direction': 'BEARISH',
                    'missed_gain_pct': abs(max_loss),
                    'trend_quality': row.get('trend_quality', 0),
                    'tradeable': row.get('tradeable', False),
                    'sideways_market': row.get('sideways_market', False),
                    'choppy_market': row.get('choppy_market', False),
                    'weak_trend': row.get('weak_trend', False)
                })
        
        self.false_negatives = pd.DataFrame(false_negatives_list)
        self.false_positives = false_positives
        
        logger.info(f"False Positives: {len(self.false_positives)}, "
                   f"False Negatives: {len(self.false_negatives)}")
        
        return self.false_positives, self.false_negatives
    
    
    def analyze_false_positives(self):
        """
        Analyze each false positive and identify the missing piece that 
        could have prevented the bad trade.
        
        Returns:
        --------
        pd.DataFrame : False positives with 'missing_piece' column
        """
        logger.info("Analyzing false positives...")
        
        if len(self.false_positives) == 0:
            logger.info("No false positives to analyze.")
            return self.false_positives
        
        missing_pieces = []
        
        for _, trade in self.false_positives.iterrows():
            entry_idx = trade['entry_idx']
            entry_row = self.df.iloc[entry_idx]
            
            reasons = []
            
            # Check trend quality
            if entry_row.get('trend_quality', 1) < 0.6:
                reasons.append(f"Low trend quality ({entry_row.get('trend_quality', 0):.2f})")
            
            # Check cloud thickness
            if entry_row.get('kumo_thickness_pct', 1) < 0.8:
                reasons.append(f"Thin cloud ({entry_row.get('kumo_thickness_pct', 0):.2f}%)")
            
            # Check TK separation
            if entry_row.get('tk_separation_pct', 1) < 0.5:
                reasons.append(f"Weak TK separation ({entry_row.get('tk_separation_pct', 0):.2f}%)")
            
            # Check if price too close to cloud
            if entry_row.get('price_kumo_distance_pct', 5) < 1.5:
                reasons.append(f"Price too close to cloud ({entry_row.get('price_kumo_distance_pct', 0):.2f}%)")
            
            # Check for false breakout risk
            if entry_row.get('false_breakout_risk', False):
                reasons.append("False breakout risk detected")
            
            # Check for weak trend
            if entry_row.get('weak_trend', False):
                reasons.append("Weak trend conditions")
            
            # Check Chikou clarity
            if not entry_row.get('chikou_clear', True):
                reasons.append("Chikou span not clear")
            
            # Check volume (if available)
            if 'total_trading_volume' in entry_row:
                avg_volume = self.df.iloc[max(0, entry_idx-20):entry_idx]['total_trading_volume'].mean()
                if entry_row['total_trading_volume'] < avg_volume * 0.8:
                    reasons.append("Low volume on entry")
            
            # Check if signal was on high volatility day
            if entry_row.get('price_range_pct', 0) > entry_row.get('avg_range_pct', 2) * 1.5:
                reasons.append("High volatility day")
            
            # Check opposing signals
            if trade['direction'] == 'LONG':
                if entry_row.get('tenkan', 0) < entry_row.get('kijun', 0):
                    reasons.append("Tenkan below Kijun (bearish momentum)")
            else:
                if entry_row.get('tenkan', 0) > entry_row.get('kijun', 0):
                    reasons.append("Tenkan above Kijun (bullish momentum)")
            
            # Compile missing piece
            if len(reasons) == 0:
                missing_piece = "Signal met all criteria - market reversal or external event"
            else:
                missing_piece = " | ".join(reasons)
            
            missing_pieces.append(missing_piece)
        
        self.false_positives['missing_piece'] = missing_pieces
        
        logger.info("False positive analysis complete.")
        return self.false_positives
    
    
    def analyze_false_negatives(self):
        """
        Analyze each false negative and identify what was missing that 
        prevented signal generation.
        
        Returns:
        --------
        pd.DataFrame : False negatives with 'missing_piece' column
        """
        logger.info("Analyzing false negatives...")
        
        if len(self.false_negatives) == 0:
            logger.info("No false negatives to analyze.")
            return self.false_negatives
        
        missing_pieces = []
        
        for _, missed in self.false_negatives.iterrows():
            idx = missed['idx']
            row = self.df.iloc[idx]
            
            reasons = []
            
            # Check why signal wasn't generated
            if missed['sideways_market']:
                reasons.append("Flagged as sideways market")
            
            if missed['choppy_market']:
                reasons.append("Flagged as choppy market")
            
            if missed['weak_trend']:
                reasons.append("Flagged as weak trend")
            
            if not missed['tradeable']:
                reasons.append("Market not tradeable")
            
            # Check specific thresholds
            if row.get('trend_quality', 0) < 0.5:
                reasons.append(f"Trend quality too low ({row.get('trend_quality', 0):.2f} < 0.5)")
            
            if row.get('kumo_thickness_pct', 0) < 1.0:
                reasons.append(f"Cloud too thin ({row.get('kumo_thickness_pct', 0):.2f}% < 1.0%)")
            
            if row.get('tk_separation_pct', 0) < 0.5:
                reasons.append(f"TK separation too small ({row.get('tk_separation_pct', 0):.2f}% < 0.5%)")
            
            # Check if price was in cloud
            if (row['closing_price'] >= row.get('kumo_bottom', 0) and 
                row['closing_price'] <= row.get('kumo_top', 0)):
                reasons.append("Price was inside cloud (neutral zone)")
            
            # Check distance from cloud
            if row.get('price_kumo_distance_pct', 0) < 1.5:
                reasons.append(f"Price too close to cloud ({row.get('price_kumo_distance_pct', 0):.2f}% < 1.5%)")
            
            # Compile missing piece
            if len(reasons) == 0:
                missing_piece = "Thresholds were marginally not met - consider relaxing filters"
            else:
                missing_piece = " | ".join(reasons)
            
            missing_pieces.append(missing_piece)
        
        self.false_negatives['missing_piece'] = missing_pieces
        
        logger.info("False negative analysis complete.")
        return self.false_negatives
    
    
    def generate_recommendations(self):
        """
        Summarize missing pieces and generate actionable recommendations 
        for improving the Ichimoku Scanner.
        
        Returns:
        --------
        dict : Recommendations with specific fixes
        """
        logger.info("Generating recommendations...")
        
        recommendations = {
            'summary': {},
            'false_positive_fixes': [],
            'false_negative_fixes': [],
            'priority_fixes': []
        }
        
        # Analyze false positives
        if len(self.false_positives) > 0:
            fp_reasons = []
            for piece in self.false_positives['missing_piece']:
                fp_reasons.extend([r.strip() for r in piece.split('|')])
            
            fp_counter = Counter(fp_reasons)
            recommendations['summary']['false_positive_patterns'] = dict(fp_counter.most_common(10))
            
            # Generate fixes for common issues
            for reason, count in fp_counter.most_common(5):
                if 'trend quality' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Increase trend_quality threshold from 0.5 to 0.6-0.65 for STRONG signals',
                        'code_change': "self.df['tradeable'] = (... & (self.df['trend_quality'] > 0.6))"
                    })
                
                elif 'thin cloud' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Increase kumo_thickness_pct threshold from 0.8% to 1.2%',
                        'code_change': "(self.df['kumo_thickness_pct'] > 1.2)"
                    })
                
                elif 'tk separation' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Require stronger TK separation: increase from 0.5% to 0.8%',
                        'code_change': "(self.df['tk_separation_pct'] > 0.8)"
                    })
                
                elif 'close to cloud' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Require price to be further from cloud: increase from 1.5% to 2.5%',
                        'code_change': "(self.df['price_kumo_distance_pct'] > 2.5)"
                    })
                
                elif 'false breakout risk' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Add volume confirmation: require above-average volume on breakout',
                        'code_change': "Add: (volume > volume.rolling(20).mean() * 1.2)"
                    })
                
                elif 'weak trend' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Strengthen weak_trend detection: require consistent direction over 3+ bars',
                        'code_change': "Check: (tenkan > kijun).rolling(3).sum() >= 2"
                    })
                
                elif 'chikou' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Make Chikou clarity mandatory for all signals',
                        'code_change': "Add: & (self.df['chikou_clear'])"
                    })
                
                elif 'low volume' in reason.lower():
                    recommendations['false_positive_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Add volume filter: require volume > 20-day average',
                        'code_change': "volume > volume.rolling(20).mean()"
                    })
        
        # Analyze false negatives
        if len(self.false_negatives) > 0:
            fn_reasons = []
            for piece in self.false_negatives['missing_piece']:
                fn_reasons.extend([r.strip() for r in piece.split('|')])
            
            fn_counter = Counter(fn_reasons)
            recommendations['summary']['false_negative_patterns'] = dict(fn_counter.most_common(10))
            
            # Generate fixes for common issues
            for reason, count in fn_counter.most_common(5):
                if 'trend quality too low' in reason.lower():
                    recommendations['false_negative_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Lower trend_quality threshold for WEAK signals from 0.5 to 0.4',
                        'code_change': "Add WEAK BUY tier: (self.df['trend_quality'] > 0.4)"
                    })
                
                elif 'cloud too thin' in reason.lower():
                    recommendations['false_negative_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Allow thinner clouds (0.6%) for WEAK signals in strong trends',
                        'code_change': "Add: | ((kumo_thickness_pct > 0.6) & (price_kumo_distance_pct > 3))"
                    })
                
                elif 'sideways market' in reason.lower():
                    recommendations['false_negative_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Refine sideways detection: allow signals if price breaks out decisively',
                        'code_change': "Modify sideways to exclude: (price_kumo_distance_pct > 2.5)"
                    })
                
                elif 'choppy market' in reason.lower():
                    recommendations['false_negative_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Reduce choppy sensitivity: increase volatility threshold multiplier to 2.0x',
                        'code_change': "(price_range_pct > avg_range_pct * 2.0) instead of 1.5x"
                    })
                
                elif 'inside cloud' in reason.lower():
                    recommendations['false_negative_fixes'].append({
                        'issue': reason,
                        'frequency': count,
                        'fix': 'Add cloud breakout anticipation: flag when price is 0.5% from cloud edge',
                        'code_change': "Add: 'near_cloud_breakout' = (distance_to_cloud < 0.5%)"
                    })
        
        # Priority fixes (most impactful)
        metrics = self.calculate_performance_metrics()
        
        if metrics.get('win_rate_pct', 0) < 40:
            recommendations['priority_fixes'].append({
                'priority': 'HIGH',
                'issue': f"Low win rate: {metrics.get('win_rate_pct', 0):.1f}%",
                'fix': 'Tighten entry criteria: Increase trend_quality to 0.65+ and require Chikou clarity'
            })
        
        if metrics.get('profit_factor', 0) < 1.2:
            recommendations['priority_fixes'].append({
                'priority': 'HIGH',
                'issue': f"Low profit factor: {metrics.get('profit_factor', 0):.2f}",
                'fix': 'Focus on STRONG signals only, filter out MODERATE/WEAK until profit factor > 1.5'
            })
        
        if abs(metrics.get('max_drawdown_pct', 0)) > 15:
            recommendations['priority_fixes'].append({
                'priority': 'HIGH',
                'issue': f"High drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%",
                'fix': 'Add drawdown protection: pause signals after 2 consecutive losses or 10% drawdown'
            })
        
        # Summary statistics
        recommendations['summary']['total_false_positives'] = len(self.false_positives)
        recommendations['summary']['total_false_negatives'] = len(self.false_negatives)
        recommendations['summary']['false_positive_rate'] = (
            len(self.false_positives) / len(self.trades_df) * 100 
            if len(self.trades_df) > 0 else 0
        )
        
        logger.info(f"Recommendations generated: {len(recommendations['false_positive_fixes'])} FP fixes, "
                   f"{len(recommendations['false_negative_fixes'])} FN fixes")
        
        self.recommendations = recommendations
        return recommendations
    
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report with all analysis results.
        
        Returns:
        --------
        dict : Complete analysis report
        """
        logger.info("Generating comprehensive report...")
        
        metrics = self.calculate_performance_metrics()
        
        report = {
            'performance_metrics': metrics,
            'recommendations': self.recommendations,
            'trade_summary': {
                'total_trades': len(self.trades_df),
                'winning_trades': len(self.trades_df[self.trades_df['net_pnl'] > 0]),
                'losing_trades': len(self.trades_df[self.trades_df['net_pnl'] < 0]),
                'avg_holding_days': self.trades_df['holding_days'].mean() if len(self.trades_df) > 0 else 0
            },
            'signal_breakdown': {
                'strong_buy_trades': len(self.trades_df[self.trades_df['signal_action'] == 'STRONG BUY']),
                'moderate_buy_trades': len(self.trades_df[self.trades_df['signal_action'] == 'MODERATE BUY']),
                'strong_sell_trades': len(self.trades_df[self.trades_df['signal_action'] == 'STRONG SELL']),
                'moderate_sell_trades': len(self.trades_df[self.trades_df['signal_action'] == 'MODERATE SELL'])
            },
            'false_signals': {
                'false_positives': len(self.false_positives),
                'false_negatives': len(self.false_negatives)
            }
        }
        
        # Add signal-specific performance
        if len(self.trades_df) > 0:
            for signal in ['STRONG BUY', 'MODERATE BUY', 'STRONG SELL', 'MODERATE SELL']:
                signal_trades = self.trades_df[self.trades_df['signal_action'] == signal]
                if len(signal_trades) > 0:
                    report['signal_breakdown'][f'{signal}_win_rate'] = (
                        len(signal_trades[signal_trades['net_pnl'] > 0]) / len(signal_trades) * 100
                    )
                    report['signal_breakdown'][f'{signal}_avg_return'] = signal_trades['return_pct'].mean()
        
        logger.info("Comprehensive report generated.")
        return report
    
    
    def print_summary(self):
        """
        Print a formatted summary of the backtest results and recommendations.
        """
        report = self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("ICHIMOKU BACKTEST SUMMARY")
        print("="*80)
        
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 80)
        metrics = report['performance_metrics']
        print(f"Total Trades:           {metrics.get('total_trades', 0)}")
        print(f"Winning Trades:         {metrics.get('winning_trades', 0)} ({metrics.get('win_rate_pct', 0):.1f}%)")
        print(f"Losing Trades:          {metrics.get('losing_trades', 0)}")
        print(f"Profit Factor:          {metrics.get('profit_factor', 0):.2f}")
        print(f"Total Net Profit:       ${metrics.get('total_net_profit', 0):,.2f}")
        print(f"Max Drawdown:           ${metrics.get('max_drawdown', 0):,.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)")
        print(f"Avg Return per Trade:   {metrics.get('avg_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}")
        
        print("\n📈 SIGNAL BREAKDOWN")
        print("-" * 80)
        signal_breakdown = report['signal_breakdown']
        for signal in ['STRONG BUY', 'MODERATE BUY', 'STRONG SELL', 'MODERATE SELL']:
            count = signal_breakdown.get(f'{signal.lower().replace(" ", "_")}_trades', 0)
            win_rate = signal_breakdown.get(f'{signal}_win_rate', 0)
            avg_return = signal_breakdown.get(f'{signal}_avg_return', 0)
            if count > 0:
                print(f"{signal:20s}: {count:3d} trades, {win_rate:5.1f}% win rate, {avg_return:6.2f}% avg return")
        
        print("\n⚠️  FALSE SIGNALS")
        print("-" * 80)
        print(f"False Positives:        {report['false_signals']['false_positives']}")
        print(f"False Negatives:        {report['false_signals']['false_negatives']}")
        
        if len(self.false_positives) > 0:
            print(f"False Positive Rate:    {report['recommendations']['summary']['false_positive_rate']:.1f}%")
        
        print("\n🔧 TOP RECOMMENDATIONS (False Positives)")
        print("-" * 80)
        if 'false_positive_fixes' in self.recommendations:
            for i, fix in enumerate(self.recommendations['false_positive_fixes'][:5], 1):
                print(f"\n{i}. Issue: {fix['issue']}")
                print(f"   Frequency: {fix['frequency']} occurrences")
                print(f"   Fix: {fix['fix']}")
                print(f"   Code: {fix['code_change']}")
        
        print("\n🔧 TOP RECOMMENDATIONS (False Negatives)")
        print("-" * 80)
        if 'false_negative_fixes' in self.recommendations:
            for i, fix in enumerate(self.recommendations['false_negative_fixes'][:5], 1):
                print(f"\n{i}. Issue: {fix['issue']}")
                print(f"   Frequency: {fix['frequency']} occurrences")
                print(f"   Fix: {fix['fix']}")
                print(f"   Code: {fix['code_change']}")
        
        print("\n🎯 PRIORITY FIXES")
        print("-" * 80)
        if 'priority_fixes' in self.recommendations:
            for i, fix in enumerate(self.recommendations['priority_fixes'], 1):
                print(f"\n{i}. [{fix['priority']}] {fix['issue']}")
                print(f"   Action: {fix['fix']}")
        
        print("\n" + "="*80)
        print("END OF REPORT")
        print("="*80 + "\n")
    
    
    def export_results(self, base_filename: str = 'ichimoku_backtest'):
        """
        Export all results to CSV files.
        
        Parameters:
        -----------
        base_filename : str
            Base filename for exports
        """
        logger.info(f"Exporting results with base filename: {base_filename}")
        
        # Export trades
        if len(self.trades_df) > 0:
            self.trades_df.to_csv(f"{base_filename}_trades.csv", index=False)
            logger.info(f"Exported trades to {base_filename}_trades.csv")
        
        # Export false positives
        if len(self.false_positives) > 0:
            self.false_positives.to_csv(f"{base_filename}_false_positives.csv", index=False)
            logger.info(f"Exported false positives to {base_filename}_false_positives.csv")
        
        # Export false negatives
        if len(self.false_negatives) > 0:
            self.false_negatives.to_csv(f"{base_filename}_false_negatives.csv", index=False)
            logger.info(f"Exported false negatives to {base_filename}_false_negatives.csv")
        
        # Export recommendations
        if self.recommendations:
            import json
            with open(f"{base_filename}_recommendations.json", 'w') as f:
                json.dump(self.recommendations, f, indent=2)
            logger.info(f"Exported recommendations to {base_filename}_recommendations.json")
        
        logger.info("Export complete.")
    
    
    def run_complete_backtest(self, holding_period: int = 5, stop_loss_pct: float = 0.05,
                             take_profit_pct: float = 0.10, profit_threshold_pct: float = 2.0):
        """
        Run complete backtesting workflow:
        1. Run backtest
        2. Calculate metrics
        3. Identify false positives/negatives
        4. Analyze them
        5. Generate recommendations
        6. Print summary
        
        Parameters:
        -----------
        holding_period : int
            Days to hold position (default 5)
        stop_loss_pct : float
            Stop loss percentage (default 5%)
        take_profit_pct : float
            Take profit percentage (default 10%)
        profit_threshold_pct : float
            Threshold for false positive/negative (default 2%)
        
        Returns:
        --------
        dict : Complete analysis report
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE BACKTEST WORKFLOW")
        logger.info("=" * 80)
        
        # Step 1: Run backtest
        self.run_backtest(holding_period, stop_loss_pct, take_profit_pct)
        
        # Step 2: Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Step 3: Identify false signals
        self.identify_false_positives_negatives(profit_threshold_pct)
        
        # Step 4: Analyze false positives
        self.analyze_false_positives()
        
        # Step 5: Analyze false negatives
        self.analyze_false_negatives()
        
        # Step 6: Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Step 7: Generate and print report
        report = self.generate_comprehensive_report()
        self.print_summary()
        
        logger.info("=" * 80)
        logger.info("BACKTEST WORKFLOW COMPLETE")
        logger.info("=" * 80)
        
        return report

