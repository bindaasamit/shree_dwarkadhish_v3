#------------------------------------------------------------------------------
###            Dual EMA Crossover Strategy Scanner - Daily Timeframe
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import Modules
from config import cfg_nifty as cfg_nifty
from config import cfg_vars as cfg_vars



#---------------------------------------------------------------------------------------------
#                        Dual EMA Crossover Strategy Class
#---------------------------------------------------------------------------------------------

class EMA_XY_Scanner:
    """
    Implements dual EMA crossover strategy (default 8/21) for daily timeframe.
    Configurable EMA periods for flexibility (can use 13/50 or other combinations).
    
    Features:
    - Parameterizable fast/slow EMA periods
    - Multiple confirmation filters (volume, RSI, trend strength)
    - Tiered signal quality (STRONG/MODERATE/WEAK)
    - Risk management metrics (stop loss, targets)
    """

    def __init__(self, df: pd.DataFrame, fast_period=8, slow_period=21):
        """
        Initialize with OHLC DataFrame and EMA periods.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data (columns: closing_price, high_price, low_price, total_trading_volume)
        fast_period : int
            Fast EMA period (default: 8)
        slow_period : int
            Slow EMA period (default: 21)
        """
        logger.info(f"Initializing EMA_CrossOver_Scanner with {fast_period}/{slow_period} EMA...")
        self.df = df.copy().reset_index(drop=True)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signals_generated = False
        
        # Validate EMA periods
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        
        logger.info(f"Scanner initialized with EMA periods: Fast={fast_period}, Slow={slow_period}")

    # ========================================================================
    # CORE EMA CALCULATIONS
    # ========================================================================

    def compute_dual_ema(self):
        """Compute dual EMA crossover indicators"""
        logger.info(f"Computing {self.fast_period}/{self.slow_period} EMA crossovers...")

        close = self.df['closing_price']

        # Calculate EMAs with configurable periods
        self.df[f'ema_{self.fast_period}'] = close.ewm(span=self.fast_period, adjust=False).mean()
        self.df[f'ema_{self.slow_period}'] = close.ewm(span=self.slow_period, adjust=False).mean()

        ema_fast = self.df[f'ema_{self.fast_period}']
        ema_slow = self.df[f'ema_{self.slow_period}']
        
        ema_fast_prev = ema_fast.shift(1)
        ema_slow_prev = ema_slow.shift(1)

        # EMA separation analysis (trend strength indicator)
        self.df['ema_separation'] = abs(ema_fast - ema_slow)
        self.df['ema_separation_pct'] = (self.df['ema_separation'] / close) * 100

        # EMA slope analysis (momentum indicators)
        fast_shifted = ema_fast.shift(self.fast_period)
        slow_shifted = ema_slow.shift(self.slow_period)
        
        self.df[f'ema_{self.fast_period}_slope'] = np.where(
            fast_shifted.notna() & (fast_shifted != 0),
            (ema_fast - fast_shifted) / fast_shifted * 100,
            0
        )
        self.df[f'ema_{self.slow_period}_slope'] = np.where(
            slow_shifted.notna() & (slow_shifted != 0),
            (ema_slow - slow_shifted) / slow_shifted * 100,
            0
        )

        # Average slope for trend momentum
        self.df['ema_avg_slope'] = (
            abs(self.df[f'ema_{self.fast_period}_slope']) +
            abs(self.df[f'ema_{self.slow_period}_slope'])
        ) / 2

        # Slope direction alignment
        self.df['ema_slopes_aligned_bullish'] = (
            (self.df[f'ema_{self.fast_period}_slope'] > 0) &
            (self.df[f'ema_{self.slow_period}_slope'] > 0)
        )
        self.df['ema_slopes_aligned_bearish'] = (
            (self.df[f'ema_{self.fast_period}_slope'] < 0) &
            (self.df[f'ema_{self.slow_period}_slope'] < 0)
        )

        # Crossover detection
        self.df['ema_golden_cross'] = (
            (ema_fast_prev <= ema_slow_prev) & (ema_fast > ema_slow)
        )
        self.df['ema_death_cross'] = (
            (ema_fast_prev >= ema_slow_prev) & (ema_fast < ema_slow)
        )

        # Position tracking (above/below)
        self.df['ema_bullish_position'] = ema_fast > ema_slow
        self.df['ema_bearish_position'] = ema_fast < ema_slow

        # Price position relative to EMAs
        self.df['price_above_both_emas'] = (
            (close > ema_fast) & (close > ema_slow)
        )
        self.df['price_below_both_emas'] = (
            (close < ema_fast) & (close < ema_slow)
        )

        # EMA bounce/rejection detection (for entries)
        self.df['bullish_ema_bounce'] = (
            self.df['ema_bullish_position'] &
            (close.shift(1) < ema_fast.shift(1)) &
            (close > ema_fast)
        )
        self.df['bearish_ema_rejection'] = (
            self.df['ema_bearish_position'] &
            (close.shift(1) > ema_fast.shift(1)) &
            (close < ema_fast)
        )

        logger.info("EMA crossovers computed successfully.")

    # ========================================================================
    # MARKET CONDITION FILTERS
    # ========================================================================

    def compute_market_filters(self):
        """Detect sideways/choppy market conditions"""
        logger.info("Computing market condition filters...")

        close = self.df['closing_price']

        # Sideways market detection
        self.df['ema_sideways_market'] = (
            (self.df['ema_separation_pct'] < 0.5) &
            (self.df['ema_avg_slope'] < 0.3)
        )

        # Whipsaw risk (frequent crossovers)
        crossover_count = (
            (self.df['ema_golden_cross'] | self.df['ema_death_cross'])
            .rolling(10).sum()
        )
        self.df['ema_whipsaw_risk'] = (
            (self.df['ema_separation_pct'] < 0.8) &
            (crossover_count > 3)
        )

        # Trending market (opposite of sideways)
        self.df['ema_trending_market'] = ~self.df['ema_sideways_market']

        logger.info("Market filters computed.")

    # ========================================================================
    # CONFIRMATION INDICATORS
    # ========================================================================

    def compute_rsi(self, period=14):
        """Calculate RSI for momentum confirmation"""
        logger.info("Computing RSI...")
        
        delta = self.df['closing_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        self.df['rsi_overbought'] = self.df['rsi'] > 70
        self.df['rsi_oversold'] = self.df['rsi'] < 30
        self.df['rsi_neutral'] = (self.df['rsi'] >= 40) & (self.df['rsi'] <= 60)
        
        logger.info("RSI computed.")

    def compute_volume_indicators(self):
        """Volume analysis for confirmation"""
        logger.info("Computing volume indicators...")
        
        # Average volume
        self.df['avg_volume_20'] = self.df['total_trading_volume'].rolling(20).mean()
        
        # Volume ratio
        self.df['volume_ratio'] = (
            self.df['total_trading_volume'] / self.df['avg_volume_20']
        )
        
        # Volume confirmation
        self.df['volume_above_average'] = self.df['volume_ratio'] > 1.0
        self.df['volume_surge'] = self.df['volume_ratio'] > 1.5
        
        logger.info("Volume indicators computed.")

    def compute_atr(self, period=14):
        """Calculate ATR for stop loss placement"""
        logger.info("Computing ATR...")
        
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['closing_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['closing_price'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=period).mean()
        
        # ATR as % of price
        self.df['atr_pct'] = (self.df['atr'] / self.df['closing_price']) * 100
        
        logger.info("ATR computed.")

    # ========================================================================
    # TREND STRENGTH COMPOSITE
    # ========================================================================

    def compute_trend_strength(self):
        """Calculate composite trend strength score"""
        logger.info("Computing trend strength...")
        
        # Component 1: EMA separation (25% weight)
        separation_score = (self.df['ema_separation_pct'] / 2).clip(0, 1)
        
        # Component 2: Slope alignment (30% weight)
        slope_alignment_score = (
            self.df['ema_slopes_aligned_bullish'].astype(int) +
            self.df['ema_slopes_aligned_bearish'].astype(int)
        )
        
        # Component 3: Slope magnitude (25% weight)
        slope_magnitude_score = (self.df['ema_avg_slope'] / 2).clip(0, 1)
        
        # Component 4: Volume confirmation (20% weight)
        volume_score = self.df['volume_ratio'].clip(0.5, 2) / 2
        
        # Composite trend strength
        self.df['trend_strength'] = (
            separation_score * 0.25 +
            slope_alignment_score * 0.30 +
            slope_magnitude_score * 0.25 +
            volume_score * 0.20
        ).clip(0, 1)
        
        logger.info("Trend strength computed.")

    # ========================================================================
    # FILTERED SIGNALS
    # ========================================================================

    def compute_filtered_signals(self):
        """Generate filtered buy/sell signals"""
        logger.info("Computing filtered crossover signals...")

        # Base crossover conditions
        golden_cross_base = self.df['ema_golden_cross']
        death_cross_base = self.df['ema_death_cross']

        # Filter conditions
        filter_conditions = (
            (~self.df['ema_sideways_market']) &
            (~self.df['ema_whipsaw_risk'])
        )

        # Filtered crossovers
        self.df['ema_golden_cross_filtered'] = golden_cross_base & filter_conditions
        self.df['ema_death_cross_filtered'] = death_cross_base & filter_conditions

        # Slope-aligned signals
        self.df['ema_golden_cross_strong'] = (
            self.df['ema_golden_cross_filtered'] &
            self.df['ema_slopes_aligned_bullish']
        )
        self.df['ema_death_cross_strong'] = (
            self.df['ema_death_cross_filtered'] &
            self.df['ema_slopes_aligned_bearish']
        )

        logger.info("Filtered signals computed.")

    # ========================================================================
    # STOP LOSS & TARGET CALCULATIONS
    # ========================================================================

    def compute_risk_management(self):
        """Calculate stop loss and target levels"""
        logger.info("Computing risk management levels...")

        ema_fast = self.df[f'ema_{self.fast_period}']
        ema_slow = self.df[f'ema_{self.slow_period}']
        close = self.df['closing_price']

        # Stop loss levels (1.5 ATR or below slow EMA)
        self.df['stop_loss_long'] = (
            close - (self.df['atr'] * 1.5)
        ).clip(lower=ema_slow * 0.95)  # Never tighter than 5% below slow EMA

        self.df['stop_loss_short'] = (
            close + (self.df['atr'] * 1.5)
        ).clip(upper=ema_slow * 1.05)  # Never tighter than 5% above slow EMA

        # Target levels (2:1 risk-reward)
        self.df['target_long'] = close + (close - self.df['stop_loss_long']) * 2
        self.df['target_short'] = close - (self.df['stop_loss_short'] - close) * 2

        # Risk-reward ratio
        self.df['risk_reward_long'] = (
            (self.df['target_long'] - close) / (close - self.df['stop_loss_long'])
        )
        self.df['risk_reward_short'] = (
            (close - self.df['target_short']) / (self.df['stop_loss_short'] - close)
        )

        logger.info("Risk management levels computed.")

    # ========================================================================
    # MASTER COMPUTE METHOD
    # ========================================================================

    def compute_all_indicators(self):
        """Compute all indicators in correct order"""
        logger.info("="*70)
        logger.info("Starting comprehensive indicator calculation...")
        
        # Core calculations
        self.compute_dual_ema()
        self.compute_market_filters()
        
        # Confirmation indicators
        self.compute_rsi()
        self.compute_volume_indicators()
        self.compute_atr()
        
        # Composite calculations
        self.compute_trend_strength()
        self.compute_filtered_signals()
        self.compute_risk_management()
        
        self.signals_generated = True
        logger.info("All indicators computed successfully!")
        
        return self.df

    # ========================================================================
    # SIGNAL INTERPRETATION
    # ========================================================================

    def interpret_signal(self, index=-1):
        """
        Interpret trading signal for a specific row.
        
        Parameters:
        -----------
        index : int
            Row index to interpret (-1 for latest)
        
        Returns:
        --------
        dict : Signal interpretation
        """
        if not self.signals_generated:
            logger.warning("Indicators not computed. Call compute_all_indicators() first.")
            return {
                'action': 'ERROR',
                'reason': 'Indicators not computed',
                'confidence': 'N/A',
                'signal_quality': 0.0,
                'details': 'Run compute_all_indicators() first'
            }
        
        latest = self.df.iloc[index]
        
        # ================================================================
        # MARKET CONDITION FILTERS
        # ================================================================
        
        if latest['ema_sideways_market']:
            return {
                'action': 'WAIT',
                'reason': 'Sideways market conditions',
                'confidence': 'N/A',
                'signal_quality': latest['trend_strength'],
                'details': f"EMA separation: {latest['ema_separation_pct']:.2f}%"
            }
        
        if latest['ema_whipsaw_risk']:
            return {
                'action': 'WAIT',
                'reason': 'High whipsaw risk',
                'confidence': 'N/A',
                'signal_quality': latest['trend_strength'],
                'details': 'Too many recent crossovers'
            }
        
        # ================================================================
        # BULLISH SIGNALS
        # ================================================================
        
        # STRONG BUY - Golden cross with all confirmations
        if (latest['ema_golden_cross_strong'] and
            latest['trend_strength'] > 0.5 and
            not latest['rsi_overbought'] and
            latest['volume_above_average']):
            return {
                'action': 'STRONG BUY',
                'reason': f'{self.fast_period}/{self.slow_period} EMA golden cross with strong confirmations',
                'confidence': 'HIGH',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}, RSI: {latest['rsi']:.1f}, Volume ratio: {latest['volume_ratio']:.2f}",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_long'],
                'target': latest['target_long']
            }
        
        # MODERATE BUY - Golden cross with some confirmations
        if (latest['ema_golden_cross_filtered'] and
            latest['trend_strength'] > 0.4 and
            latest['ema_slopes_aligned_bullish']):
            warnings = []
            if latest['rsi_overbought']:
                warnings.append("RSI overbought")
            if not latest['volume_above_average']:
                warnings.append("Below average volume")
            
            return {
                'action': 'MODERATE BUY',
                'reason': f'{self.fast_period}/{self.slow_period} EMA golden cross with moderate confirmations',
                'confidence': 'MEDIUM',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}, RSI: {latest['rsi']:.1f}",
                'warnings': "; ".join(warnings) if warnings else "None",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_long'],
                'target': latest['target_long']
            }
        
        # WEAK BUY - Golden cross without strong confirmations
        if (latest['ema_golden_cross_filtered'] and
            latest['trend_strength'] > 0.3):
            return {
                'action': 'WEAK BUY',
                'reason': f'{self.fast_period}/{self.slow_period} EMA golden cross with weak confirmations',
                'confidence': 'LOW',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}",
                'warnings': "Low trend strength - use tight stop loss",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_long'],
                'target': latest['target_long']
            }
        
        # ================================================================
        # BEARISH SIGNALS
        # ================================================================
        
        # STRONG SELL - Death cross with all confirmations
        if (latest['ema_death_cross_strong'] and
            latest['trend_strength'] > 0.5 and
            not latest['rsi_oversold'] and
            latest['volume_above_average']):
            return {
                'action': 'STRONG SELL',
                'reason': f'{self.fast_period}/{self.slow_period} EMA death cross with strong confirmations',
                'confidence': 'HIGH',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}, RSI: {latest['rsi']:.1f}, Volume ratio: {latest['volume_ratio']:.2f}",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_short'],
                'target': latest['target_short']
            }
        
        # MODERATE SELL - Death cross with some confirmations
        if (latest['ema_death_cross_filtered'] and
            latest['trend_strength'] > 0.4 and
            latest['ema_slopes_aligned_bearish']):
            warnings = []
            if latest['rsi_oversold']:
                warnings.append("RSI oversold")
            if not latest['volume_above_average']:
                warnings.append("Below average volume")
            
            return {
                'action': 'MODERATE SELL',
                'reason': f'{self.fast_period}/{self.slow_period} EMA death cross with moderate confirmations',
                'confidence': 'MEDIUM',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}, RSI: {latest['rsi']:.1f}",
                'warnings': "; ".join(warnings) if warnings else "None",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_short'],
                'target': latest['target_short']
            }
        
        # WEAK SELL - Death cross without strong confirmations
        if (latest['ema_death_cross_filtered'] and
            latest['trend_strength'] > 0.3):
            return {
                'action': 'WEAK SELL',
                'reason': f'{self.fast_period}/{self.slow_period} EMA death cross with weak confirmations',
                'confidence': 'LOW',
                'signal_quality': latest['trend_strength'],
                'details': f"Trend strength: {latest['trend_strength']:.2f}",
                'warnings': "Low trend strength - use tight stop loss",
                'entry': latest['closing_price'],
                'stop_loss': latest['stop_loss_short'],
                'target': latest['target_short']
            }
        
        # ================================================================
        # HOLD SIGNALS
        # ================================================================
        
        # HOLD LONG - In uptrend, no exit signal
        if (latest['ema_bullish_position'] and
            latest['price_above_both_emas'] and
            latest['ema_slopes_aligned_bullish']):
            return {
                'action': 'HOLD LONG',
                'reason': 'Uptrend intact - maintain long positions',
                'confidence': 'N/A',
                'signal_quality': latest['trend_strength'],
                'details': f"Fast EMA above Slow EMA, price above both EMAs"
            }
        
        # HOLD SHORT - In downtrend, no exit signal
        if (latest['ema_bearish_position'] and
            latest['price_below_both_emas'] and
            latest['ema_slopes_aligned_bearish']):
            return {
                'action': 'HOLD SHORT',
                'reason': 'Downtrend intact - maintain short positions',
                'confidence': 'N/A',
                'signal_quality': latest['trend_strength'],
                'details': f"Fast EMA below Slow EMA, price below both EMAs"
            }
        
        # ================================================================
        # DEFAULT: WAIT
        # ================================================================
        
        return {
            'action': 'WAIT',
            'reason': 'No clear signal',
            'confidence': 'N/A',
            'signal_quality': latest['trend_strength'],
            'details': 'Monitor for signal development'
        }

    def interpret_all_signals(self):
        """Interpret signals for all rows in DataFrame"""
        logger.info("Interpreting signals for all rows...")
        
        if not self.signals_generated:
            logger.error("Indicators not computed. Call compute_all_indicators() first.")
            return self.df
        
        # Initialize lists for signal columns
        actions = []
        reasons = []
        confidences = []
        qualities = []
        details = []
        
        # Process each row
        for idx in range(len(self.df)):
            signal = self.interpret_signal(index=idx)
            actions.append(signal['action'])
            reasons.append(signal['reason'])
            confidences.append(signal['confidence'])
            qualities.append(signal['signal_quality'])
            details.append(signal['details'])
        
        # Add signal columns to DataFrame
        self.df['signal_action'] = actions
        self.df['signal_reason'] = reasons
        self.df['signal_confidence'] = confidences
        self.df['signal_quality'] = qualities
        self.df['signal_details'] = details
        
        logger.info("Signal interpretation complete.")
        return self.df

    # ========================================================================
    # COMPLETE ANALYSIS PIPELINE
    # ========================================================================

    def run_complete_analysis(self):
        """
        Run complete pipeline: compute indicators and interpret signals.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with all indicators and signals
        """
        logger.info("="*70)
        logger.info(f"Starting complete analysis with {self.fast_period}/{self.slow_period} EMA strategy...")
        
        # Step 1: Compute all indicators
        self.compute_all_indicators()
        
        # Step 2: Interpret signals
        self.interpret_all_signals()
        
        logger.info("Complete analysis finished.")
        logger.info(f"Total rows processed: {len(self.df)}")
        logger.info(f"Signals generated: {self.df['signal_action'].value_counts().to_dict()}")
        
        return self.df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Load your OHLC data
    # df = pd.read_csv('your_data.csv')
    
    # Initialize scanner with default 8/21 EMA
    scanner_8_21 = EMA_CrossOver_Scanner(df, fast_period=8, slow_period=21)
    results_8_21 = scanner_8_21.run_complete_analysis()
    
    # Or use 13/50 EMA
    scanner_13_50 = EMA_CrossOver_Scanner(df, fast_period=13, slow_period=50)
    results_13_50 = scanner_13_50.run_complete_analysis()
    
    # Get latest signal
    latest_signal = scanner_8_21.interpret_signal(index=-1)
    print(f"\nLatest Signal: {latest_signal['action']}")
    print(f"Reason: {latest_signal['reason']}")
    print(f"Confidence: {latest_signal['confidence']}")
    print(f"Details: {latest_signal['details']}")