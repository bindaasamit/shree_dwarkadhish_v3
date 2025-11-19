#------------------------------------------------------------------------------
###            Filtered EMA Crossovers Generator - Simplified Version
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
#                        Filtered EMA Crossovers Generator Class
#---------------------------------------------------------------------------------------------

class Basic_EMA_Scanner:
    """
    Simplified class that generates only filtered EMA golden/death crossovers.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLC DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
        """
        logger.info("Initializing Filtered_EMACrossovers...")
        self.df = df.copy().reset_index(drop=True)
        # Enable basic momentum indicators for signal confirmation

    # ========================================================================
    # CORE EMA CALCULATIONS (for crossovers)
    # ========================================================================

    def compute_ema_with_filters(self):
        """Compute basic EMAs"""
        logger.info("Computing EMA crossovers...")

        close = self.df['closing_price']

        # Standard EMAs
        self.df['ema5'] = close.ewm(span=5, adjust=False).mean()
        self.df['ema13'] = close.ewm(span=13, adjust=False).mean()
        self.df['ema26'] = close.ewm(span=26, adjust=False).mean()

        ema5 = self.df['ema5']
        ema13 = self.df['ema13']
        ema26 = self.df['ema26']

        ema5_prev = ema5.shift(1)
        ema13_prev = ema13.shift(1)

        # Ribbon analysis (needed for market condition filters)
        self.df['ema_ribbon_width'] = abs(ema5 - ema26)

        # Slope analysis (needed for momentum filters) - with safety checks
        ema5_shifted = ema5.shift(5)
        ema13_shifted = ema13.shift(5)
        ema26_shifted = ema26.shift(5)
        
        # Avoid division by zero and handle NaN
        self.df['ema_slope_5'] = np.where(
            ema5_shifted.notna() & (ema5_shifted != 0),
            (ema5 - ema5_shifted) / ema5_shifted * 100,
            0  # Default to 0 for invalid calculations
        )
        self.df['ema_slope_13'] = np.where(
            ema13_shifted.notna() & (ema13_shifted != 0),
            (ema13 - ema13_shifted) / ema13_shifted * 100,
            0
        )
        self.df['ema_slope_26'] = np.where(
            ema26_shifted.notna() & (ema26_shifted != 0),
            (ema26 - ema26_shifted) / ema26_shifted * 100,
            0
        )
        
        self.df['ema_avg_slope'] = (
            abs(self.df['ema_slope_5']) +
            abs(self.df['ema_slope_13']) +
            abs(self.df['ema_slope_26'])
        ) / 3
                # Add this after calculating ema_slope_alignment
        
        # Slope alignment (for trend strength calculation)
        self.df['ema_slope_alignment'] = (
            ((self.df['ema_slope_5'] > 0).astype(int) +
             (self.df['ema_slope_13'] > 0).astype(int) +
             (self.df['ema_slope_26'] > 0).astype(int)) / 3
        ) - (
            ((self.df['ema_slope_5'] < 0).astype(int) +
             (self.df['ema_slope_13'] < 0).astype(int) +
             (self.df['ema_slope_26'] < 0).astype(int)) / 3
        )

        # Alignment score for sideways detection
        self.df['ema_alignment_score'] = abs(self.df['ema_slope_alignment'])

        # Market conditions (needed for filtering) - with safety checks
        ribbon_pct = (self.df['ema_ribbon_width'] / close) * 100
        self.df['ema_sideways_market'] = (
            (ribbon_pct.fillna(100) < 0.5) &  # High default for NaN
            (self.df['ema_avg_slope'].fillna(100) < 0.3) &  # High default for NaN
            (self.df['ema_alignment_score'].fillna(1) < 0.67)  # High default for NaN
        )

        # Trending market (opposite of sideways)
        self.df['ema_trending_market'] = ~self.df['ema_sideways_market']

        crossover_count = (
            ((ema5 > ema13) != (ema5_prev > ema13_prev)).astype(int)
        ).rolling(10).sum()

        self.df['ema_whipsaw_risk'] = (
            ((self.df['ema_ribbon_width'] / close) * 100 < 0.8) &
            (crossover_count > 3)
        )

        # Base crossovers
        self.df['ema_golden_cross'] = (
            (ema5_prev <= ema13_prev) & (ema5 > ema13)
        )
        self.df['ema_death_cross'] = (
            (ema5_prev >= ema13_prev) & (ema5 < ema13)
        )

        # EMA alignments for signal interpretation
        self.df['ema_bullish_alignment'] = (ema5 > ema13) & (ema13 > ema26)
        self.df['ema_bearish_alignment'] = (ema5 < ema13) & (ema13 < ema26)

        logger.info("EMA crossovers computed.")

    # ========================================================================
    # FILTERED CROSSOVERS
    # ========================================================================

    def compute_filtered_crossovers_basic(self):
        """Generate filtered EMA crossovers using basic market filters (sideways + whipsaw only)"""
        logger.info("Computing filtered crossovers (basic)...")

        # Safety check: ensure required columns exist
        required_cols = ['ema_golden_cross', 'ema_death_cross', 'ema_sideways_market', 'ema_whipsaw_risk']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logger.error(f"Missing required columns for filtered crossovers: {missing_cols}")
            raise ValueError(f"Cannot compute filtered crossovers: missing columns {missing_cols}")

        # Base condition
        golden_cross_base = self.df['ema_golden_cross']
        death_cross_base = self.df['ema_death_cross']

        # Build filter conditions (simplified version)
        filter_conditions = (
            (~self.df['ema_sideways_market']) &
            (~self.df['ema_whipsaw_risk'])
        )

        # Generate filtered signals
        self.df['ema_golden_cross_filtered'] = golden_cross_base & filter_conditions
        self.df['ema_death_cross_filtered'] = death_cross_base & filter_conditions


        #Update - Amit
        self.df['ema_slopes_aligned_buy'] = (
            self.df['ema_golden_cross_filtered'] & (
            ((self.df['ema_slope_5'] > 0.1).astype(int) +
             (self.df['ema_slope_13'] > 0.1).astype(int) + 
             (self.df['ema_slope_26'] > 0.1).astype(int)) >= 2
        ))
        
        self.df['ema_slopes_aligned_sell'] = (
            self.df['ema_death_cross_filtered'] & (
            ((self.df['ema_slope_5'] < 0.1).astype(int) +
             (self.df['ema_slope_13'] < 0.1).astype(int) +
             (self.df['ema_slope_26'] < 0.1).astype(int)) >= 2
        ))
        
        logger.info("Filtered crossovers (basic) computed.")

    """
    def compute_filtered_crossovers_full(self):
        logger.info("Computing filtered crossovers...")
        
        # Base condition
        golden_cross_base = self.df['ema_golden_cross']
        death_cross_base = self.df['ema_death_cross']
        
        # Build filter conditions
        filter_conditions = (
            (~self.df['ema_sideways_market']) &
            (~self.df['ema_whipsaw_risk'])
        )
        
        # Add momentum filters
        if self.use_momentum:
            filter_conditions = filter_conditions & (
                (~self.df.get('rsi_overbought', False)) &
                (~self.df.get('rsi_oversold', False))
            )
        
        # Add volume filters
        if self.use_volume:
            filter_conditions = filter_conditions & (
                (self.df.get('volume_ratio', 1) > 0.8)
            )
        
        # Add volatility filters
        if self.use_volatility:
            filter_conditions = filter_conditions & (
                (self.df.get('atr_pct', 2) < 5)
            )
        
        # Add trend strength filters
        if self.use_support_resistance:
            filter_conditions = filter_conditions & (
                (self.df.get('adx', 20) > 20)
            )
        
        # Generate filtered signals
        self.df['ema_golden_cross_filtered'] = golden_cross_base & filter_conditions
        self.df['ema_death_cross_filtered'] = death_cross_base & filter_conditions
    """
        
    # ========================================================================
    # COMPOSITE CALCULATIONS
    # ========================================================================
    
    def compute_ema_trend_strength(self):
        """Compute comprehensive EMA trend strength for early signal detection"""
        logger.info("Computing EMA trend strength...")
        
        # Component 1: EMA Ribbon Width (20% weight) - Tighter ribbon = stronger trend
        ribbon_score = 1 - (self.df['ema_ribbon_width'] / self.df['closing_price'])
        ribbon_score = ribbon_score.clip(0, 1)  # Bound between 0-1
        
        # Component 2: Slope Alignment (25% weight) - How many EMAs are sloping in same direction
        slope_alignment = (
            ((self.df['ema_slope_5'] > 0).astype(int) +
             (self.df['ema_slope_13'] > 0).astype(int) +
             (self.df['ema_slope_26'] > 0).astype(int)) / 3
        )
        
        # Component 3: ADX Trend Strength (20% weight) - External trend confirmation
        adx_score = (self.df['adx'] / 50).clip(0, 1)  # Normalize ADX to 0-1 scale
        
        # Component 4: Momentum Consistency (15% weight) - Recent slope acceleration
        momentum_consistency = (
            (self.df['ema_slope_5'] > self.df['ema_slope_5'].shift(3)) &
            (self.df['ema_slope_13'] > self.df['ema_slope_13'].shift(3))
        ).astype(int)
        
        # Component 5: Trend Duration (10% weight) - How long trend has been building
        # Look back 10 days for consistent direction
        trend_duration = (
            (self.df['ema_slope_5'].rolling(10).apply(lambda x: (x > 0).sum() / len(x))) *
            (self.df['ema_slope_13'].rolling(10).apply(lambda x: (x > 0).sum() / len(x)))
        ).fillna(0)
        
        # Component 6: Volume Confirmation (10% weight) - Volume supporting the move
        volume_score = self.df['volume_ratio'].clip(0.5, 2) / 2  # Normalize 0.5-2 range to 0-1
        
        # Weighted combination for comprehensive trend strength
        self.df['ema_trend_strength'] = (
            ribbon_score * 0.20 +
            slope_alignment * 0.25 +
            adx_score * 0.20 +
            momentum_consistency * 0.15 +
            trend_duration * 0.10 +
            volume_score * 0.10
        ).clip(0, 1)  # Ensure 0-1 range
        
        # Early trend detection - more sensitive version for pre-cross signals
        self.df['ema_early_trend_strength'] = (
            slope_alignment * 0.30 +  # More weight on slope alignment
            momentum_consistency * 0.25 +  # Acceleration is key for early detection
            volume_score * 0.20 +  # Volume confirmation
            trend_duration * 0.15 +  # Building momentum
            adx_score * 0.10  # Less weight on ADX for early signals
        ).clip(0, 1)
        
        logger.info("EMA trend strength computed.")

    def compute_rsi(self, period=14):
        """Calculate RSI"""
        
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
        self.df['rsi_bullish'] = self.df['rsi'] > 50
        self.df['rsi_bearish'] = self.df['rsi'] < 50
        
        # RSI momentum
        self.df['rsi_rising'] = self.df['rsi'] > self.df['rsi'].shift(1)

    def compute_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD"""
         
        logger.info("Computing MACD...")
        
        # Calculate EMAs for MACD
        ema_fast = self.df['closing_price'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.df['closing_price'].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        self.df['macd_line'] = ema_fast - ema_slow
        
        # Signal line
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        self.df['macd_hist'] = self.df['macd_line'] - self.df['macd_signal']
        
        # MACD signals
        self.df['macd_bullish'] = self.df['macd_line'] > self.df['macd_signal']
        self.df['macd_bearish'] = self.df['macd_line'] < self.df['macd_signal']
        
        # Histogram direction
        self.df['macd_hist_positive'] = self.df['macd_hist'] > 0
        self.df['macd_hist_negative'] = self.df['macd_hist'] < 0

    def compute_adx(self, period=14):
        """Calculate ADX for trend strength"""
                
        logger.info("Computing ADX...")
        
        # Calculate +DM and -DM
        high_diff = self.df['high_price'].diff()
        low_diff = -self.df['low_price'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate True Range
        tr = pd.concat([
            self.df['high_price'] - self.df['low_price'],
            abs(self.df['high_price'] - self.df['closing_price'].shift()),
            abs(self.df['low_price'] - self.df['closing_price'].shift())
        ], axis=1).max(axis=1)
        
        atr_adx = tr.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr_adx)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr_adx)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        self.df['adx'] = dx.rolling(window=period).mean()
        
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        
        # Trend strength classification
        self.df['trend_strength_adx'] = np.where(
            self.df['adx'] > 25, 'strong',
            np.where(self.df['adx'] > 20, 'moderate', 'weak')
        )
        
        # Directional movement
        self.df['adx_bullish'] = self.df['plus_di'] > self.df['minus_di']
        self.df['adx_bearish'] = self.df['plus_di'] < self.df['minus_di']

    def compute_atr(self, period=14):
        """Calculate Average True Range"""
            
        logger.info("Computing ATR...")
        
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['closing_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['closing_price'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=period).mean()
        
        # ATR as % of price
        self.df['atr_pct'] = (self.df['atr'] / self.df['closing_price']) * 100
        
        # Volatility classification
        self.df['volatility_regime'] = np.where(
            self.df['atr_pct'] > 3, 'high',
            np.where(self.df['atr_pct'] > 1.5, 'medium', 'low')
        )
        
        # Volatility expansion/contraction
        self.df['atr_expanding'] = (
            self.df['atr'] > self.df['atr'].rolling(5).mean()
        )
        self.df['atr_contracting'] = (
            self.df['atr'] < self.df['atr'].rolling(5).mean()
        )

    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================
    def compute_volume_indicators(self):
        """Comprehensive volume analysis"""
        logger.info("Computing Volume indicators...")
        
        # Average volumes
        self.df['avg_volume_20'] = self.df['total_trading_volume'].rolling(20).mean()
        self.df['avg_volume_50'] = self.df['total_trading_volume'].rolling(50).mean()
        
        # Volume ratio
        self.df['volume_ratio'] = self.df['total_trading_volume'] / self.df['avg_volume_20']
        
        # Volume trend
        self.df['volume_increasing'] = (
            self.df['avg_volume_20'] > self.df['avg_volume_50']
        )
        
        # Price-Volume relationship
        self.df['bullish_volume'] = (
            (self.df['closing_price'] > self.df['closing_price'].shift(1)) &
            (self.df['total_trading_volume'] > self.df['avg_volume_20'])
        )
        self.df['bearish_volume'] = (
            (self.df['closing_price'] < self.df['closing_price'].shift(1)) &
            (self.df['total_trading_volume'] > self.df['avg_volume_20'])
        )
    
    
    def compute_pivot_points(self):
        """Calculate pivot points"""
        
        logger.info("Computing Pivot Points...")
        
        self.df['pivot'] = (
            self.df['high_price'] + self.df['low_price'] + self.df['closing_price']
        ) / 3
        
        self.df['resistance_1'] = 2 * self.df['pivot'] - self.df['low_price']
        self.df['support_1'] = 2 * self.df['pivot'] - self.df['high_price']
        
        # Price position relative to pivot
        self.df['above_pivot'] = self.df['closing_price'] > self.df['pivot']
        self.df['below_pivot'] = self.df['closing_price'] < self.df['pivot']
        
        # Near resistance/support (within 1%)
        self.df['near_resistance'] = abs(
            self.df['closing_price'] - self.df['resistance_1']
        ) / self.df['closing_price'] < 0.01
        
        self.df['near_support'] = abs(
            self.df['closing_price'] - self.df['support_1']
        ) / self.df['closing_price'] < 0.01
    
    
    def compute_all_indicators(self):
       
        logger.info("="*70)
        logger.info("Starting comprehensive indicator calculation...")
        
        
        # Core EMA (always computed)
        self.compute_ema_with_filters()
        
        # Momentum indicators
        self.compute_rsi()
        #self.compute_stochastic()
        self.compute_macd()
        
        # Volume indicators
        self.compute_volume_indicators()
        #self.compute_obv()
        
        # Volatility indicators
        self.compute_atr()
        #self.compute_bollinger_bands()
        
        # Trend strength & support/resistance
        self.compute_adx()
        self.compute_pivot_points()
        
        # Divergence detection
        #if self.use_divergence and self.use_momentum:
        #    self.detect_divergences()
        
        # Composite calculations
        self.compute_ema_trend_strength()
        #self.compute_enhanced_signal_quality()
        self.compute_filtered_crossovers_basic()
        # Binarize boolean-like indicator columns to integer 0/1 for downstream usage
        #self._binarize_indicator_columns()
         
        self.signals_generated = True
        
        logger.info("All indicators computed successfully!")
        
        # Print summary
        #self.print_indicator_summary()
        
        return self.df
    
    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================
    
    def _check_slope_alignment_for_signal(self, latest, signal_type):
        """
        Check slope alignment requiring AT LEAST 2 slopes to be in the same direction.
        Returns (is_aligned, warning_message)
        
        Parameters:
        -----------
        latest : pd.Series
            Latest row data
        signal_type : str
            'BUY' or 'SELL'
        
        Returns:
        -----------
        tuple : (bool, str) - (is_aligned, warning_message)
        """
        slope_5 = latest['ema_slope_5']
        slope_13 = latest['ema_slope_13'] 
        slope_26 = latest['ema_slope_26']
        
        if signal_type == 'BUY':
            # For BUY signals, we want positive slopes
            slopes = [slope_5 > 0, slope_13 > 0, slope_26 > 0]
        else:  # SELL
            # For SELL signals, we want negative slopes
            slopes = [slope_5 < 0, slope_13 < 0, slope_26 < 0]
        
        # Count how many slopes are in the target direction
        aligned_count = sum(slopes)
        
        if aligned_count >= 2:
            # At least 2 slopes are aligned
            warning = "One EMA slope still negative" if aligned_count == 2 else ""
            return True, warning
        else:
            # Not enough slopes aligned
            return False, ""
    
    def interpret_signals(self, index=-1, strategy='full'):    
        """
        Interpret signals with multiple strategy options.
        
        Parameters:
        -----------
            index : int
            Row index to interpret (-1 for latest)
            strategy : str
            'ema_only' - Use only EMA signals
            'ema_momentum' - EMA + momentum indicators
            'ema_volume' - EMA + volume indicators
            'full' - Use all indicators (recommended)
        
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
                'details': 'Run compute_all_indicators() first',
                'signal_type': 'ERROR',
                'signal_warning': '',
                'signal_limiter': ''
            }
        
        latest = self.df.iloc[index]
        
        # ================================================================
        # STEP 1: Market Condition Filters (Always apply)
        # ================================================================
        
        # EMA-based filters
        if latest['ema_sideways_market']:
            return self._wait_signal(latest, 'Sideways market (EMA)', 
                                    f"Ribbon: {latest['ema_ribbon_width']:.2f}")
        
        if latest['ema_whipsaw_risk']:
            return self._wait_signal(latest, 'High whipsaw risk', 
                                    'Too many recent crossovers')
        
        # Momentum-based filters (if enabled)
        if strategy in ['ema_momentum', 'full']:
            if latest.get('rsi', 50) > 85:
                return self._wait_signal(latest, 'Extremely overbought (RSI>85)', 
                                        f"RSI: {latest['rsi']:.1f}")
            
            if latest.get('rsi', 50) < 15:
                return self._wait_signal(latest, 'Extremely oversold (RSI<15)', 
                                        f"RSI: {latest['rsi']:.1f}")
        
        # Volume-based filters (if enabled)
        if strategy in ['ema_volume', 'full']:
            if latest.get('low_volume', False):
                return self._wait_signal(latest, 'Low volume period', 
                                        f"Volume ratio: {latest.get('volume_ratio', 1):.2f}")
        
        # Volatility-based filters (if enabled)
        if strategy == 'full':
            if latest.get('atr_pct', 2) > 6:
                return self._wait_signal(latest, 'Extreme volatility', 
                                        f"ATR: {latest['atr_pct']:.2f}%")
        
        # Divergence warnings (if enabled)
        if strategy == 'full':
            if latest.get('bearish_divergence', False):
                return self._wait_signal(latest, 'Bearish divergence detected', 
                                        'Price-RSI disagreement - potential reversal')
        
        # ================================================================
        # STEP 2: Generate Buy Signals (Strategy-dependent)
        # ================================================================
        
        if strategy == 'ema_only':
            return self._interpret_ema_only(latest)
        
        elif strategy == 'ema_momentum':
            return self._interpret_ema_only(latest)
        
        elif strategy == 'ema_volume':
            return self._interpret_ema_only(latest)
        
        elif strategy == 'full':
            return self._interpret_ema_only(latest)
        
        else:
            return self._interpret_ema_only(latest)

    def _wait_signal(self, row, reason, details):
        """Helper to return WAIT signal"""
        return {
            'action': 'WAIT',
            'reason': reason,
            'confidence': 'N/A',
            'signal_quality': row.get('signal_quality_enhanced', 0),
            'details': details,
            'signal_type': 'WAIT',
            'signal_warning': '',
            'signal_limiter': ''
        }
      
    def _interpret_ema_only(self, latest):
        """Strategy 1: EMA signals only"""

        # Check for golden cross + slope alignment (base requirements for tiered BUY signals)
        slope_alignment_ok, slope_warning = self._check_slope_alignment_for_signal(latest, 'BUY')
        has_base_buy_conditions = (
            latest['ema_golden_cross_filtered'] and
            latest['ema_slopes_aligned_buy']  # At least 2 slopes must be positive
        )

        # =====================================================================
        # STANDARD BUY SIGNALS
        # =====================================================================

        # STRONG BUY - High quality trend strength
        if has_base_buy_conditions and latest['ema_trend_strength'] > 0.5:
            base_warnings = self._build_signal_warnings(latest, 'BUY')
            full_warnings = base_warnings + ("; " + slope_warning if slope_warning else "")
            return {
                'action': 'STRONG BUY',
                'reason': 'EMA golden cross (filtered) + aligned positive slopes + high trend strength',
                'confidence': 'HIGH',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Strong Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}",
                'signal_type': 'BUY',
                'signal_warning': full_warnings,
                'signal_limiter': ''
            }
        
        # MODERATE BUY - Medium quality trend strength with golden cross
        elif has_base_buy_conditions and latest['ema_trend_strength'] > 0.4:
            base_warnings = self._build_signal_warnings(latest, 'BUY')
            full_warnings = base_warnings + ("; " + slope_warning if slope_warning else "")
            return {
                'action': 'MODERATE BUY',
                'reason': 'EMA golden cross (filtered) + aligned positive slopes + moderate trend strength',
                'confidence': 'MEDIUM',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}",
                'signal_type': 'BUY',
                'signal_warning': full_warnings,
                'signal_limiter': 'Trend strength below STRONG BUY threshold'
            }
        
        # WEAK BUY - Lower quality trend strength with golden cross
        elif has_base_buy_conditions and latest['ema_trend_strength'] > 0.3:
            base_warnings = self._build_signal_warnings(latest, 'BUY')
            full_warnings = base_warnings + ("; " + slope_warning if slope_warning else "")
            return {
                'action': 'WEAK BUY',
                'reason': 'EMA golden cross (filtered) + aligned positive slopes + weak trend strength',
                'confidence': 'LOW',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}",
                'signal_type': 'BUY',
                'signal_warning': full_warnings + "; Weak trend strength",
                'signal_limiter': 'Trend strength indicates weak momentum'
            }
        
        # MODERATE BUY - Fallback for bullish alignment without golden cross
        if (latest['ema_bullish_alignment'] and 
            latest['ema_trending_market'] and 
            latest.get('ema_slope_alignment', 0) == 1):
            # Check what tier this signal would qualify for
            quality = latest.get('ema_trend_strength', 0.5)
            if quality > 0.5:
                signal_limiter = "Meets STRONG BUY criteria - use higher tier signal"
            elif quality > 0.4:
                signal_limiter = "Meets MODERATE BUY criteria - trend strength adequate"
            elif quality > 0.3:
                signal_limiter = "Meets WEAK BUY criteria - monitor closely"
            else:
                signal_limiter = f"Trend strength too low ({quality:.2f}) for BUY signals"
            
            return {
                'action': 'MODERATE BUY',
                'reason': 'EMA bullish alignment + trending market + aligned slopes (no golden cross)',
                'confidence': 'MEDIUM',
                'signal_quality': quality,
                'details': 'All EMAs aligned (5>13>26) with consistent slope direction',
                'signal_type': 'BUY',
                'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                'signal_limiter': signal_limiter
            }
        
        # WAIT - Golden cross but conflicting slopes
        if latest['ema_golden_cross_filtered'] and latest.get('ema_slope_alignment', 0) == 0:
            # Analyze what prevents this from being a STRONG BUY
            missing_conditions = []
            
            # Check slope alignment (already know this is missing)
            slope_alignment_ok, _ = self._check_slope_alignment_for_signal(latest, 'BUY')
            if not slope_alignment_ok:
                slope_count = sum([latest['ema_slope_5'] > 0, latest['ema_slope_13'] > 0, latest['ema_slope_26'] > 0])
                missing_conditions.append(f"Only {slope_count}/3 EMAs have positive slopes")
            
            # Check trend strength
            trend_strength = latest.get('ema_trend_strength', 0.5)
            if trend_strength <= 0.5:
                if trend_strength <= 0.4:
                    missing_conditions.append(f"Trend strength too low ({trend_strength:.2f} ≤ 0.4)")
                else:
                    missing_conditions.append(f"Trend strength below STRONG BUY threshold ({trend_strength:.2f} ≤ 0.5)")
            
            signal_limiter = "; ".join(missing_conditions) if missing_conditions else "Has golden cross but slopes not aligned for STRONG BUY"
            
            return {
                'action': 'WAIT',
                'reason': 'Golden cross but conflicting slope directions',
                'confidence': 'N/A',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Slopes not aligned: {latest['ema_slope_5']:.2f}, {latest['ema_slope_13']:.2f}, {latest['ema_slope_26']:.2f}",
                'signal_type': 'WAIT',
                'signal_warning': '',
                'signal_limiter': signal_limiter
            }
        
        # =====================================================================
        # STANDARD SELL SIGNALS
        # =====================================================================

        # Check for death cross + slope alignment (base requirements for tiered SELL signals)
        sell_slope_alignment_ok, sell_slope_warning = self._check_slope_alignment_for_signal(latest, 'SELL')
        has_base_sell_conditions = (
            latest['ema_death_cross_filtered'] and
            latest['ema_slopes_aligned_sell']  # At least 2 slopes must be negative
        )

        # STRONG SELL - High quality bearish trend strength
        if has_base_sell_conditions and latest['ema_trend_strength'] > 0.5:
            base_warnings = self._build_signal_warnings(latest, 'SELL')
            full_warnings = base_warnings + ("; " + sell_slope_warning if sell_slope_warning else "")
            return {
                'action': 'STRONG SELL',
                'reason': 'EMA death cross (filtered) + aligned negative slopes + high trend strength + bearish momentum',
                'confidence': 'HIGH',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Strong Bearish Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}, RSI: {latest.get('rsi', 'N/A')}, MACD: {latest.get('macd_hist', 'N/A'):.4f}",
                'signal_type': 'SELL',
                'signal_warning': full_warnings,
                'signal_limiter': ''
            }

        # MODERATE SELL - Medium quality bearish trend strength with death cross
        elif has_base_sell_conditions and latest['ema_trend_strength'] > 0.4:
            base_warnings = self._build_signal_warnings(latest, 'SELL')
            full_warnings = base_warnings + ("; " + sell_slope_warning if sell_slope_warning else "")
            return {
                'action': 'MODERATE SELL',
                'reason': 'EMA death cross (filtered) + aligned negative slopes + moderate trend strength + bearish momentum',
                'confidence': 'MEDIUM',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Bearish Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}, RSI: {latest.get('rsi', 'N/A')}, MACD: {latest.get('macd_hist', 'N/A'):.4f}",
                'signal_type': 'SELL',
                'signal_warning': full_warnings,
                'signal_limiter': 'Trend strength below STRONG SELL threshold'
            }

        # WEAK SELL - Lower quality bearish trend strength with death cross
        elif has_base_sell_conditions and latest['ema_trend_strength'] > 0.3:
            base_warnings = self._build_signal_warnings(latest, 'SELL')
            full_warnings = base_warnings + ("; " + sell_slope_warning if sell_slope_warning else "")
            return {
                'action': 'WEAK SELL',
                'reason': 'EMA death cross (filtered) + aligned negative slopes + weak trend strength + bearish momentum',
                'confidence': 'LOW',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Ribbon: {latest['ema_ribbon_width']:.2f}, Bearish Slopes: {latest['ema_slope_5']:.2f}%/{latest['ema_slope_13']:.2f}%/{latest['ema_slope_26']:.2f}%, Trend Strength: {latest['ema_trend_strength']:.2f}, RSI: {latest.get('rsi', 'N/A')}, MACD: {latest.get('macd_hist', 'N/A'):.4f}",
                'signal_type': 'SELL',
                'signal_warning': full_warnings + "; Weak trend strength",
                'signal_limiter': 'Trend strength indicates weak momentum'
            }

        # MODERATE SELL - Fallback for bearish alignment without death cross
        if (latest['ema_bearish_alignment'] and
            latest['ema_trending_market'] and
            latest.get('ema_slope_alignment', 0) == -1):  # Fully bearish alignment
            # Check what tier this signal would qualify for
            quality = latest.get('ema_trend_strength', 0.5)
            if quality > 0.5:
                signal_limiter = "Meets STRONG SELL criteria - use higher tier signal"
            elif quality > 0.4:
                signal_limiter = "Meets MODERATE SELL criteria - trend strength adequate"
            elif quality > 0.3:
                signal_limiter = "Meets WEAK SELL criteria - monitor closely"
            else:
                signal_limiter = f"Trend strength too low ({quality:.2f}) for SELL signals"

            return {
                'action': 'MODERATE SELL',
                'reason': 'EMA bearish alignment + trending market + aligned slopes (no death cross)',
                'confidence': 'MEDIUM',
                'signal_quality': quality,
                'details': 'All EMAs aligned (5<13<26) with consistent slope direction',
                'signal_type': 'SELL',
                'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                'signal_limiter': signal_limiter
            }

        # WAIT - Death cross but conflicting slopes
        if latest['ema_death_cross_filtered'] and latest.get('ema_slope_alignment', 0) == 0:
            # Analyze what prevents this from being a STRONG SELL
            missing_conditions = []

            # Check slope alignment (already know this is missing)
            slope_alignment_ok, _ = self._check_slope_alignment_for_signal(latest, 'SELL')
            if not slope_alignment_ok:
                slope_count = sum([latest['ema_slope_5'] < 0, latest['ema_slope_13'] < 0, latest['ema_slope_26'] < 0])
                missing_conditions.append(f"Only {slope_count}/3 EMAs have negative slopes")

            # Check trend strength
            trend_strength = latest.get('ema_trend_strength', 0.5)
            if trend_strength <= 0.5:
                if trend_strength <= 0.4:
                    missing_conditions.append(f"Trend strength too low ({trend_strength:.2f} ≤ 0.4)")
                else:
                    missing_conditions.append(f"Trend strength below STRONG SELL threshold ({trend_strength:.2f} ≤ 0.5)")

            signal_limiter = "; ".join(missing_conditions) if missing_conditions else "Has death cross but slopes not aligned for STRONG SELL"

            return {
                'action': 'WAIT',
                'reason': 'Death cross but conflicting slope directions',
                'confidence': 'N/A',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f"Slopes not aligned: {latest['ema_slope_5']:.2f}, {latest['ema_slope_13']:.2f}, {latest['ema_slope_26']:.2f}",
                'signal_type': 'WAIT',
                'signal_warning': '',
                'signal_limiter': signal_limiter
            }
        
        # HOLD - Bullish structure intact (maintain long positions)
        if latest['ema_bullish_alignment'] and latest.get('ema_slope_alignment', 0) > 0:
            return {
                'action': 'HOLD',
                'reason': 'Bullish structure intact - maintain long positions',
                'confidence': 'N/A',
                'signal_quality': latest.get('ema_trend_strength', 0.5),
                'details': f'Bullish alignment maintained: {latest["ema_slope_5"]:.2f}, {latest["ema_slope_13"]:.2f}, {latest["ema_slope_26"]:.2f}',
                'signal_type': 'HOLD',
                'signal_warning': '',
                'signal_limiter': ''
            }
        
        return {
            'action': 'WAIT',
            'reason': 'No clear EMA signal',
            'confidence': 'N/A',
            'signal_quality': latest.get('ema_trend_strength', 0.5),
            'details': 'Monitor for signal development',
            'signal_type': 'WAIT',
            'signal_warning': '',
            'signal_limiter': self._analyze_missing_buy_conditions(latest)
        }
    
    def interpret_all_signals(self, strategy):
        """
        Interpret signals for all rows.
        
        Parameters:
        -----------
        strategy : str
            Which strategy to use ('ema_only', 'ema_momentum', 'ema_volume', 'full')
        """
        if not self.signals_generated:
            logger.error("Indicators not computed. Call compute_all_indicators() first.")
            return self.df
        
        #Enhancing Scanner Business Logic based on Feedback from BackTester
        logger.info(f"\nNo of Records Before Filtering Bad Signals ({len(self.df)}):")
        
        logger.info(f"Interpreting all signals with strategy: {strategy}...")
        
        # Accumulate results in lists to avoid repeated single-column inserts (which fragment the DataFrame)
        n = len(self.df)
        actions = []
        reasons = []
        confidences = []
        types = []
        warnings = []
        limiters = []

        for idx in range(n):
            result = self.interpret_signals(index=idx, strategy=strategy)
            actions.append(result.get('action', ''))
            reasons.append(result.get('reason', ''))
            confidences.append(result.get('confidence', ''))
            types.append(result.get('signal_type', ''))
            warnings.append(result.get('signal_warning', ''))
            limiters.append(result.get('signal_limiter', ''))

        # Create a single DataFrame with all new signal columns and attach it once
        new_cols_df = pd.DataFrame({
            'signal_action': actions,
            'signal_reason': reasons,
            'signal_confidence': confidences,
            'signal_type': types,
            'signal_warning': warnings,
            'signal_limiter': limiters
        }, index=self.df.index)

        # Drop any existing signal columns (if present) and concat the new columns in a single operation
        cols_to_drop = [c for c in new_cols_df.columns if c in self.df.columns]
        if cols_to_drop:
            base_df = self.df.drop(columns=cols_to_drop, errors='ignore')
        else:
            base_df = self.df
        self.df = pd.concat([base_df, new_cols_df], axis=1)
        
        logger.info("Signal interpretation complete.")
        #self._print_signal_summary(strategy)
        
        return self.df
    
    def interpret_all_signals_with_early_detection(self, strategy='full', use_early_signals=True):
        """
        Interpret signals for all rows with optional early signal detection.

        Parameters:
        -----------
        strategy : str
            Signal interpretation strategy
        use_early_signals : bool
            Whether to check for early signals

        Returns:
        -----------
        pd.DataFrame : DataFrame with signal columns added
        """
        logger.info("Interpreting signals for all rows with early detection...")

        # Initialize lists to store signal data
        actions = []
        reasons = []
        confidences = []
        qualities = []
        details_list = []
        types = []
        warnings = []
        limiters = []

        # Process each row
        for idx in range(len(self.df)):
            try:
                # Get signal for this row
                signal = self.interpret_signals_with_early_detection(
                    index=idx,
                    strategy=strategy,
                    use_early_signals=use_early_signals
                )

                actions.append(signal['action'])
                reasons.append(signal['reason'])
                confidences.append(signal['confidence'])
                qualities.append(signal.get('signal_quality', 0.0))
                details_list.append(signal['details'])
                types.append(signal['signal_type'])
                warnings.append(signal['signal_warning'])
                limiters.append(signal['signal_limiter'])

            except Exception as e:
                logger.error(f"Error interpreting signal for row {idx}: {e}")
                # Add default WAIT signal for errors
                actions.append('WAIT')
                reasons.append('Error in signal interpretation')
                confidences.append('N/A')
                qualities.append(0.0)
                details_list.append('Signal interpretation failed')
                types.append('ERROR')
                warnings.append('')
                limiters.append('Technical error')

        # Create new columns DataFrame
        new_cols_df = pd.DataFrame({
            'signal_action': actions,
            'signal_reason': reasons,
            'signal_confidence': confidences,
            'signal_quality': qualities,
            'signal_details': details_list,
            'signal_type': types,
            'signal_warning': warnings,
            'signal_limiter': limiters
        }, index=self.df.index)

        # Drop any existing signal columns (if present) and concat the new columns in a single operation
        cols_to_drop = [c for c in new_cols_df.columns if c in self.df.columns]
        if cols_to_drop:
            base_df = self.df.drop(columns=cols_to_drop, errors='ignore')
        else:
            base_df = self.df
        self.df = pd.concat([base_df, new_cols_df], axis=1)

        logger.info("Signal interpretation with early detection complete.")
        return self.df

    
    
    def _analyze_missing_buy_conditions(self, latest):
        """
        Analyze what conditions are missing that prevent a STRONG BUY signal.
        
        Parameters:
        -----------
        latest : pd.Series
            Latest row data
        
        Returns:
        -----------
        str : Explanation of missing conditions
        """
        missing_conditions = []
        
        # Check golden cross
        if not latest.get('ema_golden_cross_filtered', False):
            missing_conditions.append("No golden cross")
        
        # Check slope alignment
        slope_alignment_ok, _ = self._check_slope_alignment_for_signal(latest, 'BUY')
        if not slope_alignment_ok:
            slope_count = sum([latest.get('ema_slope_5', 0) > 0, 
                              latest.get('ema_slope_13', 0) > 0, 
                              latest.get('ema_slope_26', 0) > 0])
            if slope_count == 0:
                missing_conditions.append("All EMA slopes negative")
            else:
                missing_conditions.append(f"Only {slope_count}/3 EMAs have positive slopes")
        
        # Check trend strength
        trend_strength = latest.get('ema_trend_strength', 0.5)
        if trend_strength <= 0.5:
            if trend_strength <= 0.4:
                if trend_strength <= 0.3:
                    missing_conditions.append(f"Trend strength critically low ({trend_strength:.2f} ≤ 0.3)")
                else:
                    missing_conditions.append(f"Trend strength too low for any BUY signal ({trend_strength:.2f} ≤ 0.4)")
            else:
                missing_conditions.append(f"Trend strength below STRONG BUY threshold ({trend_strength:.2f} ≤ 0.5)")
        
        # If no conditions missing but still WAIT, it might be due to market filters
        if not missing_conditions:
            if latest.get('ema_sideways_market', False):
                missing_conditions.append("Sideways market conditions")
            elif latest.get('ema_whipsaw_risk', False):
                missing_conditions.append("High whipsaw risk")
            else:
                missing_conditions.append("Market conditions prevent BUY signals")
        
        return "; ".join(missing_conditions) if missing_conditions else "All STRONG BUY conditions potentially met but market filters active"
    
    def _build_signal_warnings(self, latest, signal_type):
        """
        Build warning string for conflicting indicators.
        
        Parameters:
        -----------
        latest : pd.Series
            Latest row data
        signal_type : str
            'BUY' or 'SELL'
        
        Returns:
        -----------
        str : Warning message
        """
        warnings = []
        
        # Common checks for both BUY and SELL
        if latest.get('adx', 25) < 20:
            warnings.append("Weak trend (ADX < 20)")
        
        if latest.get('volume_ratio', 1) < 0.8:
            warnings.append("Low volume")
        
        if latest.get('atr_pct', 2) > 5:
            warnings.append("High volatility")
        
        if signal_type == 'BUY':
            # For BUY signals, warn about bearish indicators
            if latest.get('rsi', 50) > 70:
                warnings.append("RSI overbought")
            
            if latest.get('plus_di', 25) < latest.get('minus_di', 25):
                warnings.append("ADX bearish")
                
        elif signal_type == 'SELL':
            # For SELL signals, warn about bullish indicators
            if latest.get('rsi', 50) < 30:
                warnings.append("RSI oversold")
            
            if latest.get('plus_di', 25) > latest.get('minus_di', 25):
                warnings.append("ADX bullish")
        
        return "; ".join(warnings) if warnings else ""


    # ============================================================================
    # RUN THE ENTIRE WORKFLOW
    # ============================================================================

    
    def run_complete_analysis(self, strategy):
        """
        Complete pipeline: compute all indicators and interpret signals.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with all indicators and signals
        """
        logger.info("Starting complete enhanced analysis...")
        
        # Step 1: Compute all indicators
        self.compute_all_indicators()
        
        # Step 2: Interpret signals
        self.interpret_all_signals(strategy=strategy)

        #logger.info(f"No of Records returned for {self.df.tckr_symbol}: {self.df.shape}")
        logger.info("Complete enhanced analysis finished.")
        return self.df