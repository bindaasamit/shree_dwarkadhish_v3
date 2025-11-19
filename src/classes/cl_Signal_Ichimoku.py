#------------------------------------------------------------------------------
###            Common Libraries that are required across notebooks
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import date, timedelta
import concurrent.futures
from loguru import logger
import warnings
import json
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import Modules
from config import cfg_nifty as cfg_nifty
from config import cfg_vars as cfg_vars



#---------------------------------------------------------------------------------------------
#                        Class Definition for SignalScanner
#---------------------------------------------------------------------------------------------

class Basic_Ichimoku_Scanner:
    def __init__(self, df: pd.DataFrame): 
        """
        df: DataFrame containing columns:
            'tckr_symbol','trade_dt','open_price','high_price','low_price','closing_price',
            'total_trading_volume', 'total_transfer_value' (and optionally others)
        """
        logger.info("Initializing Signal_Ichimoku_Scanner.")
        self.df = df.copy().reset_index(drop=True)
        self.signals_generated = False
        

    def compute_ichimoku_with_filters(self):
        """
        Enhanced Ichimoku calculation with sideways market detection and 
        false signal filtering for improved accuracy.
        """
        logger.info("1. Enhanced Ichimoku Start.")
        
        # ========================================================================
        # PART 1: COMPUTE STANDARD ICHIMOKU COMPONENTS
        # ========================================================================
        
        high = self.df['high_price']
        low = self.df['low_price']
        close = self.df['closing_price']
        
        # Calculate rolling highs and lows
        high_9, low_9 = high.rolling(9).max(), low.rolling(9).min()
        high_26, low_26 = high.rolling(26).max(), low.rolling(26).min()
        high_52, low_52 = high.rolling(52).max(), low.rolling(52).min()
        
        # Standard Ichimoku lines
        self.df['tenkan'] = (high_9 + low_9) / 2
        self.df['kijun'] = (high_26 + low_26) / 2
        self.df['senkou_a'] = ((self.df['tenkan'] + self.df['kijun']) / 2).shift(26)
        self.df['senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        self.df['chikou_span'] = close.shift(-26)
        self.df['prev_closing_price'] = close.shift(1)
        
        # Cloud boundaries
        senkou_a = self.df['senkou_a']
        senkou_b = self.df['senkou_b']
        self.df['kumo_top'] = np.maximum(senkou_a, senkou_b)
        self.df['kumo_bottom'] = np.minimum(senkou_a, senkou_b)
        
        # ========================================================================
        # PART 2: CALCULATE FILTERING METRICS
        # ========================================================================
        
        # --- Cloud Thickness Analysis ---
        self.df['kumo_thickness'] = abs(senkou_a - senkou_b)
        self.df['kumo_thickness_pct'] = (self.df['kumo_thickness'] / close) * 100
        
        # --- Price-Cloud Distance Analysis ---
        self.df['price_kumo_distance'] = np.where(
            close > self.df['kumo_top'],
            close - self.df['kumo_top'],  # Above cloud (bullish)
            np.where(
                close < self.df['kumo_bottom'],
                self.df['kumo_bottom'] - close,  # Below cloud (bearish)
                0  # Inside cloud (neutral/sideways)
            )
        )
        self.df['price_kumo_distance_pct'] = (self.df['price_kumo_distance'] / close) * 100

        # --- Tenkan-Kijun Separation Analysis ---
        self.df['tk_separation'] = abs(self.df['tenkan'] - self.df['kijun'])
        self.df['tk_separation_pct'] = (self.df['tk_separation'] / close) * 100
        
        # --- Chikou Span Clarity ---
        chikou = self.df['chikou_span']
        price_26_ago = close.shift(26)
        self.df['chikou_clear'] = (
            ((chikou > price_26_ago * 1.02) |  # Clearly above by 2%
            (chikou < price_26_ago * 0.98))   # Clearly below by 2%
        )
        
        # --- Price Volatility (for choppiness detection) ---
        self.df['price_range_pct'] = ((high - low) / close) * 100
        self.df['avg_range_pct'] = self.df['price_range_pct'].rolling(10).mean()
        
        # ========================================================================
        # PART 3: DETECT MARKET CONDITIONS (Sideways/Choppy/Weak)
        # ========================================================================
        
        #########  SIDEWAYS MARKET DETECTION
        self.df['sideways_market'] = (
            (self.df['kumo_thickness_pct'] < 1.0) &      # Cloud less than 1% thick
            (self.df['tk_separation_pct'] < 0.5) &       # TK very close together
            (self.df['price_kumo_distance_pct'] < 2.0)   # Price close to cloud
        )
        
        #########  WEAK TREND DETECTION
        tenkan = self.df['tenkan']
        kijun = self.df['kijun']
        tenkan_prev = tenkan.shift(1)
        kijun_prev = kijun.shift(1)
        
        self.df['weak_trend'] = (
            (self.df['tk_separation_pct'] < 1.0) |       # Lines too close
            ((tenkan > kijun) != (tenkan_prev > kijun_prev))  # Direction flip-flopping
        )
        
        #########  CHOPPY MARKET DETECTION
        self.df['choppy_market'] = (
            (self.df['price_range_pct'] > self.df['avg_range_pct'] * 1.5) &  # High volatility
            (~self.df['chikou_clear']) &                  # Chikou tangled
            (self.df['price_kumo_distance_pct'] < 3.0)    # Price not decisively trending
        )
        
        #########  FALSE BREAKOUT RISK DETECTION
        self.df['false_breakout_risk'] = (
            (self.df['price_range_pct'] < self.df['avg_range_pct'] * 0.7) |  # Small candles
            (self.df['kumo_thickness_pct'] < 0.5)         # Very thin cloud (weak level)
        )
        
        # OVERALL TREND QUALITY SCORE (0 = Poor, 1 = Excellent)
        self.df['trend_quality'] = (
            0.25 * (self.df['kumo_thickness_pct'] / 3.0).clip(0, 1) +      # Thick cloud
            0.25 * (self.df['tk_separation_pct'] / 2.0).clip(0, 1) +       # Good TK separation
            0.25 * (self.df['price_kumo_distance_pct'] / 5.0).clip(0, 1) + # Price away from cloud
            0.25 * self.df['chikou_clear'].astype(int)                     # Clear Chikou
        )
        
        # TRADEABLE MARKET (All conditions favorable)
        self.df['tradeable'] = (
            (~self.df['sideways_market']) &
            (~self.df['weak_trend']) &
            (~self.df['choppy_market']) &
            (self.df['trend_quality'] > 0.5)
        )
        
        return self.df

    def generate_ichimoku_signals(self):
        """
        Generates filtered Ichimoku buy and sell signals based on computed indicators.
        This method should be called after compute_ichimoku_with_filters().
        """
        logger.info("Generating Ichimoku signals...")
        
        # ========================================================================
        # PART 4: GENERATE FILTERED BUY SIGNALS
        # ========================================================================
        
        # Get required variables
        high = self.df['high_price']
        low = self.df['low_price']
        close = self.df['closing_price']
        tenkan = self.df['tenkan']
        kijun = self.df['kijun']
        senkou_a = self.df['senkou_a']
        senkou_b = self.df['senkou_b']
        
        # Original signals (unfiltered)
        kumo_twist_raw = (
            (senkou_a.shift(1) < senkou_b.shift(1)) & (senkou_a > senkou_b)
        )
        
        price_breakout_raw = (
            (low > self.df['kumo_bottom']) & 
            (close > self.df['kumo_top']) & 
            (senkou_a > senkou_b)
        )
        
        tk_cross_raw = (tenkan > kijun)
        
        price_kijun_cross_raw = (close > kijun)
        
        #----------------------------------------------------------------------------#
        # FILTERED BUY SIGNALS (Only in favorable conditions)
        #----------------------------------------------------------------------------#
        
        # Filter 1: Kumo Twist (Cloud reversal)
        self.df['kumo_twist_bullish_filtered'] = (
            kumo_twist_raw &
            (self.df['kumo_thickness_pct'] > 0.8) &       # Decent cloud thickness
            (self.df['trend_quality'] > 0.4) &   # Minimum quality
            (self.df['tradeable'])               # Tradeable conditions
        )
        
        # Filter 2: Price Breakout Above Cloud
        self.df['price_kumo_bullish_breakout_filtered'] = (
            price_breakout_raw &
            (self.df['price_kumo_distance_pct'] > 1.5) &  # Decisive breakout (>1.5%)
            (~self.df['false_breakout_risk']) &           # Low false breakout risk
            (~self.df['choppy_market']) &                 # Not choppy
            (self.df['trend_quality'] > 0.5)     # Good quality
        )
        
        # Filter 3: Tenkan-Kijun Bullish Crossover
        self.df['tenkan_kijun_bullish_crossover_filtered'] = (
            tk_cross_raw &
            (self.df['tk_separation_pct'] > 0.5) &        # Meaningful separation
            (~self.df['sideways_market']) &               # Not sideways
            (~self.df['weak_trend']) &                    # Not weak trend
            (self.df['trend_quality'] > 0.4)
        )
        
        # Filter 4: Price-Kijun Crossover
        self.df['price_kijun_bullish_crossover_filtered'] = (
            price_kijun_cross_raw &
            (close > kijun * 1.005) &                     # Clear above (0.5% margin)
            (self.df['chikou_clear']) &                   # Chikou confirms
            (self.df['tradeable'])
        )
        
        # ========================================================================
        # PART 5: GENERATE FILTERED SELL SIGNALS
        # ========================================================================
        
        # Original sell signals (unfiltered)
        bearish_breakout_raw = (
            (high < self.df['kumo_top']) & 
            (close < self.df['kumo_bottom']) & 
            (senkou_a < senkou_b)
        )
        
        bearish_breakdown_raw = (
            (close < self.df['kumo_bottom']) & 
            (low < self.df['kumo_bottom']) & 
            (senkou_a < senkou_b)
        )
        #----------------------------------------------------------------------------#
        # FILTERED SELL SIGNALS
        #----------------------------------------------------------------------------#
        
        # Filter 5: Price Breakout Below Cloud
        self.df['price_kumo_bearish_breakout_filtered'] = (
            bearish_breakout_raw &
            (self.df['price_kumo_distance_pct'] > 1.5) &  # Decisive breakdown
            (~self.df['false_breakout_risk']) &
            (~self.df['choppy_market']) &
            (self.df['trend_quality'] > 0.5)
        )
        
        # Filter 6: Price Sustained Below Cloud
        self.df['price_kumo_bearish_breakdown_filtered'] = (
            bearish_breakdown_raw &
            (close < self.df['kumo_bottom'] * 0.98) &     # Clearly below (2% margin)
            (~self.df['sideways_market']) &
            (self.df['trend_quality'] > 0.4)
        )
        
        # ========================================================================
        # PART 6: STORE UNFILTERED SIGNALS (for comparison/analysis)
        # ========================================================================
        
        self.df['kumo_twist_bullish'] = kumo_twist_raw
        self.df['price_kumo_bullish_breakout'] = price_breakout_raw
        self.df['tenkan_kijun_bullish_crossover'] = tk_cross_raw
        self.df['price_kijun_bullish_crossover'] = price_kijun_cross_raw
        self.df['price_kumo_bearish_breakout'] = bearish_breakout_raw
        self.df['price_kumo_bearish_breakdown'] = bearish_breakdown_raw
        
        logger.info(f"   - Sideways markets detected: {self.df['sideways_market'].sum()}")
        logger.info(f"   - Tradeable periods: {self.df['tradeable'].sum()}")
        logger.info(f"   - Avg trend quality: {self.df['trend_quality'].mean():.2f}")   
        logger.info("Ichimoku signals generated.")
        self.signals_generated = True

    def _identify_strong_signal_limiters(self, latest, signal_direction):
        """
        Identify why a signal did NOT qualify as STRONG BUY / STRONG SELL.
        
        Parameters:
        -----------
        latest : pd.Series
            Latest row of Ichimoku metrics.
        signal_direction : str
            'BUY' or 'SELL'
        
        Returns:
        --------
        str : Comma-separated list of limiting factors.
        """
        limiters = []

        # ----------------------------
        # 1. Trend Quality Issues
        # ----------------------------
        if latest.get('trend_quality', 1) < 0.6:
            limiters.append(f"Trend quality below strong threshold ({latest['trend_quality']:.2f})")

        # ----------------------------
        # 2. Cloud Issues
        # ----------------------------
        if latest.get('kumo_thickness_pct', 1) < 0.8:
            limiters.append(f"Cloud too thin ({latest['kumo_thickness_pct']:.2f}%)")

        # ----------------------------
        # 3. Tenkan-Kijun Weakness
        # ----------------------------
        if latest.get('tk_separation_pct', 1) < 0.5:
            limiters.append(f"TK separation weak ({latest['tk_separation_pct']:.2f}%)")

        # ----------------------------
        # 4. Price-Cloud Distance
        # ----------------------------
        if latest.get('price_kumo_distance_pct', 1) < 1.5:
            limiters.append(f"Price not far enough from cloud ({latest['price_kumo_distance_pct']:.2f}%)")

        # ----------------------------
        # 5. False Breakout Risk
        # ----------------------------
        if latest.get('false_breakout_risk', False):
            limiters.append("False breakout risk present")

        # ----------------------------
        # 6. Choppy or Sideways Conditions
        # ----------------------------
        if latest.get('choppy_market', False):
            limiters.append("Choppy market structure")

        if latest.get('sideways_market', False):
            limiters.append("Sideways market behaviour")

        # ----------------------------
        # 7. Chikou Span Issues
        # ----------------------------
        if not latest.get('chikou_clear', True):
            limiters.append("Chikou span unclear")

        # ----------------------------
        # 8. Volatility Issues
        # ----------------------------
        if latest.get('price_range_pct', 0) > latest.get('avg_range_pct', 1) * 1.5:
            limiters.append("Volatility too high")

        # ----------------------------
        # 9. Direction-specific contradictions
        # ----------------------------
        if signal_direction == 'BUY':
            if latest.get('tenkan', 0) < latest.get('kijun', 0):
                limiters.append("Tenkan below Kijun (bearish momentum)")
            if latest.get('closing_price', 0) < latest.get('kumo_top', 0):
                limiters.append("Price not clearly above cloud")

        elif signal_direction == 'SELL':
            if latest.get('tenkan', 0) > latest.get('kijun', 0):
                limiters.append("Tenkan above Kijun (bullish momentum)")
            if latest.get('closing_price', 0) > latest.get('kumo_bottom', 0):
                limiters.append("Price not clearly below cloud")

        # Return combined limiter string
        return ", ".join(limiters) if limiters else "No limiting factors detected"

    def interpret_signals(self, index=-1):
        """
        Interprets Ichimoku signals for a specific row and returns actionable trading decision.
        
        Parameters:
        -----------
        index : int, default=-1
            Row index to interpret. Default is -1 (latest row).
            
        Returns:
        --------
        dict
            Dictionary containing:
            - action: str (STRONG BUY, MODERATE BUY, etc.)
            - reason: str (explanation of the signal)
            - confidence: str (HIGH, MEDIUM, LOW, N/A)
            - trend_quality: float (0-1 scale)
            - details: str (additional information)
            - signal_type: str (BUY, SELL, WAIT, HOLD)
            - signal_warning: str (warnings about conflicting indicators)
            - signal_limiter: str (why signal strength is limited)
        """
        if not self.signals_generated:
            logger.warning("Ichimoku signals not computed yet. Call compute_ichimoku_with_filters() first.")
            return {
                'action': 'ERROR',
                'reason': 'Signals not computed',
                'confidence': 'N/A',
                'trend_quality': 0.0,
                'details': 'Run compute_ichimoku_with_filters() first',
                'signal_type': 'ERROR',
                'signal_warning': '',
                'signal_limiter': ''
            }
        
        # Get the specified row
        latest = self.df.iloc[index]
        
        # ====================================================================
        # STEP 1: Check if market is tradeable (Market Condition Filters)
        # ====================================================================
        
        if latest['sideways_market'] == True:
            return {
                'action': 'WAIT',
                'reason': 'Sideways market detected (Ichimoku)',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': f"Cloud thickness: {latest['kumo_thickness_pct']:.2f}%, TK separation: {latest['tk_separation_pct']:.2f}% - Market is ranging, avoid trading",
                'signal_type': 'WAIT',
                'signal_warning': '',
                'signal_limiter': 'Sideways market conditions'
            }
        
        if latest['choppy_market'] == True:
            return {
                'action': 'WAIT',
                'reason': 'Choppy market conditions',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': f"Price range: {latest['price_range_pct']:.2f}%, Chikou unclear - Too volatile for reliable signals",
                'signal_type': 'WAIT',
                'signal_warning': 'High volatility detected',
                'signal_limiter': 'Choppy market conditions'
            }
        
        if latest['tradeable'] == False:
            return {
                'action': 'WAIT',
                'reason': 'Unfavorable Ichimoku conditions',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': f"Trend quality: {latest['trend_quality']:.2f} - Wait for better setup",
                'signal_type': 'WAIT',
                'signal_warning': '',
                'signal_limiter': f"Trend quality too low ({latest['trend_quality']:.2f})"
            }
        
        # ====================================================================
        # STEP 2: Check for BUY signals (Priority order: Strong → Weak)
        # ====================================================================
        trend_quality_threshold_strong = 0.6
        trend_quality_threshold_moderate = 0.5
        
        # STRONG BUY: Price breakout above cloud with high quality
        if latest['price_kumo_bullish_breakout_filtered'] == True:
            if latest['trend_quality'] > trend_quality_threshold_strong:
                return {
                    'action': 'STRONG BUY',
                    'reason': 'Price broke above Ichimoku cloud with excellent quality',
                    'confidence': 'HIGH',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Price {latest['price_kumo_distance_pct']:.2f}% above cloud. Cloud thickness: {latest['kumo_thickness_pct']:.2f}%, TK separation: {latest['tk_separation_pct']:.2f}%",
                    'signal_type': 'BUY',
                    'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                    'signal_limiter': ''
                }
            else:
                # Identify specific reasons preventing STRONG BUY
                limiters = self._identify_strong_signal_limiters(latest, 'BUY')
                return {
                    'action': 'MODERATE BUY',
                    'reason': 'Price broke above Ichimoku cloud',
                    'confidence': 'MEDIUM',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Price above cloud but trend quality is moderate ({latest['trend_quality']:.2f})",
                    'signal_type': 'BUY',
                    'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                    'signal_limiter': limiters
                }
        
        # STRONG BUY: Kumo twist (reversal signal)
        if latest['kumo_twist_bullish_filtered'] == True:
            if latest['trend_quality'] > trend_quality_threshold_strong:
                return {
                    'action': 'STRONG BUY',
                    'reason': 'Ichimoku cloud turning bullish (Kumo twist)',
                    'confidence': 'HIGH',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Cloud color changed from bearish to bullish. Trend reversal confirmed. Quality: {latest['trend_quality']:.2f}",
                    'signal_type': 'BUY',
                    'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                    'signal_limiter': ''
                }
            else:
                # Identify specific reasons preventing STRONG BUY
                limiters = self._identify_strong_signal_limiters(latest, 'BUY')
                return {
                    'action': 'MODERATE BUY',
                    'reason': 'Ichimoku cloud turning bullish',
                    'confidence': 'MEDIUM',
                    'trend_quality': latest['trend_quality'],
                    'details': 'Early reversal signal - cloud turning green',
                    'signal_type': 'BUY',
                    'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                    'signal_limiter': limiters
                }
        
        # MODERATE BUY: Tenkan-Kijun bullish crossover
        if latest['tenkan_kijun_bullish_crossover_filtered'] == True:
            if latest['trend_quality'] > trend_quality_threshold_moderate:
                return {
                    'action': 'MODERATE BUY',
                    'reason': 'Tenkan-Kijun bullish crossover',
                    'confidence': 'MEDIUM',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Tenkan crossed above Kijun. Momentum turning positive. TK separation: {latest['tk_separation_pct']:.2f}%",
                    'signal_type': 'BUY',
                    'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                    'signal_limiter': ''
                }
        
        # WEAK BUY: Price-Kijun crossover (support bounce)
        if latest['price_kijun_bullish_crossover_filtered'] == True:
            return {
                'action': 'WEAK BUY',
                'reason': 'Price crossed above Kijun (base line)',
                'confidence': 'LOW',
                'trend_quality': latest['trend_quality'],
                'details': 'Support bounce at Kijun. Good for adding to position or buy-the-dip',
                'signal_type': 'BUY',
                'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                'signal_limiter': 'Trend quality indicates weak momentum'
            }
        
        # WEAK BUY: Price above cloud without fresh signal
        if (latest['closing_price'] > latest['kumo_top'] and
            latest['trend_quality'] > trend_quality_threshold_strong):
            return {
                'action': 'WEAK BUY',
                'reason': 'Price trading above Ichimoku cloud',
                'confidence': 'LOW',
                'trend_quality': latest['trend_quality'],
                'details': f"Price sustained above cloud. Quality: {latest['trend_quality']:.2f}. Consider small position",
                'signal_type': 'BUY',
                'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                'signal_limiter': 'No fresh signal - sustained position only'
            }
        
        # ====================================================================
        # STEP 3: Check for SELL signals (Priority order: Strong → Weak)
        # ====================================================================
        
        # STRONG SELL: Price breakdown below cloud with high quality
        if latest['price_kumo_bearish_breakout_filtered'] == True:
            if latest['trend_quality'] > trend_quality_threshold_strong:
                return {
                    'action': 'STRONG SELL',
                    'reason': 'Price broke below Ichimoku cloud with strong signal',
                    'confidence': 'HIGH',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Price {latest['price_kumo_distance_pct']:.2f}% below cloud. Downtrend confirmed. Quality: {latest['trend_quality']:.2f}",
                    'signal_type': 'SELL',
                    'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                    'signal_limiter': ''
                }
            else:
                return {
                    'action': 'MODERATE SELL',
                    'reason': 'Price broke below Ichimoku cloud',
                    'confidence': 'MEDIUM',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Price below cloud but trend quality is moderate ({latest['trend_quality']:.2f})",
                    'signal_type': 'SELL',
                    'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                    'signal_limiter': 'Trend quality below STRONG SELL threshold'
                }
        
        # MODERATE SELL: Sustained weakness below cloud
        if latest['price_kumo_bearish_breakdown_filtered'] == True:
            if latest['trend_quality'] > trend_quality_threshold_moderate:
                return {
                    'action': 'MODERATE SELL',
                    'reason': 'Price sustained below Ichimoku cloud',
                    'confidence': 'MEDIUM',
                    'trend_quality': latest['trend_quality'],
                    'details': f"Continued weakness below cloud. Stay out or exit positions",
                    'signal_type': 'SELL',
                    'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                    'signal_limiter': ''
                }
        
        # WEAK SELL: Price below cloud without fresh signal
        if (latest['closing_price'] < latest['kumo_bottom'] and
            latest['trend_quality'] > trend_quality_threshold_moderate):
            return {
                'action': 'WEAK SELL',
                'reason': 'Price trading below Ichimoku cloud',
                'confidence': 'LOW',
                'trend_quality': latest['trend_quality'],
                'details': f"Price below cloud. Consider reducing exposure or tightening stops",
                'signal_type': 'SELL',
                'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                'signal_limiter': 'No fresh signal - sustained position only'
            }
        
        # ====================================================================
        # STEP 4: No clear signal - Determine HOLD or WAIT
        # ====================================================================
        
        # Check if price is in cloud (neutral zone)
        if (latest['closing_price'] >= latest['kumo_bottom'] and
            latest['closing_price'] <= latest['kumo_top']):
            return {
                'action': 'WAIT',
                'reason': 'Price inside Ichimoku cloud (neutral zone)',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': 'Wait for price to break out of cloud for clear direction',
                'signal_type': 'WAIT',
                'signal_warning': '',
                'signal_limiter': 'Price in neutral cloud zone'
            }
        
        # Bullish structure but no fresh signal
        if (latest['closing_price'] > latest['kumo_top'] and
            latest['tenkan'] > latest['kijun'] and
            latest['trend_quality'] > trend_quality_threshold_moderate):
            return {
                'action': 'HOLD',
                'reason': 'Bullish Ichimoku structure intact',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': f"Price above cloud, Tenkan > Kijun. Maintain long positions. Quality: {latest['trend_quality']:.2f}",
                'signal_type': 'HOLD',
                'signal_warning': self._build_signal_warnings(latest, 'BUY'),
                'signal_limiter': 'No fresh signal - maintain existing position'
            }
        
        # Bearish structure but no fresh signal
        if (latest['closing_price'] < latest['kumo_bottom'] and
            latest['tenkan'] < latest['kijun'] and
            latest['trend_quality'] > trend_quality_threshold_moderate):
            return {
                'action': 'HOLD',
                'reason': 'Bearish Ichimoku structure intact',
                'confidence': 'N/A',
                'trend_quality': latest['trend_quality'],
                'details': 'Price below cloud, Tenkan < Kijun. Stay out or hold short positions',
                'signal_type': 'HOLD',
                'signal_warning': self._build_signal_warnings(latest, 'SELL'),
                'signal_limiter': 'No fresh signal - maintain existing position'
            }
        
        # Default: No clear signal
        return {
            'action': 'WAIT',
            'reason': 'No clear Ichimoku signal',
            'confidence': 'N/A',
            'trend_quality': latest['trend_quality'],
            'details': f"Trend quality: {latest['trend_quality']:.2f}, Cloud thickness: {latest['kumo_thickness_pct']:.2f}% - Monitor for signal development",
            'signal_type': 'WAIT',
            'signal_warning': '',
            'signal_limiter': 'No qualifying signal conditions met'
        }
    
    def interpret_all_signals(self):
        """
        Interprets signals for all rows in the DataFrame.
        Adds new columns with interpreted signals.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional columns:
            - signal_action
            - signal_reason
            - signal_confidence
            - signal_type
            - signal_details
            - signal_warning
            - signal_limiter
        """
        if not self.signals_generated:
            logger.error("Ichimoku signals not computed yet. Call compute_ichimoku_with_filters() first.")
            return self.df
        
        logger.info("Interpreting Ichimoku signals for all rows...")
        
        # Initialize columns
        self.df['signal_action'] = ''
        self.df['signal_reason'] = ''
        self.df['signal_confidence'] = ''
        self.df['signal_type'] = ''
        self.df['signal_details'] = ''
        self.df['signal_warning'] = ''
        self.df['signal_limiter'] = ''
        
        # Interpret each row
        for idx in range(len(self.df)):
            result = self.interpret_signals(index=idx)
            self.df.at[idx, 'signal_action'] = result['action']
            self.df.at[idx, 'signal_reason'] = result['reason']
            self.df.at[idx, 'signal_confidence'] = result['confidence']
            self.df.at[idx, 'signal_type'] = result['signal_type']
            self.df.at[idx, 'signal_details'] = result['details']
            self.df.at[idx, 'signal_warning'] = result['signal_warning']
            self.df.at[idx, 'signal_limiter'] = result['signal_limiter']
        
        logger.info("Ichimoku signal interpretation complete.")
        logger.info(f"   - STRONG BUY signals: {(self.df['signal_action'] == 'STRONG BUY').sum()}")
        logger.info(f"   - MODERATE BUY signals: {(self.df['signal_action'] == 'MODERATE BUY').sum()}")
        logger.info(f"   - WEAK BUY signals: {(self.df['signal_action'] == 'WEAK BUY').sum()}")
        logger.info(f"   - STRONG SELL signals: {(self.df['signal_action'] == 'STRONG SELL').sum()}")
        logger.info(f"   - MODERATE SELL signals: {(self.df['signal_action'] == 'MODERATE SELL').sum()}")
        logger.info(f"   - WEAK SELL signals: {(self.df['signal_action'] == 'WEAK SELL').sum()}")
        
        return self.df
    

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
        if latest.get('kumo_thickness_pct', 1) < 0.5:
            warnings.append("Thin cloud (weak support/resistance)")
        
        if latest.get('trend_quality', 0.5) < 0.4:
            warnings.append("Low trend quality")
        
        if latest.get('price_range_pct', 2) > 4:
            warnings.append("High volatility")
        
        if signal_type == 'BUY':
            # For BUY signals, warn about bearish Ichimoku indicators
            if latest.get('tenkan', 0) < latest.get('kijun', 0):
                warnings.append("Tenkan below Kijun (bearish momentum)")
            
            if latest.get('closing_price', 0) < latest.get('kumo_top', 0):
                warnings.append("Price below cloud")
                
        elif signal_type == 'SELL':
            # For SELL signals, warn about bullish Ichimoku indicators
            if latest.get('tenkan', 0) > latest.get('kijun', 0):
                warnings.append("Tenkan above Kijun (bullish momentum)")
            
            if latest.get('closing_price', 0) > latest.get('kumo_bottom', 0):
                warnings.append("Price above cloud")
        
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
        self.compute_ichimoku_with_filters()
        
        # Step 2: Generate signals
        self.generate_ichimoku_signals()
        
        # Step 3: Interpret signals
        self.interpret_all_signals()

        logger.info("Complete enhanced analysis finished.")
        return self.df