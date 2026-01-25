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
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from enum import Enum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs
from src.classes.scanner.cl_Swing_VCP import (Timeframe, VCPConfig, Contraction, 
    VCPResult, DataLoader, TechnicalIndicators, 
    PivotDetector, ContractionDetector, SignalValidator)

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error


#---------------------------------------------------------------------------------------------------------------------
#                                            MAIN WORKFLOW
#---------------------------------------------------------------------------------------------------------------------
# ============================================================================
#                          VCP ANALYZER (MAIN)
# ============================================================================

class VCPAnalyzer:
    """
    Main VCP pattern analyzer - orchestrates the entire analysis process.
    This is the primary class that brings together all the other components:
    - DataLoader: Downloads stock data
    - TechnicalIndicators: Calculates indicators
    - ContractionDetector: Finds contractions
    - SignalValidator: Validates signals

    The analyzer follows this workflow:
    1. Load stock data
    2. Calculate technical indicators
    3. Detect contraction phases
    4. Validate the pattern
    5. Calculate confidence score
    6. Determine entry/stop levels
    7. Return comprehensive results

    Usage:
        analyzer = VCPAnalyzer()
        result = analyzer.analyze('AAPL')
    """
    
    def __init__(self, config: Optional[VCPConfig] = None):
        """
        Initialize VCP Analyzer with configuration.
        
        Args:
            config: VCPConfig object with pattern parameters
                If None, uses default configuration
        
        Example:
            # Use default config
            >>> analyzer = VCPAnalyzer()
            
            # Use custom config
            >>> custom_config = VCPConfig(timeframe=Timeframe.WEEKLY)
            >>> analyzer = VCPAnalyzer(custom_config)
        """
        self.config = config or VCPConfig()
        self.data_loader = DataLoader()
        self.indicators = TechnicalIndicators()
        self.contraction_detector = ContractionDetector()
        self.signal_validator = SignalValidator()
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the raw price dataframe.
        
        This method enriches the basic OHLCV data with all the technical
        indicators needed for VCP detection:
        - ATR (volatility)
        - Bollinger Bands (volatility envelopes)
        - RSI (momentum)
        - Moving Averages (trend)
        - OBV (volume trend)
        
        The indicators calculated depend on the configured timeframe
        (daily vs weekly), using appropriate periods for each.
        
        Args:
            df: Raw OHLCV DataFrame (open, high, low, close, volume)
        
        Returns:
            Enhanced DataFrame with additional columns for all indicators
        
        New columns added:
            - atr: Average True Range
            - bb_upper, bb_middle, bb_lower: Bollinger Bands
            - bb_width: Band width percentage
            - rsi: Relative Strength Index
            - ma_21, ma_50, ma_200 (or ma_10, ma_40 for weekly)
            - obv: On-Balance Volume
        
        Example:
            >>> raw_df = data_loader.load_stock_data('AAPL')
            >>> enriched_df = analyzer.prepare_data(raw_df)
            >>> print(enriched_df.columns)
            ['open', 'high', 'low', 'close', 'volume', 'atr', 'bb_width', 'rsi', ...]
        """
        timeframe = self.config.timeframe
    
        # Calculate ATR with appropriate period for timeframe
        atr_period = (self.config.daily_atr_period if timeframe == Timeframe.DAILY 
                    else self.config.weekly_atr_period)
        df['atr'] = self.indicators.calculate_atr(df, atr_period)
        
        # Calculate Bollinger Bands
        bb_period = (self.config.daily_bb_period if timeframe == Timeframe.DAILY 
                    else self.config.weekly_bb_period)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
            self.indicators.calculate_bollinger_bands(df, bb_period)
        df['bb_width'] = self.indicators.calculate_bb_width(df, bb_period)
        
        # Calculate RSI
        rsi_period = (self.config.daily_rsi_period if timeframe == Timeframe.DAILY 
                    else self.config.weekly_rsi_period)
        df['rsi'] = self.indicators.calculate_rsi(df, rsi_period)
        
        # Calculate Moving Averages - different sets for daily vs weekly
        if timeframe == Timeframe.DAILY:
            ma_periods = [
                self.config.daily_ma_short,   # 21
                self.config.daily_ma_medium,  # 50
                self.config.daily_ma_long     # 200
            ]
        else:  # Weekly
            ma_periods = [
                self.config.weekly_ma_short,  # 10
                self.config.weekly_ma_long    # 40
            ]
        
        # Calculate each MA and add to dataframe
        mas = self.indicators.calculate_moving_averages(df, ma_periods)
        for period, ma in mas.items():
            df[f'ma_{period}'] = ma
        
        # Calculate OBV
        df['obv'] = self.indicators.calculate_obv(df)
        
        # Remove any rows with NaN values (from indicator calculations)
        # This typically removes the first ~200 rows on daily charts
        return df.dropna()
    
    def calculate_confidence_score(self, signals: Dict[str, bool], contractions: List[Contraction]) -> float:
        """
        Calculate overall confidence score for the VCP pattern.
        
        The confidence score (0-100) represents how well the pattern
        matches the ideal VCP characteristics. It's a weighted combination
        of all the individual signals.
        
        Scoring system:
        - Each signal has a weight (importance)
        - Signal passed = add weight to score
        - Signal failed = add nothing
        - Bonus points for optimal contraction count
        
        Score interpretation:
        - 85-100: Excellent VCP setup
        - 70-84: Strong VCP setup
        - 60-69: Decent VCP setup (tradeable)
        - < 60: Weak or no VCP pattern
        
        Args:
            signals: Dictionary of signal validations {signal_name: passed}
            contractions: List of Contraction objects
        
        Returns:
            Confidence score from 0.0 to 100.0
        
        Example:
            >>> signals = {
            >>>     'volume_dryup': True,
            >>>     'atr_contraction': True,
            >>>     'rsi_strength': False,
            >>>     ...
            >>> }
            >>> score = calculate_confidence_score(signals, contractions)
            >>> print(f"Confidence: {score:.1f}%")
            Confidence: 72.5%
        """
        # Define weights for each signal (total = 100)
        # More important signals get higher weights
        weights = {
            'volume_dryup': 15,           # Critical: volume must dry up
            'atr_contraction': 15,        # Critical: volatility must contract
            'rsi_strength': 15,           # Critical: must show strength
            'ma_alignment': 10,           # Important: trend confirmation
            'price_near_highs': 10,       # Important: leadership
            'tight_consolidation': 10,    # Important: coiling action
            'contraction_sequence': 25     # Most critical: proper VCP structure
        }
        
        score = 0.0
        
        # Add points for each passed signal
        for signal, passed in signals.items():
            if passed and signal in weights:
                score += weights[signal]
        
        # Bonus points for optimal contraction count
        # 4 contractions is considered optimal (classic VCP)
        if len(contractions) == 4:
            score += 10  # Bonus for perfect count
        elif len(contractions) == 3:
            score += 5   # Smaller bonus for minimum count
        
        # Cap score at 100
        return min(score, 100.0)
    
    def determine_stage(self, df: pd.DataFrame, contractions: List[Contraction]) -> str:
        """
        Determine what stage the VCP pattern is currently in.
        
        Possible stages:
        1. 'no_pattern': No valid VCP detected
        2. 'consolidating': Still in the contraction phase, not broken out
        3. 'breaking_out': Currently breaking out with volume
        4. 'extended': Already broken out and moved significantly
        
        This helps traders know:
        - If they should wait (consolidating)
        - If they should enter now (breaking_out)
        - If they missed it (extended)
        
        Args:
            df: DataFrame with price and volume data
            contractions: List of Contraction objects
        
        Returns:
            String indicating current stage
        
        Stage Logic:
            no_pattern: No contractions found
            consolidating: Price still in consolidation range
            breaking_out: Price >2% above high on high volume
            extended: Price >10% above consolidation high
        
        Example:
            >>> stage = determine_stage(df, contractions)
            >>> if stage == 'breaking_out':
            >>>     print("Entry signal - breaking out now!")
            >>> elif stage == 'consolidating':
            >>>     print("Still waiting - set alert at breakout level")
        """
        if not contractions:
            return "no_pattern"
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Find the last contraction
        last_contraction = contractions[-1]
        
        # Find the highest point of the consolidation
        # (from start of last contraction to present)
        consolidation_high = df['high'].iloc[last_contraction.start_idx:].max()
        
        # Calculate average volume
        volume_period = (self.config.daily_volume_period 
                        if self.config.timeframe == Timeframe.DAILY 
                        else self.config.weekly_volume_period)
        avg_volume = df['volume'].iloc[-volume_period:].mean()
        current_volume = df['volume'].iloc[-1]
        
        # Determine stage based on price and volume
        
        # Breaking out: Price >2% above high AND volume significantly elevated
        if (current_price > consolidation_high * 1.02 and 
            current_volume > avg_volume * self.config.breakout_volume_threshold):
            return "breaking_out"
        
        # Extended: Price >10% above consolidation high (breakout already happened)
        elif current_price > consolidation_high * 1.10:
            return "extended"
        
        # Still consolidating: Price still in the range
        else:
            return "consolidating"
    
    def calculate_entry_and_stop(self, df: pd.DataFrame, contractions: List[Contraction]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate suggested entry price and stop loss level.
        
        Entry Strategy:
        - Enter on breakout above consolidation high
        - Use 2% buffer to avoid false breakouts
        - Entry = Consolidation High × 1.02
        
        Stop Loss Strategy:
        - Use tighter of two methods:
        1. Below last swing low (technical support)
        2. 2× ATR below entry (volatility-based)
        - Gives 2-3% buffer below entry
        
        Args:
            df: DataFrame with price and ATR data
            contractions: List of Contraction objects
        
        Returns:
            Tuple of (entry_price, stop_loss) or (None, None) if no pattern
        
        Example:
            >>> entry, stop = calculate_entry_and_stop(df, contractions)
            >>> risk = ((entry - stop) / entry) * 100
            >>> print(f"Entry: ${entry:.2f}")
            >>> print(f"Stop: ${stop:.2f}")
            >>> print(f"Risk: {risk:.2f}%")
            Entry: $51.20
            Stop: $48.50
            Risk: 5.27%
        """
        if not contractions:
            return None, None
        
        # Get last contraction
        last_contraction = contractions[-1]
        
        # Entry: 2% above the consolidation high
        # This filters out false breakouts and ensures commitment
        consolidation_high = df['high'].iloc[last_contraction.start_idx:].max()
        entry_price = consolidation_high * 1.02  # 2% above high
        
        # Stop Loss Method 1: Below last swing low
        # Find the lowest low in the consolidation
        consolidation_low = df['low'].iloc[last_contraction.start_idx:].min()
        current_atr = df['atr'].iloc[-1]
        
        # Use tighter of the two
        atr_stop = entry_price - (2 * current_atr)
        swing_stop = consolidation_low * 0.98
        
        # Use the tighter (higher) of the two stops
        # This provides better risk/reward while still allowing breathing room
        stop_loss = max(atr_stop, swing_stop)
        
        return entry_price, stop_loss
    
    def analyze(self, symbol: str, period: str = "1y") -> VCPResult:
        """
        Main analysis function - performs complete VCP pattern detection.
        
        This is the primary method that orchestrates the entire analysis workflow:
        1. Load historical price data
        2. Convert to weekly if configured
        3. Calculate all technical indicators
        4. Detect contraction phases
        5. Validate contraction sequence
        6. Check all individual signals
        7. Calculate confidence score
        8. Determine current stage
        9. Calculate entry/stop levels
        10. Return comprehensive results
        
        The method is designed to be robust, handling errors gracefully and
        providing detailed feedback about the analysis process.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
            period: Data period to analyze
                '6mo' = 6 months (minimum for VCP)
                '1y' = 1 year (recommended default)
                '2y' = 2 years (for finding older patterns)
        
        Returns:
            VCPResult object containing:
            - is_vcp: Whether valid pattern found
            - confidence_score: 0-100 quality score
            - contractions: List of all contractions
            - current_stage: Pattern stage
            - breakout_price: Suggested entry
            - stop_loss: Suggested stop
            - signals: All signal validations
            - metrics: Current technical values
            - messages: Detailed analysis log
        
        Example:
            >>> analyzer = VCPAnalyzer()
            >>> result = analyzer.analyze('AAPL', period='1y')
            >>> 
            >>> if result.is_vcp:
            >>>     print(f"VCP found! Confidence: {result.confidence_score:.1f}%")
            >>>     print(f"Stage: {result.current_stage}")
            >>>     print(f"Entry: ${result.breakout_price:.2f}")
            >>>     print(f"Stop: ${result.stop_loss:.2f}")
            >>> else:
            >>>     print("No VCP pattern detected")
            >>>     for msg in result.messages:
            >>>         print(f"  {msg}")
        """
        messages = []
        
        try:
            # ================================================================
            # STEP 1: LOAD DATA
            # ================================================================
            df = self.data_loader.load_stock_data(symbol, period)
            messages.append(f"Loaded {len(df)} bars of data for {symbol}")
            
            # ================================================================
            # STEP 2: CONVERT TO WEEKLY IF NEEDED
            # ================================================================
            if self.config.timeframe == Timeframe.WEEKLY:
                df = self.data_loader.convert_to_weekly(df)
                messages.append(f"Converted to weekly timeframe: {len(df)} bars")
            
            # ================================================================
            # STEP 3: CALCULATE TECHNICAL INDICATORS
            # ================================================================
            df = self.prepare_data(df)
            messages.append("Technical indicators calculated")
            
            # ================================================================
            # STEP 4: DETECT CONTRACTIONS
            # ================================================================
            contractions = self.contraction_detector.identify_contractions(df, self.config)
            messages.append(f"Detected {len(contractions)} potential contractions")
            
            # Log details of each contraction found
            if contractions:
                for i, c in enumerate(contractions):
                    messages.append(
                        f"  Contraction {i + 1}: {c.depth_percent:.2f}% depth, "
                        f"{c.duration_bars} bars"
                    )
            
            # ================================================================
            # STEP 5: VALIDATE CONTRACTION SEQUENCE
            # ================================================================
            sequence_valid, sequence_messages = \
                self.contraction_detector.validate_contraction_sequence(
                    contractions, self.config
                )
            messages.extend(sequence_messages)
            
            # ================================================================
            # STEP 6: VALIDATE INDIVIDUAL SIGNALS
            # ================================================================
            signals = {}  # Will store all signal validations
            
            # Check volume dry-up
            vol_valid, vol_msg = self.signal_validator.check_volume_dryup(
                df, contractions, self.config
            )
            signals['volume_dryup'] = vol_valid
            messages.append(vol_msg)
            
            # Check ATR contraction
            atr_valid, atr_msg = self.signal_validator.check_atr_contraction(
                df, contractions, self.config
            )
            signals['atr_contraction'] = atr_valid
            messages.append(atr_msg)
            
            # Check RSI strength
            rsi_valid, rsi_msg = self.signal_validator.check_rsi_strength(
                df, contractions, self.config
            )
            signals['rsi_strength'] = rsi_valid
            messages.append(rsi_msg)
            
            # Check MA alignment
            ma_valid, ma_msg = self.signal_validator.check_ma_alignment(
                df, self.config
            )
            signals['ma_alignment'] = ma_valid
            messages.append(ma_msg)
            
            # Check price position
            price_valid, price_msg = self.signal_validator.check_price_near_highs(df)
            signals['price_near_highs'] = price_valid
            messages.append(price_msg)
            
            # Check tight consolidation
            tight_valid, tight_msg = self.signal_validator.check_tight_consolidation(
                df, self.config
            )
            signals['tight_consolidation'] = tight_valid
            messages.append(tight_msg)
            
            # Add contraction sequence validation to signals
            signals['contraction_sequence'] = sequence_valid
            
            # ================================================================
            # STEP 7: CALCULATE CONFIDENCE SCORE
            # ================================================================
            confidence = self.calculate_confidence_score(signals, contractions)
            
            # ================================================================
            # STEP 8: DETERMINE IF THIS IS A VALID VCP
            # ================================================================
            # Needs minimum confidence (60%) and minimum contractions
            is_vcp = (confidence >= 60.0 and 
                    len(contractions) >= self.config.min_contractions)
            
            # ================================================================
            # STEP 9: DETERMINE CURRENT STAGE
            # ================================================================
            stage = self.determine_stage(df, contractions)
            
            # ================================================================
            # STEP 10: CALCULATE ENTRY AND STOP LEVELS
            # ================================================================
            entry, stop = self.calculate_entry_and_stop(df, contractions)
            
            # ================================================================
            # STEP 11: COLLECT CURRENT METRICS
            # ================================================================
            metrics = {
                'current_price': df['close'].iloc[-1],
                'current_atr': df['atr'].iloc[-1],
                'current_rsi': df['rsi'].iloc[-1],
                'bb_width': df['bb_width'].iloc[-1],
                'num_contractions': len(contractions),
                'latest_volume_ratio': (df['volume'].iloc[-5:].mean() / 
                                    df['volume'].iloc[-50:].mean())
            }
            
            # ================================================================
            # STEP 12: CREATE SUMMARY MESSAGES
            # ================================================================
            messages.append(f"\n{'='*60}")
            messages.append(f"VCP Analysis Result: {'VALID VCP ✓' if is_vcp else 'NOT VCP ✗'}")
            messages.append(f"Confidence Score: {confidence:.1f}/100")
            messages.append(f"Current Stage: {stage}")
            
            if entry and stop:
                risk_percent = ((entry - stop) / entry * 100)
                messages.append(f"Entry Level: ${entry:.2f}")
                messages.append(f"Stop Loss: ${stop:.2f}")
                messages.append(f"Risk: {risk_percent:.2f}%")
            
            # ================================================================
            # STEP 13: RETURN RESULTS
            # ================================================================
            return VCPResult(
                is_vcp=is_vcp,
                confidence_score=confidence,
                contractions=contractions,
                current_stage=stage,
                breakout_price=entry,
                stop_loss=stop,
                signals=signals,
                metrics=metrics,
                messages=messages
            )
        
        except Exception as e:
            # If any error occurs, return error result
            messages.append(f"Error during analysis: {str(e)}")
            return VCPResult(
                is_vcp=False,
                confidence_score=0.0,
                contractions=[],
                current_stage="error",
                breakout_price=None,
                stop_loss=None,
                signals={},
                metrics={},
                messages=messages
            )
# ============================================================================
#                     BATCH SCANNER - FOR MULTIPLE STOCKS
# ============================================================================

class VCPScanner:
    """
    Scan multiple stocks for VCP patterns in batch mode
    This class provides efficient batch scanning capabilities:
    - Process multiple symbols in one operation
    - Filter results by minimum confidence
    - Return sorted results (best patterns first)
    - Handle errors gracefully (continue on failures)

    Useful for:
    - Scanning watchlists for VCP setups
    - Finding the best VCP candidates from a universe
    - Regular screening of stocks for new patterns

    Example workflow:
    1. Define a list of stocks to scan
    2. Set minimum confidence threshold
    3. Run scan
    4. Get back only stocks meeting criteria
    5. Review top candidates for trading

    """
    def __init__(self, config: Optional[VCPConfig] = None):
        """
        Initialize scanner with VCP configuration.
        
        Args:
            config: VCPConfig object with pattern parameters
                If None, uses default configuration
        
        Example:
            >>> # Use default config
            >>> scanner = VCPScanner()
            >>> 
            >>> # Use custom config (e.g., for weekly charts)
            >>> weekly_config = VCPConfig(timeframe=Timeframe.WEEKLY)
            >>> scanner = VCPScanner(weekly_config)
        """
        self.analyzer = VCPAnalyzer(config)

    def scan_stocks(self, symbols: List[str], min_confidence: float = 60.0) -> pd.DataFrame:
        """
        Scan multiple stocks for VCP patterns.
        
        This method processes each symbol in the list, analyzes it for
        VCP patterns, and returns only those meeting the confidence threshold.
        
        The scan continues even if individual stocks fail (network issues,
        invalid symbols, etc.), ensuring maximum coverage.
        
        Args:
            symbols: List of stock ticker symbols to scan
                    Example: ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
            min_confidence: Minimum confidence score to include (0-100)
                        Default 60.0 = only decent or better patterns
                        Use 70+ for stronger setups only
        
        Returns:
            DataFrame with VCP scan results, sorted by confidence (best first)
            
            Columns:
            - symbol: Stock ticker
            - is_vcp: Boolean indicating valid pattern
            - confidence: Confidence score (0-100)
            - stage: Current pattern stage
            - num_contractions: Number of contractions found
            - entry_price: Suggested breakout entry
            - stop_loss: Suggested stop loss
            - current_price: Current stock price
            - rsi: Current RSI value
            - volume_dryup, atr_contraction, etc.: Individual signals (True/False)
        
        Example:
            >>> scanner = VCPScanner()
            >>> watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD']
            >>> 
            >>> # Find all VCP patterns with 60+ confidence
            >>> results = scanner.scan_stocks(watchlist, min_confidence=60.0)
            >>> 
            >>> if not results.empty:
            >>>     print(f"Found {len(results)} VCP patterns:")
            >>>     print(results[['symbol', 'confidence', 'stage', 'entry_price']])
            >>> else:
            >>>     print("No VCP patterns found")
            
            Output:
            Found 2 VCP patterns:
            symbol  confidence         stage  entry_price
            0   NVDA        78.5  consolidating        51.20
            1   MSFT        65.0  breaking_out        420.50
        """
        results = []
        
        for symbol in symbols:
            print(f"Scanning {symbol}...")
            
            try:
                # Analyze this symbol for VCP
                result = self.analyzer.analyze(symbol)
                
                # Only include if meets minimum confidence
                if result.confidence_score >= min_confidence:
                    results.append({
                        'symbol': symbol,
                        'is_vcp': result.is_vcp,
                        'confidence': result.confidence_score,
                        'stage': result.current_stage,
                        'num_contractions': len(result.contractions),
                        'entry_price': result.breakout_price,
                        'stop_loss': result.stop_loss,
                        'current_price': result.metrics.get('current_price'),
                        'rsi': result.metrics.get('current_rsi'),
                        **result.signals
                    })
            
            except Exception as e:
                # If this symbol fails, log it but continue with others
                print(f"  Error scanning {symbol}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        
        # Sort by confidence (best patterns first)
        if not df.empty:
            df = df.sort_values('confidence', ascending=False)
        
        return df

# ============================================================================
#                     USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of VCP analyzer
        This function shows:
            1. How to create a configuration
            2. How to analyze a single stock
            3. How to interpret the results
            4. How to scan multiple stocks
    """
    # Create configuration
    config = VCPConfig(
        timeframe=Timeframe.DAILY,
        min_contractions=3
    )
    #------------------------------------------------------------------------------------------
    #                         Single stock analysis
    #------------------------------------------------------------------------------------------
    print("="*70)
    print("VCP PATTERN ANALYZER")
    print("="*70)

    analyzer = VCPAnalyzer(config)

    # Analyze a stock
    symbol = "INFY"
    result = analyzer.analyze(symbol, period="1y")

    # Print results
    print(f"\nAnalyzing: {symbol}")
    print("-"*70)

    for message in result.messages:
        print(message)

    print("\n" + "="*70)
    print("SIGNAL SUMMARY")
    print("="*70)

    for signal, passed in result.signals.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{signal:.<30} {status}")

    #-------------------------------------------------------------------------------------------
    #                        Multiple stock analysis
    #-------------------------------------------------------------------------------------------
    # Batch scanning example
    print("\n" + "="*70)
    print("BATCH SCANNING")
    print("="*70)

    scanner = VCPScanner(config)

    # Example stock list
    watchlist = cfg_nifty.nifty_metal

    scan_results = scanner.scan_stocks(watchlist, min_confidence=50.0)

    if not scan_results.empty:
        print("\nStocks with VCP patterns:")
        print(scan_results.to_string(index=False))
    else:
        print("\nNo VCP patterns found in watchlist")

#-------------------------------------------------------------------------------------------
#                        Trigger the Swing Scanner
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()