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

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

# ============================================================================

"""
VCP (Volatility Contraction Pattern) Detection Algorithm
=========================================================
This module provides a comprehensive, modular implementation for identifying 
Volatility Contraction Patterns (VCP) in stock price data.

VCP Pattern Overview:
- A consolidation pattern where price volatility progressively contracts
- Each successive pullback is shallower than the previous one
- Volume dries up during contractions
- Pattern typically forms before significant price breakouts
- Popularized by Mark Minervini for swing trading

Author: Shree Dwarkadhish
Date: 2026-01-24
"""

# ============================================================================
# DATA MODELS
# ============================================================================

class Timeframe(Enum):
    """
    Enumeration of supported chart timeframes for analysis.
    
    VCP patterns can be identified on both daily and weekly charts,
    with different parameter settings for each timeframe.
    """
    DAILY = "daily"    # For intraday to multi-week swing trades
    WEEKLY = "weekly"  # For longer-term position trades

@dataclass
class VCPConfig:
    """
    Configuration parameters for VCP pattern detection.
    
    This class contains all the tunable parameters that define what constitutes
    a valid VCP pattern. Different markets or trading styles may require
    different parameter values.
    
    Attributes:
        timeframe: Whether to analyze daily or weekly charts
        
        min_contractions: Minimum number of consolidation phases required
                         (3 is standard, but some patterns have 4-5)
        max_contractions: Maximum contractions to look for (prevents false positives)
        
        daily_atr_period: Number of days for ATR calculation on daily charts
        daily_bb_period: Number of days for Bollinger Bands on daily charts
        daily_ma_short: Short-term moving average period (21 = ~1 month)
        daily_ma_medium: Medium-term moving average period (50 = ~10 weeks)
        daily_ma_long: Long-term moving average period (200 = ~1 year)
        daily_rsi_period: RSI calculation period for daily charts
        daily_volume_period: Period for average volume comparison (daily)
        
        weekly_atr_period: Number of weeks for ATR calculation
        weekly_bb_period: Number of weeks for Bollinger Bands
        weekly_ma_short: Short MA for weekly (10 weeks = 50 days)
        weekly_ma_long: Long MA for weekly (40 weeks = 200 days)
        weekly_rsi_period: RSI period for weekly charts
        weekly_volume_period: Period for volume comparison (weekly)
        
        contraction_1_min/max: Valid depth range for first pullback (%)
        contraction_2_min/max: Valid depth range for second pullback (%)
        contraction_3_min/max: Valid depth range for third pullback (%)
        contraction_4_max: Maximum depth for optional fourth pullback (%)
        
        volume_dryup_threshold: How low volume should drop (0.60 = 60% of avg)
        breakout_volume_threshold: How high volume should spike on breakout
        
        rsi_floor: RSI shouldn't drop below this in later contractions
        rsi_consolidation_min/max: Expected RSI range during final consolidation
        rsi_breakout: RSI level that confirms breakout momentum
        
        atr_contraction_threshold: How much ATR should contract (0.70 = 30% reduction)
        
        tight_range_threshold: What constitutes a "tight" day (1.5% range)
        min_tight_days: Minimum consecutive tight days required
        
        ma_alignment_required: Whether to enforce bullish MA alignment
        
        rs_threshold: How close to RS highs stock should be (0.90 = within 10%)

        FIXED: Relaxed thresholds calibrated for Indian equity markets.
    
    Key changes from original:
        - Wider contraction depth ranges (Indian stocks are more volatile)
        - Higher volume_dryup_threshold (0.60 → 0.75) - less strict
        - Higher atr_contraction_threshold (0.70 → 0.80) - less strict
        - Lower rsi_floor (40 → 35) - Indian stocks can dip more
        - Wider RSI consolidation range (48-65 → 40-70)
        - Looser tight_range_threshold (1.5% → 2.5%)
        - Fewer min_tight_days (3 → 2)
        - Added lookback_days to focus on recent patterns only
    """
    # Timeframe selection
    timeframe: Timeframe = Timeframe.DAILY
    
    # Contraction count requirements
    # Most VCP patterns have 3-4 contractions; fewer than 3 is usually not a VCP
    min_contractions: int = 3
    max_contractions: int = 5
    
    # Daily chart technical indicator periods
    # These are standard values used by most traders
    daily_atr_period: int = 14          # Wilder's original ATR period
    daily_bb_period: int = 20           # Standard Bollinger Band period
    daily_ma_short: int = 21            # ~1 month of trading days
    daily_ma_medium: int = 50           # ~10 weeks / 2.5 months
    daily_ma_long: int = 200            # ~1 year of trading
    daily_rsi_period: int = 14          # Wilder's original RSI period
    daily_volume_period: int = 50       # ~10 weeks for volume average
    
    # Weekly chart technical indicator periods
    # Weekly periods are shorter because each bar represents more time
    weekly_atr_period: int = 10         # ~2.5 months
    weekly_bb_period: int = 20          # ~5 months
    weekly_ma_short: int = 10           # 10 weeks = 50 days
    weekly_ma_long: int = 40            # 40 weeks = 200 days
    weekly_rsi_period: int = 14         # Same as daily
    weekly_volume_period: int = 10      # 10 weeks for average
    
    # Contraction depth thresholds (percentage from high to low)
    # These define the "progressive tightening" characteristic of VCP
    # Each contraction should be shallower than the previous one
    # FIX 1a: Wider contraction depth ranges for Indian markets
    # Indian stocks can have deeper initial pullbacks
    contraction_1_min: float = 5.0      # Was 8.0 → relaxed to 5.0
    contraction_1_max: float = 35.0     # Was 25.0 → relaxed to 35.0
    contraction_2_min: float = 3.0      # Was 5.0 → relaxed to 3.0
    contraction_2_max: float = 25.0     # Was 15.0 → relaxed to 25.0
    contraction_3_min: float = 2.0      # Was 3.0 → relaxed to 2.0
    contraction_3_max: float = 15.0     # Was 10.0 → relaxed to 15.0
    contraction_4_max: float = 10.0     # Was 6.0 → relaxed to 10.0
    
    # Volume thresholds
    # Volume should "dry up" (decrease) as contractions tighten
    # Then spike on breakout (showing new buying interest)
    # FIX 1b: Relaxed volume threshold
    # 0.75 means volume should be below 75% of average (was 60%)
    volume_dryup_threshold: float = 0.75        # Was 0.60
    breakout_volume_threshold: float = 1.30     # Was 1.50 → easier to trigger
    
    # RSI thresholds
    # RSI should stay elevated (not oversold) during VCP contractions
    # This shows underlying strength despite price consolidation
    # FIX 1c: Relaxed RSI thresholds
    rsi_floor: float = 35.0                     # Was 40.0
    rsi_consolidation_min: float = 40.0         # Was 48.0
    rsi_consolidation_max: float = 70.0         # Was 65.0
    rsi_breakout: float = 55.0                  # Was 60.0

    # ATR contraction requirement
    # ATR measures volatility; it should decrease as pattern tightens
    # 0.70 means final ATR should be 70% or less of initial ATR (30% reduction)
    # FIX 1d: Relaxed ATR contraction
    # 0.80 means final ATR only needs to be 80% of initial (20% reduction)
    atr_contraction_threshold: float = 0.80     # Was 0.70    
    
    # Price tightness requirements
    # "Tight" days have small high-low ranges, showing low volatility
    # FIX 1e: Relaxed tight consolidation
    tight_range_threshold: float = 0.025        # Was 0.015 (2.5% range allowed)
    min_tight_days: int = 2                     # Was 3
    
    # Moving average alignment
    # Bullish setup: shorter MAs above longer MAs (uptrend confirmation)
    ma_alignment_required: bool = True
    
    # Relative strength requirement
    # Stock should be outperforming the market (within 10% of RS highs)
    rs_threshold: float = 0.90

    # FIX 1f: NEW - Lookback window to focus on RECENT patterns only
    # Avoids detecting patterns from 18+ months ago that are irrelevant
    lookback_days: int = 180            # Focus on last 6 months for daily
    lookback_weeks: int = 26            # Focus on last 26 weeks for weekly

    # FIX 1g: NEW - Pivot lookback (separate from contraction lookback)
    # Larger value = fewer but more significant pivots
    pivot_lookback: int = 5             # Was hardcoded as 3 (too small)

    # FIX 1h: NEW - Minimum contraction duration
    # Avoids micro-contractions caused by noise
    min_contraction_duration: int = 5   # Minimum 5 bars per contraction

    # FIX 1i: NEW - ATR tolerance for non-strict monotonic decline
    # Allows slight ATR increases between contractions (10% tolerance)
    atr_tolerance: float = 0.10         # 10% tolerance

@dataclass
class Contraction:
    """
    Represents a single contraction (consolidation) phase within a VCP pattern.
    
    A contraction is a pullback from a swing high to a swing low, followed by
    recovery. VCP patterns consist of multiple contractions with progressively
    decreasing depth and volatility.
    
    Attributes:
        start_idx: Index position where contraction begins (swing high)
        end_idx: Index position where contraction ends (swing low)
        high_price: Highest price during this contraction phase
        low_price: Lowest price during this contraction phase
        depth_percent: How far price pulled back from high to low (percentage)
        duration_bars: Number of bars (days/weeks) the contraction lasted
        avg_volume: Average volume during the contraction period
        avg_atr: Average ATR during the contraction (volatility measure)
        rsi_low: Lowest RSI reading during the contraction
    
    Example:
        Contraction(
            start_idx=50,           # Day 50 of the chart
            end_idx=65,             # Day 65 of the chart
            high_price=100.00,      # Peak at $100
            low_price=88.00,        # Pullback to $88
            depth_percent=12.0,     # 12% pullback
            duration_bars=15,       # Lasted 15 days
            avg_volume=1_000_000,   # Average 1M shares/day
            avg_atr=2.50,           # ATR of $2.50
            rsi_low=45.0            # RSI dipped to 45
        )
    """
    #start_idx: int          # Where this contraction phase started
    #end_idx: int            # Where it ended (the low point)
    start_pos: int          # FIX: Use positional index, not label index
    end_pos: int
    start_date: object      # Store actual date for reference
    end_date: object

    high_price: float       # The high price at start
    low_price: float        # The low price at end
    depth_percent: float    # Depth of pullback: (high - low) / high * 100
    duration_bars: int      # How many bars it took
    avg_volume: float       # Average volume during contraction
    avg_atr: float          # Average ATR (volatility measure)
    rsi_low: float          # Lowest RSI reading during contraction

@dataclass
class VCPResult:
    """
    Complete results of VCP pattern detection analysis.
    
    This class encapsulates all the information from analyzing a stock
    for VCP patterns, including whether a pattern exists, how strong it is,
    and specific trading levels.
    
    Attributes:
        is_vcp: Boolean indicating if a valid VCP pattern was detected
        confidence_score: Numerical score 0-100 indicating pattern quality
                         (60+ = decent, 75+ = strong, 85+ = excellent)
        contractions: List of all Contraction objects found
        current_stage: Where the pattern is now:
                      - 'consolidating': Still in the pattern
                      - 'breaking_out': Currently breaking out
                      - 'extended': Already extended beyond breakout
                      - 'no_pattern': No VCP detected
        breakout_price: Suggested entry price for the breakout
        stop_loss: Suggested stop loss level for risk management
        signals: Dictionary of individual signal checks (True/False for each)
        metrics: Dictionary of current technical measurements
        messages: List of human-readable analysis messages
    
    Example:
        VCPResult(
            is_vcp=True,
            confidence_score=78.5,
            contractions=[...],
            current_stage='consolidating',
            breakout_price=51.20,
            stop_loss=48.50,
            signals={'volume_dryup': True, 'rsi_strength': True, ...},
            metrics={'current_price': 50.00, 'current_rsi': 55.2, ...},
            messages=['Loaded 252 bars...', 'Detected 3 contractions...']
        )
    """
    is_vcp: bool                      # Is this a valid VCP pattern?
    confidence_score: float           # 0-100 score
    contractions: List[Contraction]   # All contractions found
    current_stage: str                # Current pattern stage
    breakout_price: Optional[float]   # Where to enter (None if no pattern)
    stop_loss: Optional[float]        # Where to place stop (None if no pattern)
    signals: Dict[str, bool]          # Individual signal validations
    metrics: Dict[str, float]         # Current technical metrics
    messages: List[str]               # Detailed analysis log
    fail_reasons: List[str]     # FIX: Explicit fail reasons for debugging


# ============================================================================
# DATA ACQUISITION
# ============================================================================

class DataLoader:
    """
    Handles downloading and preparing stock price data.
    
    This class is responsible for:
    1. Downloading historical price data from Yahoo Finance
    2. Converting daily data to weekly timeframe when needed
    3. Ensuring data quality and proper formatting
    
    The class uses yfinance library to fetch data, which provides
    free access to Yahoo Finance historical data.
    """

    @staticmethod
    def load_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance.
        
        This method fetches OHLCV (Open, High, Low, Close, Volume) data
        for a given stock symbol. The data is cleaned and formatted for
        further analysis.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
            period: How far back to fetch data. Options include:
                   - '1mo' = 1 month
                   - '3mo' = 3 months
                   - '6mo' = 6 months
                   - '1y' = 1 year (default, good for VCP detection)
                   - '2y' = 2 years
                   - 'max' = all available data
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index is DatetimeIndex with trading dates
        
        Raises:
            ValueError: If no data is returned (invalid symbol or period)
            Exception: If there's a network error or API issue
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_stock_data('AAPL', period='1y')
            >>> print(df.head())
                          open   high    low  close     volume
            2025-01-24  150.0  152.0  149.5  151.5  50000000
        """
        try:
            # Create a Ticker object for the symbol
            stock = yf.Ticker(symbol)
            
            # Download historical data
            # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            df = stock.history(period=period)
            
            # Check if data was actually returned
            if df.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Standardize column names to lowercase for consistency
            # yfinance returns capitalized names, we want: open, high, low, close, volume
            df.columns = [col.lower() for col in df.columns]
            
            return df
        
        except Exception as e:
            # Wrap any errors with context about which symbol failed
            raise Exception(f"Error loading data for {symbol}: {str(e)}")
    
    @staticmethod
    def convert_to_weekly(symbol: str) -> pd.DataFrame:
        """
        Read pre-calculated weekly data from Excel file for a specific symbol.
        
        Loads weekly OHLCV data from the Excel file, filters by symbol,
        selects the required columns, and renames them to standard format.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
        Returns:
            DataFrame with weekly OHLCV data (fewer rows than daily)
            Columns: open, high, low, close, volume
            Index: DatetimeIndex with dates
        
        Example:
            >>> weekly_data = convert_to_weekly('AAPL')
            >>> print(f"Loaded {len(weekly_data)} weeks for AAPL")
        """
        try:
            # Path to the Excel file
            excel_path = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/weekly_data/stocks_weekly_data.xlsx'
            
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Filter by symbol
            df = df[df['symbol'] == symbol]
            
            if df.empty:
                raise ValueError(f"No weekly data found for symbol {symbol}")
            
            # Select only the required columns
            df = df[['date', 'w_open', 'w_high', 'w_low', 'w_close', 'w_volume']]
            
            # Rename columns
            df.rename(columns={
                'w_open': 'open',
                'w_high': 'high',
                'w_low': 'low',
                'w_close': 'close',
                'w_volume': 'volume'
            }, inplace=True)
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date (in case not sorted)
            df.sort_index(inplace=True)
            
            return df
        
        except Exception as e:
            raise Exception(f"Error loading weekly data for {symbol}: {str(e)}")


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """
    Calculate various technical indicators used in VCP pattern detection.
    
    This class contains static methods for calculating common technical
    indicators from price data. Each method is self-contained and can be
    used independently.
    
    Indicators calculated:
    - ATR: Average True Range (volatility)
    - Bollinger Bands: Price envelopes based on standard deviation
    - RSI: Relative Strength Index (momentum)
    - Moving Averages: Trend-following indicators
    - OBV: On-Balance Volume (accumulation/distribution)
    - Relative Strength: Performance vs. market
    
    All calculations follow standard financial formulas used by
    professional traders and charting platforms.
    """
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures market volatility by decomposing the entire range of
        an asset price for that period. It was introduced by J. Welles Wilder.
        
        True Range is the greatest of:
        1. Current High - Current Low
        2. Absolute value of Current High - Previous Close
        3. Absolute value of Current Low - Previous Close
        
        ATR is the moving average of True Range over the specified period.
        
        For VCP detection, we want to see ATR declining (contracting volatility).
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Number of periods for the moving average (default 14)
        
        Returns:
            Series of ATR values, same length as input DataFrame
        
        Example:
            >>> atr = TechnicalIndicators.calculate_atr(df, period=14)
            >>> print(f"Current ATR: ${atr.iloc[-1]:.2f}")
            Current ATR: $2.45
            
            Higher ATR = More volatile
            Lower ATR = Less volatile (what we want in VCP)
            FIX: Added min_periods=1 to handle short data windows gracefully.
        """
        # Extract price series
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate three components of True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # FIX: min_periods=1 avoids NaN for early rows
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of:
        - Middle Band: Simple moving average (SMA)
        - Upper Band: SMA + (Standard Deviation × multiplier)
        - Lower Band: SMA - (Standard Deviation × multiplier)
        
        The bands expand during volatile periods and contract during
        quiet periods. For VCP, we want to see the bands contracting
        (narrowing), indicating decreasing volatility.
        
        Standard parameters: 20-period SMA, 2 standard deviations
        
        Args:
            df: DataFrame with 'close' column
            period: Number of periods for SMA (default 20)
            std: Standard deviation multiplier (default 2.0)
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band) as Series
        
        Example:
            >>> upper, middle, lower = calculate_bollinger_bands(df)
            >>> current_price = df['close'].iloc[-1]
            >>> print(f"Price ${current_price:.2f} vs Bands: ${lower.iloc[-1]:.2f} - ${upper.iloc[-1]:.2f}")
            
            In VCP:
            - Price should stay in upper half of bands
            - Band width should be contracting
        """
        close = df['close']
        
        # Middle band is just the moving average
        middle = close.rolling(window=period).mean()
        
        # Calculate standard deviation over the same period
        std_dev = close.rolling(window=period).std()
        
        # Upper band: Add (std_dev × multiplier) to middle band
        upper = middle + (std_dev * std)
        
        # Lower band: Subtract (std_dev × multiplier) from middle band
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_bb_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Bollinger Band Width.
        
        BB Width measures the distance between upper and lower Bollinger Bands
        as a percentage of the middle band. It's a direct measure of volatility.
        
        Formula: ((Upper Band - Lower Band) / Middle Band) × 100
        
        For VCP detection:
        - BB Width should be at multi-month lows
        - Width should be progressively declining
        - Very low width indicates a volatility squeeze (coiling pattern)
        
        Args:
            df: DataFrame with 'close' column
            period: Period for Bollinger Bands calculation (default 20)
        
        Returns:
            Series of BB Width percentages
        
        Example:
            >>> bb_width = calculate_bb_width(df)
            >>> current_width = bb_width.iloc[-1]
            >>> min_6month = bb_width.iloc[-126:].min()
            >>> print(f"Current width: {current_width:.2f}%, 6mo low: {min_6month:.2f}%")
            
            Low BB Width = Low volatility = Potential VCP setup
            FIX: Added min_periods=1.
        """
        # Calculate standard Bollinger Bands
        close = df['close']
        middle = close.rolling(window=period, min_periods=1).mean()
        std_dev = close.rolling(window=period, min_periods=1).std()
        upper = middle + (2 * std_dev)
        lower = middle - (2 * std_dev)
        bb_width = ((upper - lower) / middle.replace(0, np.nan)) * 100
        return bb_width.fillna(0)
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes. It oscillates between 0 and 100.
        
        Formula:
        1. Calculate price changes (deltas)
        2. Separate gains and losses
        3. Calculate average gain and average loss
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))
        
        Interpretation:
        - RSI > 70: Overbought (potential reversal down)
        - RSI < 30: Oversold (potential reversal up)
        - For VCP: We want RSI staying above 40, showing strength
        
        Args:
            df: DataFrame with 'close' column
            period: Number of periods for RSI calculation (default 14)
        
        Returns:
            Series of RSI values (0-100)
        
        Example:
            >>> rsi = calculate_rsi(df)
            >>> current_rsi = rsi.iloc[-1]
            >>> if current_rsi > 40:
            >>>     print("Strong - RSI holding above 40")
            
            In VCP:
            - RSI should make higher lows during contractions
            - Should stay in 40-65 range (not oversold, not overbought)

        FIX: Added min_periods and fillna to handle edge cases.
        FIX: Use EMA-based smoothing (Wilder's method) for accuracy.
        """
        close = df['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # FIX: Use Wilder's smoothing (ewm) instead of simple rolling mean
        # This matches most charting platforms including Chartink
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Fill NaN with neutral 50
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, periods: List[int]) -> Dict[int, pd.Series]:
        """
        Calculate multiple Simple Moving Averages (SMA).
        
        A Simple Moving Average is the arithmetic mean of prices over
        a specified number of periods. It's a trend-following indicator.
        
        Moving averages smooth out price action and help identify trends:
        - Price above MA = Potential uptrend
        - Price below MA = Potential downtrend
        - MA slope indicates trend strength
        
        For VCP:
        - Shorter MAs should be above longer MAs (bullish alignment)
        - Price should stay above key MAs during contractions
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods to calculate (e.g., [21, 50, 200])
        
        Returns:
            Dictionary mapping period to Series of MA values
            Example: {21: Series(...), 50: Series(...), 200: Series(...)}
        
        Example:
            >>> mas = calculate_moving_averages(df, [21, 50, 200])
            >>> ma21 = mas[21].iloc[-1]
            >>> ma50 = mas[50].iloc[-1]
            >>> ma200 = mas[200].iloc[-1]
            >>> if ma21 > ma50 > ma200:
            >>>     print("Bullish MA alignment!")

        FIX: Added min_periods=1 so early rows aren't all NaN.
        """
        mas = {}
        
        # Calculate SMA for each period requested
        mas = {}
        for period in periods:
            mas[period] = df['close'].rolling(window=period, min_periods=1).mean()
        return mas
        
        return mas
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV is a cumulative volume indicator that adds volume on up days
        and subtracts volume on down days. It's used to confirm price trends
        by showing whether volume is flowing into or out of a security.
        
        Logic:
        - If close > previous close: OBV += volume (accumulation)
        - If close < previous close: OBV -= volume (distribution)
        - If close = previous close: OBV unchanged
        
        For VCP detection:
        - OBV should be rising or flat during contractions (accumulation)
        - Declining OBV during consolidation = distribution (bearish)
        - OBV breakout before price = very bullish signal
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
        
        Returns:
            Series of cumulative OBV values
        
        Example:
            >>> obv = calculate_obv(df)
            >>> obv_trend = obv.iloc[-20:].diff().mean()
            >>> if obv_trend > 0:
            >>>     print("OBV trending up - accumulation")
            >>> else:
            >>>     print("OBV trending down - distribution")
        """
        # Calculate the sign of price changes
        # +1 if price went up, -1 if down, 0 if unchanged
        # diff() calculates close - previous close
        # sign() converts to -1, 0, or +1
        direction = np.sign(df['close'].diff()).fillna(0)
        
        # Multiply direction by volume, then cumsum to get running total
        # fillna(0) handles the first row which has no previous close
        obv = (direction * df['volume']).cumsum()
        
        return obv
    
    @staticmethod
    def calculate_relative_strength(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Relative Strength vs. market index (usually SPY).
        
        Relative Strength (RS) shows how a stock performs compared to
        the overall market. It's NOT the same as RSI.
        
        Formula: (Stock Price / Market Price) × 100
        
        Rising RS line = Stock outperforming market
        Falling RS line = Stock underperforming market
        
        For VCP:
        - RS line should be at or near new highs
        - RS should stay strong during market pullbacks
        - IBD-style RS rating > 80 is ideal
        
        Args:
            stock_df: DataFrame with stock 'close' prices
            market_df: DataFrame with market index 'close' prices (e.g., SPY)
        
        Returns:
            Series of RS ratio values
        
        Example:
            >>> spy_data = load_stock_data('SPY', '1y')
            >>> aapl_data = load_stock_data('AAPL', '1y')
            >>> rs_line = calculate_relative_strength(aapl_data, spy_data)
            >>> 
            >>> # Check if RS at new highs
            >>> rs_high = rs_line.max()
            >>> current_rs = rs_line.iloc[-1]
            >>> pct_from_high = ((rs_high - current_rs) / rs_high) * 100
            >>> print(f"RS {pct_from_high:.1f}% from highs")
        """
        # Extract close prices from both dataframes
        stock_close = stock_df['close']
        market_close = market_df['close']
        
        # Calculate the ratio and multiply by 100 for easier reading
        # Higher values = stock outperforming
        # Lower values = stock underperforming
        rs_line = (stock_close / market_close) * 100
        
        return rs_line

# ============================================================================
# PATTERN DETECTION
# ============================================================================

class PivotDetector:
    """
    Detect swing highs and swing lows in price data.
    
    Pivots (also called swing points) are local maxima and minima in price.
    They're essential for identifying the structure of consolidation patterns.
    
    A swing high requires:
    - The high is higher than N bars to the left AND right
    
    A swing low requires:
    - The low is lower than N bars to the left AND right
    
    These pivots define the boundaries of each contraction phase in a VCP.

    FIXES:
    - Use configurable lookback (not hardcoded 3)
    - Handle flat tops/bottoms with <= instead of <
    - Return both positional index AND date for reliability
    """
    
    @staticmethod
    def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """
        A swing high is a local maximum - a bar whose high is greater than
        the highs of N bars on both sides. This confirms it's a genuine
        peak, not just noise.

        FIX: Changed strict < to <= for the comparison to handle flat tops.
        FIX: Increased default lookback from 3 to 5 to reduce noise.
        
        Algorithm:
        1. For each bar (excluding first and last N bars)
        2. Check if its high is greater than all bars within lookback range
        3. Must be higher than bars both before AND after it
        4. If yes, it's a swing high
        
        Args:
            df: DataFrame with 'high' column
            lookback: Number of bars on each side to confirm pivot (default 5)
                    Higher values = more significant pivots, fewer detected
                    Lower values = more pivots detected, potentially more noise
        
        Returns:
            List of integer indices where swing highs occur
        
        Example:
            >>> swing_highs = find_swing_highs(df, lookback=5)
            >>> print(f"Found {len(swing_highs)} swing highs")
            >>> for idx in swing_highs:
            >>>     print(f"Swing high at index {idx}: ${df['high'].iloc[idx]:.2f}")
            
            For VCP: Swing highs mark the start of each contraction
        """
        highs = []  # Will store indices of swing highs
        high_prices = df['high'].values  # Convert to numpy array for speed
        n = len(high_prices)

        # Loop through bars, excluding first and last 'lookback' bars
        # (we need bars on both sides to confirm a pivot)
        for i in range(lookback, n - lookback):
            current = high_prices[i]
            is_high = True  # Assume it's a high until proven otherwise
            
            # Check if current bar is higher than all bars within lookback range
            for j in range(1, lookback + 1):
                # FIX: Use >= to handle flat tops (equal highs)
                if current <= high_prices[i - j] or current <= high_prices[i + j]:
                    is_high = False  # Found a higher bar, so not a swing high
                    break
            
            # If it passed all checks, it's a swing high
            if is_high:
                highs.append(i)
    
        return highs
    
    @staticmethod
    def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """
        Find swing low pivots in price data.
        
        A swing low is a local minimum - a bar whose low is less than
        the lows of N bars on both sides.
        
        FIX: Changed strict > to >= for the comparison to handle flat bottoms.
        FIX: Increased default lookback from 3 to 5 to reduce noise.

        Algorithm:
        1. For each bar (excluding first and last N bars)
        2. Check if its low is less than all bars within lookback range
        3. Must be lower than bars both before AND after it
        4. If yes, it's a swing low
        
        Args:
            df: DataFrame with 'low' column
            lookback: Number of bars on each side to confirm pivot (default 5)
        
        Returns:
            List of integer indices where swing lows occur
        
        Example:
            >>> swing_lows = find_swing_lows(df, lookback=5)
            >>> print(f"Found {len(swing_lows)} swing lows")
            >>> for idx in swing_lows:
            >>>     print(f"Swing low at index {idx}: ${df['low'].iloc[idx]:.2f}")
            
            For VCP: Swing lows mark the end (bottom) of each contraction
        """
        lows = []  # Will store indices of swing lows
        low_prices = df['low'].values  # Convert to numpy array for speed
        n = len(low_prices)

        # Loop through bars, excluding edges
        for i in range(lookback, n - lookback):
            current = low_prices[i]
            is_low = True  # Assume it's a low until proven otherwise
            
            # Check if current bar is lower than all bars within lookback range
            for j in range(1, lookback + 1):
                # FIX: Use <= to handle flat bottoms
                if current > low_prices[i - j] or current > low_prices[i + j]:
                    is_low = False  # Found a lower bar, so not a swing low
                    break
            
            # If it passed all checks, it's a swing low
            if is_low:
                lows.append(i)
        
        return lows


class ContractionDetector:
    """
    Detect consolidation/contraction phases
    This class is responsible for:
    1. Identifying individual contraction phases
    2. Validating that contractions follow the VCP pattern rules
    3. Checking for progressive tightening (each contraction shallower than last)

    A contraction is defined as:
    - Starting at a swing high
    - Pulling back to a swing low
    - Followed by recovery toward the next swing high

    For a valid VCP, contractions must show:
    - Progressive depth reduction (each pullback shallower)
    - Progressive ATR reduction (volatility contracting)
    - Appropriate depth ranges for each phase

    MAJOR FIXES:
    - Use positional indexing throughout (no more DatetimeIndex issues)
    - Limit analysis to recent lookback window
    - Relaxed validation rules
    - Minimum contraction duration check
    - ATR tolerance for near-monotonic decline
    """
    
    @staticmethod
    def identify_contractions(df: pd.DataFrame, config: VCPConfig) -> List[Contraction]:
        """
        Identify all contraction phases in the price data.
        
        This method finds contraction phases by:
        1. Finding all swing highs and lows
        2. Pairing consecutive swing highs
        3. Finding the lowest low between each pair
        4. Creating a Contraction object for each valid pullback
        
        Only includes contractions with reasonable depth (2-30%) to
        filter out noise and overly-deep corrections.
        
        Args:
            df: DataFrame with price data and technical indicators
                Must include: high, low, close, volume, atr, rsi
            config: VCPConfig with pattern detection parameters
        
        Returns:
            List of Contraction objects, sorted chronologically
        
        Process:
            1. Find swing highs: [idx10, idx30, idx50, idx70]
            2. For each pair of highs (10-30, 30-50, 50-70):
            - Find the lowest low between them
            - Calculate depth, duration, metrics
            - Create Contraction object if valid
        
        Example:
            >>> contractions = identify_contractions(df, config)
            >>> for i, c in enumerate(contractions):
            >>>     print(f"Contraction {i+1}:")
            >>>     print(f"  Depth: {c.depth_percent:.2f}%")
            >>>     print(f"  Duration: {c.duration_bars} bars")
            >>>     print(f"  ATR: ${c.avg_atr:.2f}")

        FIX 2: Replaced all label-based indexing (df.loc[]) with
                positional indexing (df.iloc[]) to avoid DatetimeIndex errors.
        
        FIX 3: Added lookback window - only analyze last N bars
                to find RECENT patterns (not patterns from 2 years ago).
        
        FIX: Added minimum duration filter to remove noise contractions.

        """
        # ----------------------------------------------------------------
        # FIX 3: Focus only on the recent lookback window
        # This ensures we find CURRENT patterns, not historical ones
        # ----------------------------------------------------------------
        if config.timeframe == Timeframe.DAILY:
            lookback = config.lookback_days
        else:
            lookback = config.lookback_weeks

        # Slice to recent window using positional indexing
        recent_df = df.iloc[-lookback:].copy()
        recent_df = recent_df.reset_index(drop=False)  # Keep date as column

        # ----------------------------------------------------------------
        # FIX 8: Use config pivot_lookback (was hardcoded as 3)
        # ----------------------------------------------------------------
        pivot_lookback = config.pivot_lookback

        swing_highs = PivotDetector.find_swing_highs(recent_df, pivot_lookback)
        swing_lows  = PivotDetector.find_swing_lows(recent_df, pivot_lookback)

        contractions = []

        if len(swing_highs) < 2:
            return contractions

        # Get numpy arrays for fast access
        highs_arr  = recent_df['high'].values
        lows_arr   = recent_df['low'].values
        volume_arr = recent_df['volume'].values
        atr_arr    = recent_df['atr'].values
        rsi_arr    = recent_df['rsi'].values

        # ----------------------------------------------------------------
        # FIX 2: Use purely positional indexing
        # ----------------------------------------------------------------
        for i in range(len(swing_highs) - 1):
            sh_pos      = swing_highs[i]        # Positional index of swing high
            next_sh_pos = swing_highs[i + 1]    # Positional index of next swing high

            high_price = highs_arr[sh_pos]

            # Find lowest low between the two swing highs
            segment_lows = lows_arr[sh_pos: next_sh_pos + 1]
            local_min_offset = int(np.argmin(segment_lows))
            sl_pos = sh_pos + local_min_offset   # Positional index of swing low
            low_price = lows_arr[sl_pos]

            # Calculate depth
            depth_percent = ((high_price - low_price) / high_price) * 100

            # FIX: Filter by reasonable depth
            if depth_percent < 2.0 or depth_percent > 40.0:
                continue

            # FIX: Filter by minimum contraction duration
            duration = sl_pos - sh_pos
            if duration < config.min_contraction_duration:
                continue

            # Calculate metrics over the contraction segment
            seg_volume = volume_arr[sh_pos: sl_pos + 1]
            seg_atr    = atr_arr[sh_pos: sl_pos + 1]
            seg_rsi    = rsi_arr[sh_pos: sl_pos + 1]

            avg_volume = float(np.mean(seg_volume)) if len(seg_volume) > 0 else 0.0
            avg_atr    = float(np.mean(seg_atr))    if len(seg_atr)    > 0 else 0.0
            rsi_low    = float(np.min(seg_rsi))     if len(seg_rsi)    > 0 else 50.0

            # FIX: Store positional index AND actual date
            start_date = recent_df.index[sh_pos] if 'index' not in recent_df.columns \
                         else recent_df.iloc[sh_pos].get('index', sh_pos)
            end_date   = recent_df.index[sl_pos] if 'index' not in recent_df.columns \
                         else recent_df.iloc[sl_pos].get('index', sl_pos)

            contraction = Contraction(
                start_pos     = sh_pos,
                end_pos       = sl_pos,
                start_date    = start_date,
                end_date      = end_date,
                high_price    = high_price,
                low_price     = low_price,
                depth_percent = depth_percent,
                duration_bars = duration,
                avg_volume    = avg_volume,
                avg_atr       = avg_atr,
                rsi_low       = rsi_low
            )
            contractions.append(contraction)

        return contractions

    
    @staticmethod
    def validate_contraction_sequence(contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, List[str]]:
        """
        Validate that contractions follow the VCP pattern rules.
        
        A valid VCP requires:
        1. Minimum number of contractions (usually 3)
        2. Progressive depth reduction (each contraction shallower than previous)
        3. Each contraction depth within valid ranges
        4. Progressive ATR reduction (volatility contracting)
        
        This is the core validation that determines if a pattern is truly a VCP.
        
        Args:
            contractions: List of Contraction objects to validate
            config: VCPConfig with validation thresholds
        
        Returns:
            Tuple of (is_valid, messages)
            - is_valid: Boolean indicating if sequence is valid VCP
            - messages: List of strings explaining validation results
        
        Validation Logic:
            PASS if:
            - Has 3+ contractions
            - Each contraction is shallower than the previous
            - Depths are within expected ranges
            - ATR is declining
            
            FAIL if:
            - Too few contractions
            - Contractions getting deeper (not tightening)
            - Depths outside acceptable ranges
        
        Example:
            >>> is_valid, msgs = validate_contraction_sequence(contractions, config)
            >>> if is_valid:
            >>>     print("Valid VCP pattern!")
            >>> else:
            >>>     print("Not a VCP:")
            >>>     for msg in msgs:
            >>>         print(f"  - {msg}")

        FIX 4a: Only validate the LAST min_contractions (3) contractions,
                not all contractions. Older ones may be from different cycles.
        
        FIX 4b: Allow ATR tolerance (10%) instead of strict monotonic decline.
                Real-world patterns aren't perfectly monotonic.
        
        FIX 4c: Depth violations are warnings, not hard failures.
                Allow 1 depth violation before failing.

        """
        messages    = []
        fail_count  = 0

        if len(contractions) < config.min_contractions:
            messages.append(
                f"FAIL: Only {len(contractions)} contractions found "
                f"(need {config.min_contractions})"
            )
            return False, messages

        # FIX 4a: Use only the most recent N contractions
        recent = contractions[-config.min_contractions:]
        depths = [c.depth_percent for c in recent]

        # Rule 1: Progressive depth reduction
        for i in range(len(depths) - 1):
            if depths[i + 1] >= depths[i]:
                fail_count += 1
                messages.append(
                    f"WARN: Contraction {i+1}→{i+2} depth not reducing: "
                    f"{depths[i]:.1f}% → {depths[i+1]:.1f}%"
                )

        # If more than 1 non-reducing pair, fail
        if fail_count > 1:
            messages.append("FAIL: Multiple non-reducing contractions")
            return False, messages

        # Rule 2: Depth range validation (using relaxed thresholds)
        depth_ranges = [
            (config.contraction_1_min, config.contraction_1_max, "1st"),
            (config.contraction_2_min, config.contraction_2_max, "2nd"),
            (config.contraction_3_min, config.contraction_3_max, "3rd"),
        ]
        depth_violations = 0
        for idx, (mn, mx, label) in enumerate(depth_ranges):
            if idx < len(recent):
                d = recent[idx].depth_percent
                if not (mn <= d <= mx):
                    depth_violations += 1
                    messages.append(
                        f"WARN: {label} contraction depth {d:.1f}% "
                        f"outside [{mn}-{mx}]"
                    )

        # FIX 4c: Allow 1 depth violation
        if depth_violations > 1:
            messages.append("FAIL: Too many depth range violations")
            return False, messages

        # Rule 3: ATR decline with tolerance
        # FIX 4b: Allow ATR to occasionally increase slightly
        atr_vals = [c.avg_atr for c in recent]
        atr_violations = 0
        for i in range(len(atr_vals) - 1):
            # Allow 10% tolerance: atr[i+1] can be up to 110% of atr[i]
            if atr_vals[i + 1] > atr_vals[i] * (1 + config.atr_tolerance):
                atr_violations += 1
                messages.append(
                    f"WARN: ATR increased from contraction {i+1} to {i+2}: "
                    f"{atr_vals[i]:.2f} → {atr_vals[i+1]:.2f}"
                )

        if atr_violations > 1:
            messages.append("FAIL: ATR not declining (too many violations)")
            return False, messages

        messages.append(
            f"PASS: Valid sequence with {len(recent)} contractions, "
            f"depths: {[f'{d:.1f}%' for d in depths]}"
        )
        return True, messages

# ============================================================================
# SIGNAL VALIDATORS
# ============================================================================

class SignalValidator:
    """
    Validates individual technical signals (volume dry-up, ATR contraction, RSI strength, etc.) to confirm VCP validity.
    Each method checks one aspect and returns (is_valid, message); signals include volume, volatility, momentum, and price position.
    FIXES:
        - Volume comparison uses pre-consolidation baseline
        - Tight consolidation checks consecutive days properly
        - RSI range widened for Indian markets
    """
    
    @staticmethod
    def check_volume_dryup(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Checks if volume has dried up in later contractions (below threshold of average).
        Compares last contraction volume to overall average; low volume indicates exhausted sellers.
        FIX 5: Compare last contraction volume to the period BEFORE consolidation.
        
        Problem: Using recent 50-day average includes the consolidation itself,
        which has lower volume, making the ratio look artificially fine OR skewed.
        
        Fix: Use the 50-100 day window (before consolidation) as baseline.
        This gives the "normal active trading" volume as the reference.

        """
        if not contractions:
            return False, "FAIL: No contractions to analyze"

        # FIX: Use pre-consolidation volume as baseline
        # Take 50-day window that ends ~100 days ago (before consolidation)
        vol_period = config.daily_volume_period  # 50

        if len(df) > (vol_period * 2):
            # Baseline: 50 days ending 50 days before the end
            # This captures the "normal" volume before the consolidation
            baseline_vol = df['volume'].iloc[-(vol_period * 2): -vol_period].mean()
        else:
            # Fallback to full available history
            baseline_vol = df['volume'].iloc[:].mean()

        if baseline_vol == 0:
            return False, "FAIL: Baseline volume is zero"

        # Get last contraction's average volume
        last_c     = contractions[-1]
        last_vol   = last_c.avg_volume
        vol_ratio  = last_vol / baseline_vol

        is_valid = vol_ratio <= config.volume_dryup_threshold
        mark = '✓' if is_valid else '✗'
        msg = (
            f"Volume dry-up: {vol_ratio:.2%} of pre-consolidation avg "
            f"(baseline={baseline_vol:,.0f}) "
            f"({mark} target: <{config.volume_dryup_threshold:.0%})"
        )
        return is_valid, msg
    
    @staticmethod
    def check_atr_contraction(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Checks if ATR has contracted sufficiently (below threshold ratio from initial).
        Measures volatility reduction; significant decline indicates coiling pattern.
        Check ATR contraction with tolerance.
        Uses first vs last contraction ATR ratio.
        """
        if not contractions:
            return False, "FAIL: No contractions"

        first_atr = contractions[0].avg_atr
        last_atr  = contractions[-1].avg_atr

        if first_atr == 0:
            return False, "FAIL: First ATR is zero"

        atr_ratio = last_atr / first_atr
        is_valid  = atr_ratio <= config.atr_contraction_threshold

        mark = '✓' if is_valid else '✗'
        msg = (
            f"ATR contraction: {atr_ratio:.2%} of initial ATR "
            f"({first_atr:.2f} → {last_atr:.2f}) "
            f"({mark} target: <{config.atr_contraction_threshold:.0%})"
        )
        return is_valid, msg
    
    @staticmethod
    def check_rsi_strength(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Checks if RSI shows strength (above floor in later contractions, in consolidation range).
        Ensures RSI makes higher lows and stays elevated; shows consolidation from strength.
        FIX 7: Widened RSI ranges for Indian equity market characteristics.
        
        Indian stocks (especially mid/small cap) often:
        - Have RSI dipping to 35-40 during healthy consolidations
        - Consolidate with RSI in 45-70 range (slightly higher than US stocks)
        
        Additional fix: Check RSI trend (higher lows) separately from range.
        """
        if not contractions:
            return False, "FAIL: No contractions"

        rsi_lows = [c.rsi_low for c in contractions]

        # Check floor violations in RECENT contractions only
        recent_contrs = contractions[-2:] if len(contractions) >= 2 else contractions
        violations = sum(1 for c in recent_contrs if c.rsi_low < config.rsi_floor)
        floor_ok = (violations == 0)

        # Check current RSI is in consolidation range (widened)
        current_rsi = df['rsi'].iloc[-1]
        in_range = config.rsi_consolidation_min <= current_rsi <= config.rsi_consolidation_max

        # FIX: Check RSI higher lows trend (bonus signal)
        rsi_higher_lows = all(
            rsi_lows[i] <= rsi_lows[i + 1]
            for i in range(len(rsi_lows) - 1)
        )

        is_valid = floor_ok and in_range
        mark = '✓' if is_valid else '✗'
        higher_lows_str = "↑ higher lows" if rsi_higher_lows else "↓ not higher lows"
        msg = (
            f"RSI strength: current={current_rsi:.1f}, "
            f"lows={[f'{x:.1f}' for x in rsi_lows]}, "
            f"{higher_lows_str} ({mark})"
        )
        return is_valid, msg
    
    @staticmethod
    def check_ma_alignment(df: pd.DataFrame, config: VCPConfig) -> Tuple[bool, str]:
        """
        Checks if moving averages are in bullish alignment (shorter above longer).
        Daily: 21>50>200; Weekly: 10>40; confirms uptrend for VCP formation.
        Check MA alignment.
        FIX: Added null-check for MA columns before accessing.
        FIX: Use .iloc[-1] with explicit null check.
        """
        if not config.ma_alignment_required:
            return True, "MA alignment not required (skipped)"

        tf = config.timeframe

        if tf == Timeframe.DAILY:
            col_s = f'ma_{config.daily_ma_short}'
            col_m = f'ma_{config.daily_ma_medium}'
            col_l = f'ma_{config.daily_ma_long}'

            # FIX: Check columns exist before accessing
            missing = [c for c in [col_s, col_m, col_l] if c not in df.columns]
            if missing:
                return False, f"FAIL: Missing MA columns: {missing}"

            short_ma  = df[col_s].iloc[-1]
            medium_ma = df[col_m].iloc[-1]
            long_ma   = df[col_l].iloc[-1]

            # FIX: Check for NaN values
            if any(pd.isna([short_ma, medium_ma, long_ma])):
                return False, "FAIL: MA values contain NaN"

            is_valid = short_ma > medium_ma > long_ma
            mark = '✓' if is_valid else '✗'
            msg = (
                f"MA alignment: {col_s}={short_ma:.2f} > "
                f"{col_m}={medium_ma:.2f} > {col_l}={long_ma:.2f} ({mark})"
            )
        else:
            col_s = f'ma_{config.weekly_ma_short}'
            col_l = f'ma_{config.weekly_ma_long}'

            missing = [c for c in [col_s, col_l] if c not in df.columns]
            if missing:
                return False, f"FAIL: Missing MA columns: {missing}"

            short_ma = df[col_s].iloc[-1]
            long_ma  = df[col_l].iloc[-1]

            if any(pd.isna([short_ma, long_ma])):
                return False, "FAIL: MA values contain NaN"

            is_valid = short_ma > long_ma
            mark = '✓' if is_valid else '✗'
            msg = f"MA alignment: {col_s}={short_ma:.2f} > {col_l}={long_ma:.2f} ({mark})"

        return is_valid, msg
    
    @staticmethod
    def check_price_near_highs(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Checks if current price is near 52-week highs (within 10%).
        Shows stock is leading; less resistance for breakout.
        FIX: Use available data if less than 252 bars.
        """
        lookback      = min(252, len(df))
        high_52w      = df['high'].iloc[-lookback:].max()
        current_price = df['close'].iloc[-1]

        if high_52w == 0:
            return False, "FAIL: 52-week high is zero"

        dist_pct = ((high_52w - current_price) / high_52w) * 100
        is_valid  = dist_pct <= 15.0    # FIX: Relaxed from 10% to 15%

        mark = '✓' if is_valid else '✗'
        msg  = (
            f"Price near highs: {dist_pct:.1f}% below 52w high "
            f"(high={high_52w:.2f}, current={current_price:.2f}) "
            f"({mark} target: <15%)"
        )
        return is_valid, msg
    
    
    @staticmethod
    def check_tight_consolidation(df: pd.DataFrame, config: VCPConfig) -> Tuple[bool, str]:
        """
        Checks for tight price action in recent bars (small ranges, min consecutive tight days).
        Counts days with range < threshold; tightness indicates readiness for breakout.
        FIX 6: Check for CONSECUTIVE tight days, not just total count.
        
        Original code counted all tight days in last 10 bars.
        Fix counts the longest run of consecutive tight days.
        
        Also FIX: Use last 15 bars (was 10) to give more opportunity to find
        the consecutive tight days.
        """
        lookback   = min(15, len(df))   # FIX: Increased from 10 to 15
        recent_df  = df.iloc[-lookback:]

        # Daily range as percentage
        daily_ranges = ((recent_df['high'] - recent_df['low']) /
                        recent_df['close'].replace(0, np.nan)) * 100

        tight_mask = (daily_ranges < (config.tight_range_threshold * 100)).values

        # FIX 6: Find longest CONSECUTIVE run of tight days
        max_consec  = 0
        curr_consec = 0
        for t in tight_mask:
            if t:
                curr_consec += 1
                max_consec   = max(max_consec, curr_consec)
            else:
                curr_consec  = 0

        is_valid = max_consec >= config.min_tight_days
        avg_range = daily_ranges.mean()

        mark = '✓' if is_valid else '✗'
        msg = (
            f"Tight consolidation: {max_consec} consecutive tight days "
            f"(avg range={avg_range:.2f}%, threshold={config.tight_range_threshold*100:.1f}%) "
            f"({mark} target: ≥{config.min_tight_days})"
        )
        return is_valid, msg

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def load_all_stock_data(start_date: str) -> pd.DataFrame:
    """
    Load historical stock data for all symbols from SQLite database beyond a start date.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with multi-index (symbol, date) and columns: open, high, low, close, volume
    """
    try:
        # Connect to SQLite database
        db_path = cfg_vars.db_dir + cfg_vars.db_name
        conn = sqlite3.connect(db_path)
        
        # Query all data for all symbols from start_date
        query = """
            SELECT tckr_symbol AS symbol, trade_dt AS date, open_price AS open, high_price AS high, low_price AS low, closing_price AS close, total_trading_volume AS volume
            FROM historical_stocks
            WHERE trade_dt >= ?
            ORDER BY tckr_symbol, trade_dt         
        """
        df = pd.read_sql_query(query, conn, params=(start_date,))
        
        # Close the connection
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set multi-index: symbol and date
        df.set_index(['symbol', 'date'], inplace=True)
        
        # Ensure columns are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading bulk stock data: {str(e)}")
        return pd.DataFrame()

def process_multiple_stocks(symbols: List[str], config: VCPConfig = None, max_workers: int = 4) -> Dict[str, VCPResult]:
    """
    Process multiple stocks concurrently for VCP patterns.
    
    This function loads all data once from the database, then filters for each symbol
    in memory for faster processing.
    
    Args:
        symbols: List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        config: VCPConfig object with detection parameters (default: VCPConfig())
        max_workers: Maximum number of concurrent threads (default: 4)
    
    Returns:
        Dictionary mapping symbol to VCPResult object
        Failed analyses will have VCPResult with is_vcp=False and error in messages
    
    Example:
        >>> symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        >>> config = VCPConfig(timeframe=Timeframe.DAILY)
        >>> results = process_multiple_stocks(symbols, config)
        >>> for symbol, result in results.items():
        >>>     print(f"{symbol}: VCP={result.is_vcp}, Score={result.confidence_score:.1f}")
    """
    if config is None:
        config = VCPConfig()
    
    # Load all stock data once (2 years back to cover all periods)
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    bulk_df = load_all_stock_data(start_date)
    
    if bulk_df.empty:
        logger.error("No bulk data loaded from database")
        return {symbol: VCPResult(
            is_vcp=False, confidence_score=0.0, contractions=[],
            current_stage="no_data", breakout_price=None, stop_loss=None,
            signals={}, metrics={}, messages=["No data available"]
        ) for symbol in symbols}
    
    def analyze_single_stock(symbol: str) -> Tuple[str, VCPResult]:
        """Helper function to analyze one stock (used by ThreadPoolExecutor)"""
        try:
            # Filter data for this symbol
            if symbol not in bulk_df.index.get_level_values(0):
                return symbol, VCPResult(
                    is_vcp=False, confidence_score=0.0, contractions=[],
                    current_stage="no_data", breakout_price=None, stop_loss=None,
                    signals={}, metrics={}, messages=[f"No data available for {symbol}"]
                )
            
            df = bulk_df.loc[symbol].copy()
            if df.empty:
                return symbol, VCPResult(
                    is_vcp=False, confidence_score=0.0, contractions=[],
                    current_stage="no_data", breakout_price=None, stop_loss=None,
                    signals={}, metrics={}, messages=[f"No data available for {symbol}"]
                )
            
            # Calculate required technical indicators
            df['atr'] = TechnicalIndicators.calculate_atr(df, 
                config.daily_atr_period if config.timeframe == Timeframe.DAILY else config.weekly_atr_period)
            df['rsi'] = TechnicalIndicators.calculate_rsi(df,
                config.daily_rsi_period if config.timeframe == Timeframe.DAILY else config.weekly_rsi_period)
            
            # Calculate moving averages
            ma_periods = [config.daily_ma_short, config.daily_ma_medium, config.daily_ma_long] if config.timeframe == Timeframe.DAILY else [config.weekly_ma_short, config.weekly_ma_long]
            mas = TechnicalIndicators.calculate_moving_averages(df, ma_periods)
            for period, ma_series in mas.items():
                df[f'ma_{period}'] = ma_series
            
            # Find swing highs and lows
            swing_highs = PivotDetector.find_swing_highs(df, lookback=3)
            swing_lows = PivotDetector.find_swing_lows(df, lookback=3)

            # Detect contractions
            contractions = ContractionDetector.identify_contractions(df, config)
            is_valid, validation_msgs = ContractionDetector.validate_contraction_sequence(contractions, config)
            
            # Validate individual signals
            signals = {}
            signals['volume_dryup'], _ = SignalValidator.check_volume_dryup(df, contractions, config)
            signals['atr_contraction'], _ = SignalValidator.check_atr_contraction(df, contractions, config)
            signals['rsi_strength'], _ = SignalValidator.check_rsi_strength(df, contractions, config)
            signals['ma_alignment'], _ = SignalValidator.check_ma_alignment(df, config)
            signals['price_near_highs'], _ = SignalValidator.check_price_near_highs(df)
            signals['tight_consolidation'], _ = SignalValidator.check_tight_consolidation(df, config)
            
            # Calculate confidence score based on signal validation
            signal_count = len(signals)
            valid_signals = sum(signals.values())
            confidence_score = (valid_signals / signal_count) * 100 if signal_count > 0 else 0.0
            
            # Determine current stage
            if not is_valid:
                current_stage = "no_pattern"
            elif df['close'].iloc[-1] > df['high'].iloc[-len(contractions[-1].duration_bars):].max() if contractions else False:
                current_stage = "breaking_out"
            else:
                current_stage = "consolidating"
            
            """
            # Create result object
            result = VCPResult(
                is_vcp=is_valid,
                confidence_score=confidence_score,
                contractions=contractions,
                current_stage=current_stage,
                breakout_price=df['high'].iloc[-1] * 1.02 if is_valid else None,  # Simple breakout estimate
                stop_loss=df['low'].iloc[-1] * 0.95 if is_valid else None,       # Simple stop loss estimate
                signals=signals,
                metrics={
                    'current_price': df['close'].iloc[-1],
                    'current_rsi': df['rsi'].iloc[-1],
                    'current_atr': df['atr'].iloc[-1]
                },
                messages=validation_msgs
            )
            
            return symbol, result
            """
            # Determine current stage
            if not is_valid:
                current_stage = "no_pattern"
            elif df['close'].iloc[-1] > df['high'].iloc[-len(contractions[-1].duration_bars):].max() if contractions else False:
                current_stage = "breaking_out"
            else:
                current_stage = "consolidating"

            # Define breakout_price and stop_loss
            breakout_price = df['high'].iloc[-1] * 1.02 if is_valid else None
            stop_loss = df['low'].iloc[-1] * 0.95 if is_valid else None

            # Create output DataFrame with per-date data
            df_output = df.copy()
            df_output.reset_index(inplace=True)  # Make 'date' a column

            # Add swing high/low flags
            df_output['is_swing_high'] = False
            df_output['is_swing_low'] = False
            for idx in swing_highs:
                if idx < len(df_output):
                    df_output.loc[idx, 'is_swing_high'] = True
            for idx in swing_lows:
                if idx < len(df_output):
                    df_output.loc[idx, 'is_swing_low'] = True

            # Add VCP results (repeated for each date)
            df_output['is_vcp'] = is_valid
            df_output['confidence_score'] = confidence_score
            df_output['current_stage'] = current_stage
            df_output['breakout_price'] = breakout_price
            df_output['stop_loss'] = stop_loss

            # Add signals (repeated for each date)
            for sig_name, sig_value in signals.items():
                df_output[sig_name] = sig_value

            # Add metrics (repeated for each date)
            df_output['current_price'] = df['close'].iloc[-1]
            df_output['current_rsi'] = df['rsi'].iloc[-1]
            df_output['current_atr'] = df['atr'].iloc[-1]

            return symbol, df_output

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return symbol, pd.DataFrame()    
        
    # Process stocks concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(analyze_single_stock, symbol): symbol for symbol in symbols}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                sym, df_result = future.result()
                if not df_result.empty:
                    df_result['symbol'] = sym  # Add symbol column
                    results.append(df_result)
                logger.info(f"Completed analysis for {sym}")
            except Exception as exc:
                logger.error(f"Exception processing {symbol}: {exc}")

    # Concatenate all results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df

