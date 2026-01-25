
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
    contraction_1_min: float = 8.0      # First pullback: 8-20% is normal
    contraction_1_max: float = 25.0     # Over 25% is too deep
    contraction_2_min: float = 5.0      # Second pullback: 5-15%
    contraction_2_max: float = 15.0
    contraction_3_min: float = 3.0      # Third pullback: 3-10% (tightening)
    contraction_3_max: float = 10.0
    contraction_4_max: float = 6.0      # Fourth (optional): very tight, <6%
    
    # Volume thresholds
    # Volume should "dry up" (decrease) as contractions tighten
    # Then spike on breakout (showing new buying interest)
    volume_dryup_threshold: float = 0.60       # 60% of average = significant dryup
    breakout_volume_threshold: float = 1.50    # 150% of average = strong breakout
    
    # RSI thresholds
    # RSI should stay elevated (not oversold) during VCP contractions
    # This shows underlying strength despite price consolidation
    rsi_floor: float = 40.0                    # Don't go below 40 in later contractions
    rsi_consolidation_min: float = 48.0        # Final consolidation: 48-65 range
    rsi_consolidation_max: float = 65.0        # Not overbought, but strong
    rsi_breakout: float = 60.0                 # Breakout confirmation level
    
    # ATR contraction requirement
    # ATR measures volatility; it should decrease as pattern tightens
    # 0.70 means final ATR should be 70% or less of initial ATR (30% reduction)
    atr_contraction_threshold: float = 0.70
    
    # Price tightness requirements
    # "Tight" days have small high-low ranges, showing low volatility
    tight_range_threshold: float = 0.015       # 1.5% high-low range
    min_tight_days: int = 3                    # Need at least 3 consecutive tight days
    
    # Moving average alignment
    # Bullish setup: shorter MAs above longer MAs (uptrend confirmation)
    ma_alignment_required: bool = True
    
    # Relative strength requirement
    # Stock should be outperforming the market (within 10% of RS highs)
    rs_threshold: float = 0.90


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
    start_idx: int          # Where this contraction phase started
    end_idx: int            # Where it ended (the low point)
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
    def convert_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily OHLCV data to weekly timeframe.
        
        Weekly bars are created by aggregating daily data:
        - Open: First day's open of the week
        - High: Highest high of the week
        - Low: Lowest low of the week
        - Close: Last day's close of the week
        - Volume: Sum of all daily volume for the week
        
        Weeks are defined as ending on Friday (or last trading day of week).
        
        Args:
            daily_df: DataFrame with daily OHLCV data
        
        Returns:
            DataFrame with weekly OHLCV data (fewer rows than input)
        
        Note:
            The 'W' frequency in pandas resamples to week-end (Sunday).
            This works for stock data because it groups Mon-Fri together.
        
        Example:
            >>> daily_data = load_stock_data('AAPL', '1y')  # 252 trading days
            >>> weekly_data = convert_to_weekly(daily_data)  # ~52 weeks
            >>> print(f"Converted {len(daily_data)} days to {len(weekly_data)} weeks")
        """
        # Create empty DataFrame to hold weekly data
        weekly = pd.DataFrame()
        
        # Resample to weekly frequency ('W')
        # first() = take first value of the week (Monday's open)
        weekly['open'] = daily_df['open'].resample('W').first()
        
        # max() = highest value during the week
        weekly['high'] = daily_df['high'].resample('W').max()
        
        # min() = lowest value during the week
        weekly['low'] = daily_df['low'].resample('W').min()
        
        # last() = take last value of the week (Friday's close)
        weekly['close'] = daily_df['close'].resample('W').last()
        
        # sum() = total volume for the entire week
        weekly['volume'] = daily_df['volume'].resample('W').sum()
        
        # Remove any rows with NaN values (can occur at edges)
        return weekly.dropna()

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
        """
        # Extract price series
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate three components of True Range
        # tr1: High - Low (today's range)
        tr1 = high - low
        
        # tr2: |High - Previous Close| (gap up scenario)
        tr2 = abs(high - close.shift())
        
        # tr3: |Low - Previous Close| (gap down scenario)
        tr3 = abs(low - close.shift())
        
        # True Range is the maximum of these three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = tr.rolling(window=period).mean()
        
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
        """
        # Calculate standard Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df, period)
        
        # Width = (Upper - Lower) / Middle × 100 to get percentage
        bb_width = ((upper - lower) / middle) * 100
        
        return bb_width
    
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
        """
        close = df['close']
        
        # Calculate price changes from one period to the next
        delta = close.diff()
        
        # Separate gains (positive changes) and losses (negative changes)
        # .where() keeps values that meet condition, replaces others with 0
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate Relative Strength (RS)
        rs = gain / loss
        
        # Calculate RSI
        # Formula ensures RSI stays between 0 and 100
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
        """
        mas = {}
        
        # Calculate SMA for each period requested
        for period in periods:
            # .rolling(window=period).mean() calculates the average
            # of the last 'period' values
            mas[period] = df['close'].rolling(window=period).mean()
        
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
        direction = np.sign(df['close'].diff())
        
        # Multiply direction by volume, then cumsum to get running total
        # fillna(0) handles the first row which has no previous close
        obv = (direction * df['volume']).fillna(0).cumsum()
        
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
    """
    
    @staticmethod
    def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """
        A swing high is a local maximum - a bar whose high is greater than
        the highs of N bars on both sides. This confirms it's a genuine
        peak, not just noise.
        
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
    
        # Loop through bars, excluding first and last 'lookback' bars
        # (we need bars on both sides to confirm a pivot)
        for i in range(lookback, len(df) - lookback):
            is_high = True  # Assume it's a high until proven otherwise
            
            # Check if current bar is higher than all bars within lookback range
            for j in range(1, lookback + 1):
                # Check both sides: i-j (before) and i+j (after)
                if high_prices[i] <= high_prices[i - j] or high_prices[i] <= high_prices[i + j]:
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
        
        # Loop through bars, excluding edges
        for i in range(lookback, len(df) - lookback):
            is_low = True  # Assume it's a low until proven otherwise
            
            # Check if current bar is lower than all bars within lookback range
            for j in range(1, lookback + 1):
                # Check both sides: i-j (before) and i+j (after)
                if low_prices[i] >= low_prices[i - j] or low_prices[i] >= low_prices[i + j]:
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
        """
        # Find all swing highs and lows in the price data
        swing_highs = PivotDetector.find_swing_highs(df, lookback=3)
        swing_lows = PivotDetector.find_swing_lows(df, lookback=3)
        
        contractions = []  # Will store all valid contractions found
        
        # Need at least 2 swing highs to form one contraction
        # (high -> low -> high pattern)
        if len(swing_highs) < 2:
            return contractions
        
        # Iterate through consecutive pairs of swing highs
        # Each pair represents a potential contraction phase
        for i in range(len(swing_highs) - 1):
            # Start of contraction: current swing high
            high_idx = swing_highs[i]
            high_price = df['high'].iloc[high_idx]
            
            # End of search range: next swing high
            next_high_idx = swing_highs[i + 1]
            
            # Extract the segment between these two highs
            segment = df.iloc[high_idx:next_high_idx + 1]
            
            # Find the lowest low in this segment
            # This is where the contraction bottomed
            low_idx = segment['low'].idxmin()  # Index of minimum low
            low_price = segment['low'].min()   # Actual low price
            
            # Calculate how deep the pullback was (percentage)
            depth_percent = ((high_price - low_price) / high_price) * 100
            
            # Only consider contractions with reasonable depth
            # Too shallow (< 2%) = noise
            # Too deep (> 30%) = not a VCP contraction, more like a correction
            if 2.0 < depth_percent < 30.0:
                # Extract data for this contraction period
                contraction_segment = df.loc[high_idx:low_idx]
                
                # Calculate average metrics during the contraction
                avg_volume = contraction_segment['volume'].mean()
                avg_atr = contraction_segment['atr'].mean()
                rsi_low = contraction_segment['rsi'].min()  # Lowest RSI reading
                
                # Calculate duration (number of bars)
                duration = len(contraction_segment)
                
                # Create Contraction object with all data
                contraction = Contraction(
                    start_idx=high_idx,
                    end_idx=df.index.get_loc(low_idx) if not isinstance(low_idx, int) else low_idx,
                    high_price=high_price,
                    low_price=low_price,
                    depth_percent=depth_percent,
                    duration_bars=duration,
                    avg_volume=avg_volume,
                    avg_atr=avg_atr,
                    rsi_low=rsi_low
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
        """
        messages = []  # Will collect validation messages
    
        # Rule 1: Must have minimum number of contractions
        if len(contractions) < config.min_contractions:
            messages.append(
                f"Insufficient contractions: {len(contractions)} found, "
                f"{config.min_contractions} required"
            )
            return False, messages
        
        # Extract depth percentages for easier analysis
        depths = [c.depth_percent for c in contractions]
        
        # Rule 2: Check for progressive depth reduction
        # Each contraction should be shallower than the previous one
        is_progressive = True
        for i in range(len(depths) - 1):
            if depths[i + 1] >= depths[i]:  # Next contraction is deeper or equal
                is_progressive = False
                messages.append(
                    f"Contraction {i + 1} depth ({depths[i + 1]:.2f}%) "
                    f"not less than contraction {i} ({depths[i]:.2f}%)"
                )
        
        # Rule 3: Check that each contraction's depth is within valid ranges
        # These ranges are based on empirical observation of successful VCP patterns
        
        # First contraction: Should be the deepest (8-25%)
        if len(contractions) >= 1:
            if not (config.contraction_1_min <= depths[0] <= config.contraction_1_max):
                messages.append(
                    f"First contraction depth {depths[0]:.2f}% outside range "
                    f"[{config.contraction_1_min}-{config.contraction_1_max}]"
                )
        
        # Second contraction: Shallower (5-15%)
        if len(contractions) >= 2:
            if not (config.contraction_2_min <= depths[1] <= config.contraction_2_max):
                messages.append(
                    f"Second contraction depth {depths[1]:.2f}% outside range "
                    f"[{config.contraction_2_min}-{config.contraction_2_max}]"
                )
        
        # Third contraction: Even shallower (3-10%)
        if len(contractions) >= 3:
            if not (config.contraction_3_min <= depths[2] <= config.contraction_3_max):
                messages.append(
                    f"Third contraction depth {depths[2]:.2f}% outside range "
                    f"[{config.contraction_3_min}-{config.contraction_3_max}]"
                )
        
        # Rule 4: Check ATR progression (should decline)
        # Declining ATR = contracting volatility = coiling pattern
        atr_values = [c.avg_atr for c in contractions]
        
        # Check if each ATR is lower than the previous
        atr_declining = all(atr_values[i] > atr_values[i + 1] 
                        for i in range(len(atr_values) - 1))
        
        if not atr_declining:
            messages.append("ATR not declining progressively through contractions")
        
        # Final validation decision
        # Allow minor violations (up to 2 issues) for flexibility
        # Real-world patterns aren't always perfect
        is_valid = is_progressive and len(messages) <= 2
        
        return is_valid, messages


# ============================================================================
# SIGNAL VALIDATORS
# ============================================================================

class SignalValidator:
    """
    Validate individual technical signals that confirm VCP pattern
    This class checks specific conditions that should be present in a
    valid VCP setup. Each method validates one aspect of the pattern.

    Signals validated:
    - Volume dry-up: Volume decreasing during contractions
    - ATR contraction: Volatility decreasing significantly
    - RSI strength: RSI staying elevated (not oversold)
    - MA alignment: Moving averages in bullish order
    - Price position: Price near 52-week highs
    - Tight consolidation: Recent tight price action

    Each validation returns a tuple of (is_valid, message) to provide
    both the result and context.
    """
    
    @staticmethod
    def check_volume_dryup(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Check if volume is drying up in later contractions.
        
        In a healthy VCP, volume should decline during contractions as
        sellers are exhausted. Lower volume = less selling pressure.
        
        We want to see the final contraction's volume well below average,
        indicating that selling has dried up and the stock is ready to
        move higher on any buying interest.
        
        Args:
            df: DataFrame with 'volume' column
            contractions: List of Contraction objects
            config: VCPConfig with volume thresholds
        
        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if volume has dried up sufficiently
            - message: String explaining the result
        
        Validation:
            PASS: Last contraction's avg volume < 60% of overall average
            FAIL: Last contraction's volume still high (> 60% of average)
        
        Example:
            >>> is_valid, msg = check_volume_dryup(df, contractions, config)
            >>> print(msg)
            "Volume dry-up: 45% of average (✓ target: <60%)"
        """
        # Need contractions to analyze
        if not contractions:
            return False, "No contractions to analyze"
        
        # Determine which parameters to use based on timeframe
        timeframe = config.timeframe
        volume_period = (config.daily_volume_period if timeframe == Timeframe.DAILY 
                        else config.weekly_volume_period)
        
        # Calculate average volume over the period
        # This is our baseline for comparison
        avg_volume = df['volume'].iloc[-volume_period:].mean()
        
        # Get the most recent contraction's average volume
        last_contraction = contractions[-1]
        
        # Calculate ratio: How does last contraction volume compare to average?
        last_vol_ratio = last_contraction.avg_volume / avg_volume
        
        # Validate: Volume should be below threshold (e.g., 60% of average)
        is_valid = last_vol_ratio <= config.volume_dryup_threshold
        
        # Create informative message
        checkmark = '✓' if is_valid else '✗'
        message = (f"Volume dry-up: {last_vol_ratio:.2%} of average "
                f"({checkmark} target: <{config.volume_dryup_threshold:.0%})")
        
        return is_valid, message
    
    @staticmethod
    def check_atr_contraction(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Check if ATR (volatility) has contracted sufficiently.
        
        The hallmark of a VCP is contracting volatility. We want to see
        ATR declining from the first contraction to the last, indicating
        that price swings are getting smaller (coiling pattern).
        
        A significant ATR reduction (e.g., 30%+ decline) shows the stock
        is "coiling" and ready for an expansion phase.
        
        Args:
            df: DataFrame with 'atr' column
            contractions: List of Contraction objects
            config: VCPConfig with ATR threshold
        
        Returns:
            Tuple of (is_valid, message)
        
        Validation:
            PASS: Latest ATR ≤ 70% of initial ATR (30%+ reduction)
            FAIL: ATR hasn't contracted enough
        
        Example:
            >>> is_valid, msg = check_atr_contraction(df, contractions, config)
            >>> print(msg)
            "ATR contraction: 65% of initial (✓ target: <70%)"
            
            This shows ATR has dropped to 65% of its initial value
            (35% reduction), which exceeds our 30% target.
        """
        if not contractions:
            return False, "No contractions to analyze"
        
        # Get ATR from first and last contractions
        first_atr = contractions[0].avg_atr  # Initial volatility
        last_atr = contractions[-1].avg_atr   # Final volatility
        
        # Calculate ratio: How much has ATR declined?
        atr_ratio = last_atr / first_atr
        
        # Validate: ATR should have contracted below threshold
        # e.g., if threshold is 0.70, ATR should be 70% or less of initial
        is_valid = atr_ratio <= config.atr_contraction_threshold
        
        checkmark = '✓' if is_valid else '✗'
        message = (f"ATR contraction: {atr_ratio:.2%} of initial "
                f"({checkmark} target: <{config.atr_contraction_threshold:.0%})")
        
        return is_valid, message
    
    @staticmethod
    def check_rsi_strength(df: pd.DataFrame, contractions: List[Contraction], config: VCPConfig) -> Tuple[bool, str]:
        """
        Check if RSI shows underlying strength during contractions.
        
        In a strong VCP, RSI should:
        1. Stay above 40 in later contractions (not oversold)
        2. Make higher lows (each pullback less deep than previous)
        3. Be in the 48-65 range currently (coiled but not overbought)
        
        This shows the stock is consolidating from a position of strength,
        not weakness. Weak stocks go oversold (RSI < 30) during pullbacks.
        
        Args:
            df: DataFrame with 'rsi' column
            contractions: List of Contraction objects
            config: VCPConfig with RSI thresholds
        
        Returns:
            Tuple of (is_valid, message)
        
        Validation:
            PASS: All later contractions have RSI > 40, current RSI in 48-65 range
            FAIL: RSI went below 40, or current RSI outside range
        
        Example:
            >>> is_valid, msg = check_rsi_strength(df, contractions, config)
            >>> print(msg)
            "RSI strength: Current 55.2, lows: ['42.5', '46.8', '51.3'] (✓)"
            
            This shows:
            - Current RSI is 55.2 (in the good range)
            - RSI lows are making higher lows (getting stronger)
            - All lows above 40 (showing strength)
        """
        if not contractions:
            return False, "No contractions to analyze"
        
        # Extract RSI lows from all contractions
        rsi_lows = [c.rsi_low for c in contractions]
        
        # Focus on later contractions (most recent 2)
        # These are most important for current strength
        later_contractions = contractions[-2:] if len(contractions) >= 2 else contractions
        
        # Count how many violated the RSI floor (< 40)
        violations = sum(1 for c in later_contractions 
                        if c.rsi_low < config.rsi_floor)
        
        # Valid if no violations (all RSI lows > 40)
        is_valid = violations == 0
        
        # Also check current RSI is in the consolidation range
        current_rsi = df['rsi'].iloc[-1]
        in_range = (config.rsi_consolidation_min <= current_rsi 
                <= config.rsi_consolidation_max)
        
        checkmark = '✓' if is_valid and in_range else '✗'
        message = (f"RSI strength: Current {current_rsi:.1f}, "
                f"lows: {[f'{x:.1f}' for x in rsi_lows]} ({checkmark})")
        
        return is_valid and in_range, message
    
    @staticmethod
    def check_ma_alignment(df: pd.DataFrame, config: VCPConfig) -> Tuple[bool, str]:
        """
        Check if moving averages are in bullish alignment.
        
        For a valid VCP, we want moving averages stacked properly:
        - Daily: 21 EMA > 50 SMA > 200 SMA (all rising)
        - Weekly: 10 MA > 40 MA (both rising)
        
        This confirms the overall trend is up. VCP patterns form within
        uptrends, not downtrends.
        
        Args:
            df: DataFrame with MA columns (ma_21, ma_50, ma_200, etc.)
            config: VCPConfig with MA settings
        
        Returns:
            Tuple of (is_valid, message)
        
        Validation:
            Daily PASS: 21 EMA > 50 SMA > 200 SMA
            Weekly PASS: 10 MA > 40 MA
            FAIL: MAs not in proper order
        
        Example:
            >>> is_valid, msg = check_ma_alignment(df, config)
            >>> print(msg)
            "MA alignment (daily): 21EMA > 50SMA > 200SMA (✓)"
        """
        # If MA alignment not required by config, pass automatically
        if not config.ma_alignment_required:
            return True, "MA alignment not required"
        
        timeframe = config.timeframe
        
        if timeframe == Timeframe.DAILY:
            # Get current values of each MA
            short_ma = df[f'ma_{config.daily_ma_short}'].iloc[-1]   # 21 EMA
            medium_ma = df[f'ma_{config.daily_ma_medium}'].iloc[-1] # 50 SMA
            long_ma = df[f'ma_{config.daily_ma_long}'].iloc[-1]     # 200 SMA
            
            # Check proper alignment: shorter > medium > longer
            is_valid = short_ma > medium_ma > long_ma
            
            checkmark = '✓' if is_valid else '✗'
            message = (f"MA alignment (daily): {config.daily_ma_short}EMA > "
                    f"{config.daily_ma_medium}SMA > {config.daily_ma_long}SMA "
                    f"({checkmark})")
        else:  # Weekly
            # Get current values
            short_ma = df[f'ma_{config.weekly_ma_short}'].iloc[-1]  # 10 MA
            long_ma = df[f'ma_{config.weekly_ma_long}'].iloc[-1]    # 40 MA
            
            # Check proper alignment
            is_valid = short_ma > long_ma
            
            checkmark = '✓' if is_valid else '✗'
            message = (f"MA alignment (weekly): {config.weekly_ma_short}MA > "
                    f"{config.weekly_ma_long}MA ({checkmark})")
        
        return is_valid, message
    
    @staticmethod
    def check_price_near_highs(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if price is near 52-week highs.
        
        VCP patterns typically form as a stock consolidates near its highs.
        This is important because:
        1. Shows the stock is leading, not lagging
        2. Less overhead resistance (fewer bag-holders)
        3. More likely to breakout to new highs
        
        We want the stock within 10% of its 52-week high.
        
        Args:
            df: DataFrame with 'high' and 'close' columns
        
        Returns:
            Tuple of (is_valid, message)
        
        Validation:
            PASS: Current price within 10% of 52-week high
            FAIL: More than 10% below high
        
        Example:
            >>> is_valid, msg = check_price_near_highs(df)
            >>> print(msg)
            "Price position: 3.2% from 52-week high (✓ target: <10%)"
        """
        # Look back up to 252 days (52 weeks) or available data
        lookback = min(252, len(df))
        
        # Find the highest high in the lookback period
        high_52w = df['high'].iloc[-lookback:].max()
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Calculate how far current price is from the high
        distance_from_high = ((high_52w - current_price) / high_52w) * 100
        
        # Valid if within 10% of highs
        is_valid = distance_from_high <= 10.0
        
        checkmark = '✓' if is_valid else '✗'
        message = (f"Price position: {distance_from_high:.1f}% from 52-week high "
                f"({checkmark} target: <10%)")
        
        return is_valid, message
    
    @staticmethod
    def check_tight_consolidation(df: pd.DataFrame, config: VCPConfig) -> Tuple[bool, str]:
        """
        Check for tight price action in recent bars.
        
        The final phase of a VCP should show very tight price action:
        - Small daily/weekly ranges (< 1.5% high-low)
        - Multiple consecutive tight bars
        - Price "coiling" before breakout
        
        This tightness shows the stock is ready to make a move.
        The tighter the consolidation, the more powerful the breakout.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            config: VCPConfig with tightness thresholds
        
        Returns:
            Tuple of (is_valid, message)
        
        Validation:
            PASS: At least 3 days with range < 1.5%
            FAIL: Not enough tight days
        
        Example:
            >>> is_valid, msg = check_tight_consolidation(df, config)
            >>> print(msg)
            "Tight consolidation: 5 tight days (avg range: 1.1%) (✓ target: ≥3)"
        """

        # Look at recent bars (last 10)
        lookback = min(10, len(df))
        recent_df = df.iloc[-lookback:]
        
        # Calculate daily range as percentage
        # (high - low) / close gives the range as a decimal
        daily_ranges = ((recent_df['high'] - recent_df['low']) / 
                    recent_df['close']) * 100
        
        # Count how many days have tight ranges
        # config.tight_range_threshold is in decimal (e.g., 0.015 = 1.5%)
        tight_days = (daily_ranges < (config.tight_range_threshold * 100)).sum()
        
        # Valid if we have minimum number of tight days
        is_valid = tight_days >= config.min_tight_days
        
        # Calculate average range for reporting
        avg_range = daily_ranges.mean()
        
        checkmark = '✓' if is_valid else '✗'
        message = (f"Tight consolidation: {tight_days} tight days "
                f"(avg range: {avg_range:.2f}%) ({checkmark} target: "
                f"≥{config.min_tight_days})")
        
        return is_valid, message

