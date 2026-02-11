"""
VCP Pattern Detector - Comprehensive Fixes
===========================================
ROOT CAUSES IDENTIFIED & FIXED:

FIX 1: VCPConfig - Thresholds too strict for Indian markets
    - contraction depth ranges were too narrow
    - volume_dryup_threshold too aggressive (0.60 → 0.75)
    - atr_contraction_threshold too aggressive (0.70 → 0.80)
    - rsi_floor too high (40 → 35)
    - tight_range_threshold too tight (0.015 → 0.025)
    - min_tight_days reduced (3 → 2)

FIX 2: identify_contractions() - Wrong index handling
    - df.index.get_loc(low_idx) fails when index is DatetimeIndex
    - Segment slicing was using positional slicing incorrectly
    - contraction_segment = df.loc[high_idx:low_idx] fails on DatetimeIndex
    - Fixed by using iloc-based positional indexing throughout

FIX 3: identify_contractions() - Contractions not being limited to recent window
    - Was finding contractions across entire 2-year history
    - Most recent pattern (last 6 months) is what matters for trading
    - Added a lookback_days parameter to focus on recent price action

FIX 4: validate_contraction_sequence() - Too strict, rejecting valid patterns
    - Was requiring ALL contractions to be in range
    - Only the LAST 3 contractions matter for VCP
    - ATR strict monotonic decline was rejecting near-monotonic patterns
    - Fixed: allow small ATR violations (10% tolerance)

FIX 5: check_volume_dryup() - Wrong volume comparison window
    - Was comparing last contraction volume to recent 50-day average
    - But recent 50 days IS the consolidation, skewing the average
    - Fixed: compare to 50-100 day average BEFORE the consolidation

FIX 6: check_tight_consolidation() - Not checking consecutive days
    - Was just counting total tight days, not consecutive ones
    - Fixed: check for consecutive tight days properly

FIX 7: check_rsi_strength() - Wrong RSI range for Indian stocks
    - Indian mid/small caps often have RSI 45-75 range
    - Was rejecting valid patterns where RSI was slightly above 65
    - Fixed: widened RSI consolidation range

FIX 8: PivotDetector - lookback=3 too small, finding too many false pivots
    - Creates too many micro-contractions
    - Fixed: use adaptive lookback based on timeframe

FIX 9: find_swing_highs/lows - Uses strict inequality, misses flat tops
    - >= instead of > for equal bars check
    - Fixed: use <= for flat top handling
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# FIX 1: VCPConfig - Relaxed thresholds for Indian markets
# ============================================================================

class Timeframe(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class VCPConfig:
    """
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
    timeframe: Timeframe = Timeframe.DAILY

    # Contraction count
    min_contractions: int = 3
    max_contractions: int = 5

    # Daily indicator periods
    daily_atr_period: int = 14
    daily_bb_period: int = 20
    daily_ma_short: int = 21
    daily_ma_medium: int = 50
    daily_ma_long: int = 200
    daily_rsi_period: int = 14
    daily_volume_period: int = 50

    # Weekly indicator periods
    weekly_atr_period: int = 10
    weekly_bb_period: int = 20
    weekly_ma_short: int = 10
    weekly_ma_long: int = 40
    weekly_rsi_period: int = 14
    weekly_volume_period: int = 10

    # FIX 1a: Wider contraction depth ranges for Indian markets
    # Indian stocks can have deeper initial pullbacks
    contraction_1_min: float = 5.0      # Was 8.0 → relaxed to 5.0
    contraction_1_max: float = 35.0     # Was 25.0 → relaxed to 35.0
    contraction_2_min: float = 3.0      # Was 5.0 → relaxed to 3.0
    contraction_2_max: float = 25.0     # Was 15.0 → relaxed to 25.0
    contraction_3_min: float = 2.0      # Was 3.0 → relaxed to 2.0
    contraction_3_max: float = 15.0     # Was 10.0 → relaxed to 15.0
    contraction_4_max: float = 10.0     # Was 6.0 → relaxed to 10.0

    # FIX 1b: Relaxed volume threshold
    # 0.75 means volume should be below 75% of average (was 60%)
    volume_dryup_threshold: float = 0.75        # Was 0.60
    breakout_volume_threshold: float = 1.30     # Was 1.50 → easier to trigger

    # FIX 1c: Relaxed RSI thresholds
    rsi_floor: float = 35.0                     # Was 40.0
    rsi_consolidation_min: float = 40.0         # Was 48.0
    rsi_consolidation_max: float = 70.0         # Was 65.0
    rsi_breakout: float = 55.0                  # Was 60.0

    # FIX 1d: Relaxed ATR contraction
    # 0.80 means final ATR only needs to be 80% of initial (20% reduction)
    atr_contraction_threshold: float = 0.80     # Was 0.70

    # FIX 1e: Relaxed tight consolidation
    tight_range_threshold: float = 0.025        # Was 0.015 (2.5% range allowed)
    min_tight_days: int = 2                     # Was 3

    # MA alignment
    ma_alignment_required: bool = True
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


# ============================================================================
# FIX 2 + 3: ContractionDetector - Index handling & recent window
# ============================================================================

@dataclass
class Contraction:
    """Represents a single contraction phase."""
    start_pos: int          # FIX: Use positional index, not label index
    end_pos: int
    start_date: object      # Store actual date for reference
    end_date: object
    high_price: float
    low_price: float
    depth_percent: float
    duration_bars: int
    avg_volume: float
    avg_atr: float
    rsi_low: float


@dataclass
class VCPResult:
    """Complete results of VCP pattern detection."""
    is_vcp: bool
    confidence_score: float
    contractions: List[Contraction]
    current_stage: str
    breakout_price: Optional[float]
    stop_loss: Optional[float]
    signals: Dict[str, bool]
    metrics: Dict[str, float]
    messages: List[str]
    fail_reasons: List[str]     # FIX: Explicit fail reasons for debugging


# ============================================================================
# TechnicalIndicators - Minor fixes
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators with fixes for edge cases."""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        FIX: Added min_periods=1 to handle short data windows gracefully.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # FIX: min_periods=1 avoids NaN for early rows
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI.
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
        Calculate multiple SMAs.
        FIX: Added min_periods=1 so early rows aren't all NaN.
        """
        mas = {}
        for period in periods:
            mas[period] = df['close'].rolling(window=period, min_periods=1).mean()
        return mas

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(df['close'].diff()).fillna(0)
        obv = (direction * df['volume']).cumsum()
        return obv

    @staticmethod
    def calculate_bb_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Bollinger Band Width.
        FIX: Added min_periods=1.
        """
        close = df['close']
        middle = close.rolling(window=period, min_periods=1).mean()
        std_dev = close.rolling(window=period, min_periods=1).std()
        upper = middle + (2 * std_dev)
        lower = middle - (2 * std_dev)
        bb_width = ((upper - lower) / middle.replace(0, np.nan)) * 100
        return bb_width.fillna(0)


# ============================================================================
# FIX 8 + 9: PivotDetector - Better pivot detection
# ============================================================================

class PivotDetector:
    """
    Detect swing highs and lows.
    
    FIXES:
    - Use configurable lookback (not hardcoded 3)
    - Handle flat tops/bottoms with <= instead of <
    - Return both positional index AND date for reliability
    """

    @staticmethod
    def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """
        Find swing high pivots.
        
        FIX: Changed strict < to <= for the comparison to handle flat tops.
        FIX: Increased default lookback from 3 to 5 to reduce noise.
        
        Args:
            df: DataFrame with 'high' column
            lookback: Bars on each side (higher = fewer but more significant pivots)
        
        Returns:
            List of positional indices (iloc positions)
        """
        highs = []
        high_prices = df['high'].values
        n = len(high_prices)

        for i in range(lookback, n - lookback):
            current = high_prices[i]
            is_high = True

            for j in range(1, lookback + 1):
                # FIX: Use >= to handle flat tops (equal highs)
                if current < high_prices[i - j] or current < high_prices[i + j]:
                    is_high = False
                    break

            if is_high:
                highs.append(i)

        return highs

    @staticmethod
    def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> List[int]:
        """
        Find swing low pivots.
        
        FIX: Changed strict > to >= for the comparison to handle flat bottoms.
        FIX: Increased default lookback from 3 to 5 to reduce noise.
        
        Returns:
            List of positional indices (iloc positions)
        """
        lows = []
        low_prices = df['low'].values
        n = len(low_prices)

        for i in range(lookback, n - lookback):
            current = low_prices[i]
            is_low = True

            for j in range(1, lookback + 1):
                # FIX: Use <= to handle flat bottoms
                if current > low_prices[i - j] or current > low_prices[i + j]:
                    is_low = False
                    break

            if is_low:
                lows.append(i)

        return lows


# ============================================================================
# FIX 2 + 3 + 4: ContractionDetector - Full rewrite with fixes
# ============================================================================

class ContractionDetector:
    """
    Detect and validate contraction phases.
    
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
        Identify contraction phases using positional indexing.
        
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
    def validate_contraction_sequence(
        contractions: List[Contraction],
        config: VCPConfig
    ) -> Tuple[bool, List[str]]:
        """
        Validate contraction sequence with relaxed rules.
        
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
# FIX 5 + 6 + 7: SignalValidator - Fixed volume, RSI, tight consolidation
# ============================================================================

class SignalValidator:
    """
    Validate VCP signals.
    
    FIXES:
    - Volume comparison uses pre-consolidation baseline
    - Tight consolidation checks consecutive days properly
    - RSI range widened for Indian markets
    """

    @staticmethod
    def check_volume_dryup(
        df: pd.DataFrame,
        contractions: List[Contraction],
        config: VCPConfig
    ) -> Tuple[bool, str]:
        """
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
    def check_atr_contraction(
        df: pd.DataFrame,
        contractions: List[Contraction],
        config: VCPConfig
    ) -> Tuple[bool, str]:
        """
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
    def check_rsi_strength(
        df: pd.DataFrame,
        contractions: List[Contraction],
        config: VCPConfig
    ) -> Tuple[bool, str]:
        """
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
    def check_ma_alignment(
        df: pd.DataFrame,
        config: VCPConfig
    ) -> Tuple[bool, str]:
        """
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
        Check price is near 52-week highs.
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
    def check_tight_consolidation(
        df: pd.DataFrame,
        config: VCPConfig
    ) -> Tuple[bool, str]:
        """
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
# VCPAnalyzer - Main analyzer with all fixes integrated
# ============================================================================

class VCPAnalyzer:
    """
    Main VCP analyzer with all fixes integrated.
    
    FIXES applied here:
    - prepare_data uses min_periods=1 for indicators
    - confidence_score uses lenient thresholds (50 instead of 60)
    - fail_reasons collected for easy debugging
    - Added diagnostic mode to print why pattern failed
    """

    def __init__(self, config: Optional[VCPConfig] = None):
        self.config              = config or VCPConfig()
        self.indicators          = TechnicalIndicators()
        self.contraction_detector = ContractionDetector()
        self.signal_validator    = SignalValidator()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicators to dataframe.
        FIX: Ensure all indicator calculations use min_periods=1
             so we don't lose rows at the beginning.
        """
        tf = self.config.timeframe

        atr_p = self.config.daily_atr_period if tf == Timeframe.DAILY \
                else self.config.weekly_atr_period
        df['atr'] = self.indicators.calculate_atr(df, atr_p)

        bb_p = self.config.daily_bb_period if tf == Timeframe.DAILY \
               else self.config.weekly_bb_period
        df['bb_width'] = self.indicators.calculate_bb_width(df, bb_p)

        rsi_p = self.config.daily_rsi_period if tf == Timeframe.DAILY \
                else self.config.weekly_rsi_period
        df['rsi'] = self.indicators.calculate_rsi(df, rsi_p)

        if tf == Timeframe.DAILY:
            ma_periods = [
                self.config.daily_ma_short,
                self.config.daily_ma_medium,
                self.config.daily_ma_long
            ]
        else:
            ma_periods = [
                self.config.weekly_ma_short,
                self.config.weekly_ma_long
            ]

        mas = self.indicators.calculate_moving_averages(df, ma_periods)
        for period, ma in mas.items():
            df[f'ma_{period}'] = ma

        df['obv'] = self.indicators.calculate_obv(df)

        # FIX: Use dropna only on critical columns, not all columns
        # This preserves more rows
        critical_cols = ['atr', 'rsi', 'close', 'high', 'low', 'volume']
        df = df.dropna(subset=critical_cols)

        return df

    def calculate_confidence_score(
        self,
        signals: Dict[str, bool],
        contractions: List[Contraction]
    ) -> float:
        """
        Calculate confidence score.
        FIX: Rebalanced weights; 'contraction_sequence' is most critical.
        """
        weights = {
            'contraction_sequence': 30,   # Most important
            'volume_dryup':         15,
            'atr_contraction':      15,
            'rsi_strength':         15,
            'ma_alignment':         10,
            'price_near_highs':     10,
            'tight_consolidation':  5,    # Least critical (was 10)
        }
        score = sum(weights.get(sig, 0) for sig, passed in signals.items() if passed)

        # Bonus for contraction count
        n = len(contractions)
        if n == 4:   score += 10
        elif n == 3: score += 5
        elif n >= 5: score += 3

        return min(score, 100.0)

    def determine_stage(
        self,
        df: pd.DataFrame,
        contractions: List[Contraction]
    ) -> str:
        """Determine current VCP stage."""
        if not contractions:
            return "no_pattern"

        current_price = df['close'].iloc[-1]
        last_c        = contractions[-1]

        # FIX: Use positional indexing for consolidation high
        consol_high = df['high'].iloc[last_c.start_pos:].max()

        vol_period  = (self.config.daily_volume_period
                       if self.config.timeframe == Timeframe.DAILY
                       else self.config.weekly_volume_period)
        avg_vol     = df['volume'].iloc[-vol_period:].mean()
        curr_vol    = df['volume'].iloc[-1]

        if (current_price > consol_high * 1.02 and
                curr_vol > avg_vol * self.config.breakout_volume_threshold):
            return "breaking_out"
        elif current_price > consol_high * 1.10:
            return "extended"
        else:
            return "consolidating"

    def calculate_entry_and_stop(
        self,
        df: pd.DataFrame,
        contractions: List[Contraction]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate entry and stop levels."""
        if not contractions:
            return None, None

        last_c      = contractions[-1]
        consol_high = df['high'].iloc[last_c.start_pos:].max()
        entry_price = consol_high * 1.02

        consol_low  = df['low'].iloc[last_c.start_pos:].min()
        swing_stop  = consol_low * 0.98
        current_atr = df['atr'].iloc[-1]
        atr_stop    = entry_price - (2 * current_atr)

        stop_loss   = max(atr_stop, swing_stop)
        return entry_price, stop_loss

    def analyze(self, symbol: str, df_input: pd.DataFrame) -> VCPResult:
        """
        Analyze a stock for VCP pattern.
        
        FIX: Accepts pre-loaded DataFrame instead of downloading.
             This suits the SQLite-based architecture described.
        
        Added fail_reasons to VCPResult for easy debugging.
        
        Args:
            symbol: Stock ticker (for logging)
            df_input: Pre-loaded OHLCV DataFrame
        
        Returns:
            VCPResult with full analysis
        """
        messages    = []
        fail_reasons = []

        try:
            df = df_input.copy()

            # Ensure columns are lowercase
            df.columns = [c.lower() for c in df.columns]

            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            messages.append(f"[{symbol}] Loaded {len(df)} bars")

            # Add indicators
            df = self.prepare_data(df)
            messages.append(f"[{symbol}] Indicators calculated, {len(df)} bars after dropna")

            # Detect contractions in recent window
            contractions = self.contraction_detector.identify_contractions(df, self.config)
            messages.append(f"[{symbol}] Found {len(contractions)} contractions")

            for i, c in enumerate(contractions):
                messages.append(
                    f"  C{i+1}: depth={c.depth_percent:.1f}%, "
                    f"dur={c.duration_bars}d, "
                    f"atr={c.avg_atr:.2f}, "
                    f"rsi_low={c.rsi_low:.1f}"
                )

            # Validate contraction sequence
            seq_valid, seq_msgs = self.contraction_detector.validate_contraction_sequence(
                contractions, self.config
            )
            messages.extend(seq_msgs)
            if not seq_valid:
                fail_reasons.extend([m for m in seq_msgs if m.startswith("FAIL")])

            # Individual signal checks
            signals = {}

            checks = [
                ('volume_dryup',
                 lambda: self.signal_validator.check_volume_dryup(df, contractions, self.config)),
                ('atr_contraction',
                 lambda: self.signal_validator.check_atr_contraction(df, contractions, self.config)),
                ('rsi_strength',
                 lambda: self.signal_validator.check_rsi_strength(df, contractions, self.config)),
                ('ma_alignment',
                 lambda: self.signal_validator.check_ma_alignment(df, self.config)),
                ('price_near_highs',
                 lambda: self.signal_validator.check_price_near_highs(df)),
                ('tight_consolidation',
                 lambda: self.signal_validator.check_tight_consolidation(df, self.config)),
            ]

            for name, fn in checks:
                valid, msg = fn()
                signals[name] = valid
                messages.append(f"  [{name}] {msg}")
                if not valid:
                    fail_reasons.append(f"{name}: {msg}")

            signals['contraction_sequence'] = seq_valid

            # Confidence + decision
            confidence = self.calculate_confidence_score(signals, contractions)

            # FIX: Lower threshold to 50 (was 60)
            is_vcp = (confidence >= 50.0 and
                      len(contractions) >= self.config.min_contractions)

            stage        = self.determine_stage(df, contractions)
            entry, stop  = self.calculate_entry_and_stop(df, contractions)

            metrics = {
                'current_price':       float(df['close'].iloc[-1]),
                'current_atr':         float(df['atr'].iloc[-1]),
                'current_rsi':         float(df['rsi'].iloc[-1]),
                'bb_width':            float(df['bb_width'].iloc[-1]),
                'num_contractions':    float(len(contractions)),
                'latest_volume_ratio': float(
                    df['volume'].iloc[-5:].mean() /
                    max(df['volume'].iloc[-50:].mean(), 1)
                ),
            }

            messages.append(f"{'='*50}")
            messages.append(f"[{symbol}] Result: {'VCP ✓' if is_vcp else 'NOT VCP ✗'} | "
                           f"Confidence: {confidence:.1f}/100 | Stage: {stage}")
            if fail_reasons:
                messages.append(f"[{symbol}] Fail reasons: {'; '.join(fail_reasons)}")

            return VCPResult(
                is_vcp=is_vcp,
                confidence_score=confidence,
                contractions=contractions,
                current_stage=stage,
                breakout_price=entry,
                stop_loss=stop,
                signals=signals,
                metrics=metrics,
                messages=messages,
                fail_reasons=fail_reasons
            )

        except Exception as e:
            err = f"[{symbol}] ERROR: {str(e)}"
            messages.append(err)
            return VCPResult(
                is_vcp=False, confidence_score=0.0, contractions=[],
                current_stage="error", breakout_price=None, stop_loss=None,
                signals={}, metrics={}, messages=messages,
                fail_reasons=[err]
            )


# ============================================================================
# DIAGNOSTIC TOOL - Run this to see WHY a stock is failing
# ============================================================================

def diagnose_stock(symbol: str, df: pd.DataFrame, config: VCPConfig = None) -> None:
    """
    Diagnostic tool to print exactly WHY a stock is failing VCP checks.
    
    Run this on stocks you KNOW are VCP patterns (from Chartink)
    to see which specific checks are failing and tune parameters.
    
    Args:
        symbol: Stock name (for display)
        df: OHLCV DataFrame
        config: VCPConfig (uses default if None)
    
    Example:
        >>> df = load_stock_from_sqlite('RELIANCE')
        >>> diagnose_stock('RELIANCE', df)
    """
    config   = config or VCPConfig()
    analyzer = VCPAnalyzer(config)
    result   = analyzer.analyze(symbol, df)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {symbol}")
    print(f"{'='*60}")

    print(f"\nIS VCP: {result.is_vcp}")
    print(f"CONFIDENCE: {result.confidence_score:.1f}/100")
    print(f"STAGE: {result.current_stage}")
    print(f"CONTRACTIONS FOUND: {len(result.contractions)}")

    print(f"\n--- CONTRACTION DETAILS ---")
    for i, c in enumerate(result.contractions):
        print(f"  C{i+1}: {c.depth_percent:.1f}% depth | "
              f"{c.duration_bars} bars | "
              f"ATR={c.avg_atr:.2f} | "
              f"RSI_low={c.rsi_low:.1f} | "
              f"{c.start_date} → {c.end_date}")

    print(f"\n--- SIGNALS ---")
    for sig, passed in result.signals.items():
        mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {sig:<25} {mark}")

    print(f"\n--- FAIL REASONS ---")
    if result.fail_reasons:
        for r in result.fail_reasons:
            print(f"  ✗ {r}")
    else:
        print("  None - pattern passed all checks")

    print(f"\n--- METRICS ---")
    for k, v in result.metrics.items():
        print(f"  {k:<25} {v:.2f}")

    print(f"\n--- FULL LOG ---")
    for msg in result.messages:
        print(f"  {msg}")