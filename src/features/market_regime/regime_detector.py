"""
MA-based market regime detection.

A market regime is classified from the relationship between short and long
moving averages and the current close price.  The logic intentionally stays
pure (no ML) so it is explainable, fast and reusable across symbols.

Regime definitions:
    BULL: price sits above a rising short-MA, and short-MA is above a rising
          long-MA.  This captures sustained upward momentum.
    BEAR: price sits below a falling short-MA, and short-MA is below a falling
          long-MA.  This captures sustained downward momentum.
    SIDEWAYS: everything else - the two MAs are mixed, flat, or the price is
              crossing between them, indicating consolidation / uncertainty.
"""
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"

_REGIME_ORDER = [REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR]


def detect_regime_trend_ma(
    close: pd.Series,
    short_window: int = 7,
    long_window: int = 30,
    slope_window: int = 3,
    require_slope: bool = True,
) -> pd.Series:
    """
    Label each bar as bull / sideways / bear from MA relationships.

    Args:
        close: closing price series (sorted ascending by time).
        short_window: window for the short moving average.
        long_window: window for the long moving average.
        slope_window: window used to test whether an MA is rising/falling.
        require_slope: if False, classification only uses level relationships
            (close vs MA_short vs MA_long), ignoring slope.  Useful when data
            is too short to obtain a stable slope estimate.

    Returns:
        pd.Series of regime labels aligned with `close`.
    """
    if len(close) < long_window:
        raise ValueError(
            f"close length ({len(close)}) must be >= long_window ({long_window})"
        )

    ma_short = close.rolling(window=short_window, min_periods=1).mean()
    ma_long = close.rolling(window=long_window, min_periods=1).mean()

    above_short = close > ma_short
    above_long = close > ma_long
    short_above_long = ma_short > ma_long

    if require_slope:
        short_rising = ma_short > ma_short.shift(slope_window)
        short_falling = ma_short < ma_short.shift(slope_window)
        long_rising = ma_long > ma_long.shift(slope_window)
        long_falling = ma_long < ma_long.shift(slope_window)
    else:
        short_rising = pd.Series(True, index=close.index)
        short_falling = pd.Series(True, index=close.index)
        long_rising = pd.Series(True, index=close.index)
        long_falling = pd.Series(True, index=close.index)

    bull = (
        above_short
        & above_long
        & short_above_long
        & short_rising
        & long_rising
    )
    bear = (
        ~above_short
        & ~above_long
        & ~short_above_long
        & short_falling
        & long_falling
    )

    regime = pd.Series(REGIME_SIDEWAYS, index=close.index)
    regime[bull] = REGIME_BULL
    regime[bear] = REGIME_BEAR
    return regime


@dataclass(frozen=True)
class RegimeSegment:
    """A contiguous run of a single regime."""

    start_idx: int
    end_idx: int  # inclusive
    label: str

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1


def segment_regimes(regime: pd.Series) -> List[RegimeSegment]:
    """
    Collapse a regime label series into contiguous segments.

    Args:
        regime: pd.Series of regime labels.

    Returns:
        List of RegimeSegment objects.
    """
    if regime.empty:
        return []

    segments: List[RegimeSegment] = []
    current_label = regime.iloc[0]
    start = 0

    for i in range(1, len(regime)):
        if regime.iloc[i] != current_label:
            segments.append(RegimeSegment(start, i - 1, current_label))
            current_label = regime.iloc[i]
            start = i

    segments.append(RegimeSegment(start, len(regime) - 1, current_label))
    return segments


def add_regime_features(
    df: pd.DataFrame,
    close_col: str = "close",
    short_window: int = 7,
    long_window: int = 30,
    slope_window: int = 3,
    onehot: bool = True,
) -> pd.DataFrame:
    """
    Add MA-based regime columns to a DataFrame.

    Args:
        df: DataFrame containing `close_col`.
        close_col: name of the close-price column.
        short_window, long_window, slope_window: passed to
            `detect_regime_trend_ma`.
        onehot: if True, add `regime_bull`, `regime_bear`, `regime_sideways`
            one-hot columns; otherwise add a single `regime` categorical column.

    Returns:
        DataFrame with regime columns appended.
    """
    if close_col not in df.columns:
        raise ValueError(f"close column '{close_col}' not found")

    close = df[close_col]
    require_slope = len(close) >= max(long_window, slope_window)

    ma_short = close.rolling(window=short_window, min_periods=1).mean()
    ma_long = close.rolling(window=long_window, min_periods=1).mean()

    regime = detect_regime_trend_ma(
        close,
        short_window=short_window,
        long_window=long_window,
        slope_window=slope_window,
        require_slope=require_slope,
    )

    out = df.copy()
    out["ma_regime"] = regime.values
    # Add MA columns only if they are not already present to avoid duplicates.
    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"
    if short_col not in out.columns:
        out[short_col] = ma_short.values
    if long_col not in out.columns:
        out[long_col] = ma_long.values

    if onehot:
        for label in _REGIME_ORDER:
            out[f"regime_{label}"] = (regime == label).astype(int).values

    return out


def regime_summary(regime: pd.Series) -> pd.DataFrame:
    """
    Return a small summary table: label, count, pct, longest_run.
    """
    counts = regime.value_counts()
    segments = segment_regimes(regime)
    max_run = {label: 0 for label in _REGIME_ORDER}
    for seg in segments:
        max_run[seg.label] = max(max_run.get(seg.label, 0), seg.length)

    rows = []
    for label in _REGIME_ORDER:
        rows.append(
            {
                "regime": label,
                "count": int(counts.get(label, 0)),
                "pct": float(counts.get(label, 0)) / len(regime) * 100 if len(regime) else 0.0,
                "longest_run": max_run[label],
            }
        )
    return pd.DataFrame(rows)
