"""
Technical Indicators - pure, stateless functions.

Each function takes a pandas Series (or DataFrame) and returns the computed
indicator. Keeping these pure makes them trivial to test and reuse.
"""
from typing import Tuple

import pandas as pd


def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over `window` periods."""
    return close.rolling(window=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line = EMA(fast) - EMA(slow)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns:
        (upper, middle, lower) bands.
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range - volatility measure.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change - momentum indicator.

    ROC = (close - close_period_ago) / close_period_ago
    """
    return (close - close.shift(period)) / close.shift(period)


def volume_change(volume: pd.Series, period: int = 1) -> pd.Series:
    """
    Volume change rate.

    vol_change = (volume - volume_period_ago) / volume_period_ago
    """
    return (volume - volume.shift(period)) / volume.shift(period)
