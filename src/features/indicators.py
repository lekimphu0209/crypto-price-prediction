"""
Technical Indicators - pure, stateless functions.

Each function takes a pandas Series (or DataFrame) and returns the computed
indicator. Keeping these pure makes them trivial to test and reuse.
"""
from typing import Tuple

import numpy as np
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


def log_return(series: pd.Series, period: int = 1) -> pd.Series:
    """Log return over `period` periods."""
    return np.log(series / series.shift(period)).fillna(0).replace([np.inf, -np.inf], 0)


def close_to_ma_ratio(close: pd.Series, ma: pd.Series) -> pd.Series:
    """Close price relative to its moving average (1.0 = at MA)."""
    return (close / ma).fillna(1.0).replace([np.inf, -np.inf], 1.0)


def bollinger_position(close: pd.Series, upper: pd.Series, middle: pd.Series, lower: pd.Series) -> pd.Series:
    """Position of close within Bollinger Bands (0 = lower, 1 = upper, 0.5 = middle)."""
    width = upper - lower
    return ((close - lower) / width).fillna(0.5).replace([np.inf, -np.inf], 0.5)


def bollinger_width(upper: pd.Series, middle: pd.Series, lower: pd.Series) -> pd.Series:
    """Bollinger Band width normalized by middle band."""
    width = upper - lower
    return (width / middle).fillna(0).replace([np.inf, -np.inf], 0)


def atr_close_ratio(atr: pd.Series, close: pd.Series) -> pd.Series:
    """ATR normalized by close price (volatility relative to price level)."""
    return (atr / close).fillna(0).replace([np.inf, -np.inf], 0)


def log_volume(volume: pd.Series) -> pd.Series:
    """Natural log of volume, with inf handling."""
    return np.log(volume).replace([np.inf, -np.inf], np.nan).fillna(0)


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD signal line (EMA of MACD line)."""
    macd_line = macd(close, fast=fast, slow=slow)
    return macd_line.ewm(span=signal, adjust=False).mean()


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    return macd(close, fast=fast, slow=slow) - macd_signal(close, fast=fast, slow=slow, signal=signal)


def _dm(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Helper for ADX: +DM, -DM, TR."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = high.shift(1)  # use close if available, but high is fine for tr helper
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return plus_dm, minus_dm, tr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength, 0-100)."""
    plus_dm, minus_dm, tr = _dm(high, low)
    atr_val = atr(high, low, close, period=period)
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr_val
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr_val
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(window=period).mean()


def plus_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Plus Directional Indicator."""
    plus_dm, _, tr = _dm(high, low)
    atr_val = atr(high, low, close, period=period)
    return 100 * plus_dm.rolling(window=period).mean() / atr_val


def minus_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Minus Directional Indicator."""
    _, minus_dm, tr = _dm(high, low)
    atr_val = atr(high, low, close, period=period)
    return 100 * minus_dm.rolling(window=period).mean() / atr_val


def close_to_bb_upper(close: pd.Series, upper: pd.Series) -> pd.Series:
    """Close relative to upper Bollinger Band (1.0 = at upper)."""
    return (close / upper).fillna(1.0).replace([np.inf, -np.inf], 1.0)


def close_to_bb_lower(close: pd.Series, lower: pd.Series) -> pd.Series:
    """Close relative to lower Bollinger Band (1.0 = at lower)."""
    return (close / lower).fillna(1.0).replace([np.inf, -np.inf], 1.0)


def volume_ma_ratio(volume: pd.Series, window: int = 7) -> pd.Series:
    """Volume relative to its moving average."""
    ma = volume.rolling(window=window).mean()
    return (volume / ma).fillna(1.0).replace([np.inf, -np.inf], 1.0)


def return_volatility(close: pd.Series, window: int = 7) -> pd.Series:
    """Rolling standard deviation of log returns."""
    lr = log_return(close, 1)
    return lr.rolling(window=window).std().fillna(0).replace([np.inf, -np.inf], 0)


def price_momentum(close: pd.Series, short_period: int = 7, long_period: int = 14) -> pd.Series:
    """Difference between short and long log returns (momentum acceleration)."""
    return log_return(close, short_period) - log_return(close, long_period)
