"""
TechnicalExtractor - builds the feature set from OHLCV data.

Supports four feature sets:
    A: original features (close, volume, MA, BB, RSI, MACD, ATR, ROC,
       volume_change, macro absolute levels + changes).
    B: feature set A plus stationary additions (log returns, close-to-MA
       ratios, Bollinger position/width, ATR/close ratio, log volume).
    C: only stationary features (returns, ratios, normalized volatility,
       RSI/MACD/ROC, volume changes, macro changes).
    E: enhanced feature set based on B plus ADX, multi-period RSI, MACD
       signal/histogram, longer log returns, volume MA ratio, volatility,
       and momentum acceleration.

Implements IFeatureExtractor so it can be swapped for other extractors
(macro, sentiment, ...) without touching the training pipeline.
"""
import os
from typing import List

import pandas as pd
import yfinance as yf

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.features import indicators

# Macro tickers fetched from Yahoo Finance (Gold, US Dollar Index, US 10Y rate).
# Each entry lists fallback tickers tried in order (yfinance can be flaky).
MACRO_TICKERS = {
    "gold": ["GC=F", "GLD"],
    "dxy": ["DX-Y.NYB", "DX=F"],
    "usd_rate": ["^TNX"],
}

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_MACRO_CACHE = os.path.join(_PROJECT_ROOT, "data", "macro_cache.csv")


class TechnicalExtractor(IFeatureExtractor):
    """Extract core technical-analysis features from OHLCV data."""

    def __init__(
        self,
        ma_short: int = 7,
        ma_long: int = 14,
        rsi_period: int = 14,
        bb_window: int = 20,
        atr_period: int = 14,
        roc_period: int = 12,
        feature_set: str = "A",
        exclude_features: List[str] | None = None,
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.atr_period = atr_period
        self.roc_period = roc_period
        self.feature_set = feature_set.upper()
        if self.feature_set not in {"A", "B", "C", "E"}:
            raise ValueError("feature_set must be 'A', 'B', 'C', or 'E'")
        self.exclude_features = set(exclude_features or [])

    def to_config(self) -> dict:
        """Serialize extractor settings so a saved model can rebuild it at predict time."""
        return {
            "name": "technical",
            "params": {
                "ma_short": self.ma_short,
                "ma_long": self.ma_long,
                "rsi_period": self.rsi_period,
                "bb_window": self.bb_window,
                "atr_period": self.atr_period,
                "roc_period": self.roc_period,
                "feature_set": self.feature_set,
                "exclude_features": sorted(self.exclude_features),
            },
        }

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = data.copy()
        close = df["close"]

        ma_short = indicators.moving_average(close, self.ma_short)
        ma_long = indicators.moving_average(close, self.ma_long)
        df[f"ma_{self.ma_short}"] = ma_short
        df[f"ma_{self.ma_long}"] = ma_long
        df["rsi"] = indicators.rsi(close, self.rsi_period)
        df["macd"] = indicators.macd(close)
        upper, middle, lower = indicators.bollinger_bands(close, self.bb_window)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

        # New features
        df["atr"] = indicators.atr(df["high"], df["low"], close, self.atr_period)
        df["roc"] = indicators.roc(close, self.roc_period)
        df["volume_change"] = indicators.volume_change(df["volume"])

        # Stationary features added for feature sets B, C, and E
        if self.feature_set in {"B", "C"}:
            df["log_return_1"] = indicators.log_return(close, 1)
        df["log_return_7"] = indicators.log_return(close, 7)
        df["close_to_ma_7"] = indicators.close_to_ma_ratio(close, ma_short)
        df["close_to_ma_14"] = indicators.close_to_ma_ratio(close, ma_long)
        df["bb_position"] = indicators.bollinger_position(close, upper, middle, lower)
        df["bb_width"] = indicators.bollinger_width(upper, middle, lower)
        df["atr_close_ratio"] = indicators.atr_close_ratio(df["atr"], close)
        df["log_volume"] = indicators.log_volume(df["volume"])

        # Enhanced features for feature set E
        if self.feature_set == "E":
            df["log_return_14"] = indicators.log_return(close, 14)
            df["log_return_21"] = indicators.log_return(close, 21)
            df["rsi_7"] = indicators.rsi(close, 7)
            df["rsi_21"] = indicators.rsi(close, 21)
            df["macd_signal"] = indicators.macd_signal(close)
            df["macd_histogram"] = indicators.macd_histogram(close)
            df["adx"] = indicators.adx(df["high"], df["low"], close, self.atr_period)
            df["plus_di"] = indicators.plus_di(df["high"], df["low"], close, self.atr_period)
            df["minus_di"] = indicators.minus_di(df["high"], df["low"], close, self.atr_period)
            df["close_to_bb_upper"] = indicators.close_to_bb_upper(close, upper)
            df["close_to_bb_lower"] = indicators.close_to_bb_lower(close, lower)
            df["volume_ma_ratio"] = indicators.volume_ma_ratio(df["volume"], self.ma_short)
            df["return_volatility_7"] = indicators.return_volatility(close, 7)
            df["price_momentum_7_14"] = indicators.price_momentum(close, 7, 14)

        # Macro features (Gold, DXY, USD interest rate) merged by calendar day.
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        macro = self._fetch_macro()
        macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
        df = df.merge(macro, on="date", how="left")
        for col in ("gold", "dxy", "usd_rate"):
            if col not in df.columns:
                df[col] = 0.0
        # Robust fill so a flaky macro feed never wipes the whole dataset.
        df[["gold", "dxy", "usd_rate"]] = (
            df[["gold", "dxy", "usd_rate"]].ffill().bfill().fillna(0.0)
        )
        df["gold_change"] = df["gold"].pct_change(fill_method=None).fillna(0.0)
        df["dxy_change"] = df["dxy"].pct_change(fill_method=None).fillna(0.0)
        df["rate_change"] = df["usd_rate"].diff().fillna(0.0)

        # Keep timestamp + selected features; always keep 'close' as target source.
        keep = ["timestamp"] + self.feature_names()
        if "close" not in keep:
            keep.append("close")
        keep = [c for c in keep if c in df.columns]
        df = df[keep].dropna().reset_index(drop=True)
        return df

    def _fetch_macro(self) -> pd.DataFrame:
        """Fetch macro indicators (Gold, DXY, USD rate) as a daily DataFrame.

        Tries fallback tickers per series and caches the last good result so a
        transient Yahoo Finance failure never crashes the pipeline.
        """
        out = None
        for col, tickers in MACRO_TICKERS.items():
            series = None
            for ticker in tickers:
                try:
                    m = yf.Ticker(ticker).history(period="5y")[["Close"]]
                except Exception:
                    m = None
                if m is not None and not m.empty:
                    series = m.rename(columns={"Close": col})
                    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                    break
            if series is not None:
                out = series if out is None else out.join(series, how="outer")

        if out is not None and not out.empty:
            out = out.sort_index().ffill().reset_index()
            out = out.rename(columns={out.columns[0]: "date"})
            for col in MACRO_TICKERS:
                if col not in out.columns:
                    out[col] = pd.NA
            try:
                os.makedirs(os.path.dirname(_MACRO_CACHE), exist_ok=True)
                out.to_csv(_MACRO_CACHE, index=False)
            except Exception:
                pass
            return out

        # Fetch failed entirely -> reuse cached macro data if available.
        if os.path.exists(_MACRO_CACHE):
            print("  [macro] live fetch failed, using cached macro_cache.csv")
            return pd.read_csv(_MACRO_CACHE, parse_dates=["date"])

        print("  [macro] WARNING: no macro data available; filling with zeros")
        return pd.DataFrame(columns=["date", "gold", "dxy", "usd_rate"])

    def feature_names(self) -> List[str]:
        if self.feature_set == "A":
            base = [
                "close",
                "volume",
                f"ma_{self.ma_short}",
                f"ma_{self.ma_long}",
                "rsi",
                "macd",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "atr",
                "roc",
                "volume_change",
                "gold",
                "dxy",
                "usd_rate",
                "gold_change",
                "dxy_change",
                "rate_change",
            ]
        elif self.feature_set == "B":
            base = [
                "close",
                "volume",
                f"ma_{self.ma_short}",
                f"ma_{self.ma_long}",
                "rsi",
                "macd",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "atr",
                "roc",
                "volume_change",
                "gold",
                "dxy",
                "usd_rate",
                "gold_change",
                "dxy_change",
                "rate_change",
                "log_return_1",
                "log_return_7",
                "close_to_ma_7",
                "close_to_ma_14",
                "bb_position",
                "bb_width",
                "atr_close_ratio",
                "log_volume",
            ]
        elif self.feature_set == "C":
            base = [
                "log_return_1",
                "log_return_7",
                "close_to_ma_7",
                "close_to_ma_14",
                "rsi",
                "macd",
                "bb_position",
                "bb_width",
                "roc",
                "volume_change",
                "log_volume",
                "gold_change",
                "dxy_change",
                "rate_change",
                "atr_close_ratio",
            ]
        else:  # Feature Set E
            base = [
                "close",
                "volume",
                f"ma_{self.ma_short}",
                f"ma_{self.ma_long}",
                "rsi",
                "rsi_7",
                "rsi_21",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_position",
                "bb_width",
                "atr",
                "atr_close_ratio",
                "roc",
                "volume_change",
                "log_volume",
                "volume_ma_ratio",
                "gold",
                "dxy",
                "usd_rate",
                "gold_change",
                "dxy_change",
                "rate_change",
                "log_return_7",
                "log_return_14",
                "log_return_21",
                "close_to_ma_7",
                "close_to_ma_14",
                "close_to_bb_upper",
                "close_to_bb_lower",
                "adx",
                "plus_di",
                "minus_di",
                "return_volatility_7",
                "price_momentum_7_14",
            ]
        return [c for c in base if c not in self.exclude_features]
