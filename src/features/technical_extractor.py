"""
TechnicalExtractor - builds the enhanced feature set from OHLCV data.

Feature set (Phase 2):
    close, volume, MA7, MA14, RSI, MACD, Bollinger Bands (upper/middle/lower),
    ATR (volatility), ROC (momentum), volume_change

Implements IFeatureExtractor so it can be swapped for other extractors
(macro, sentiment, ...) without touching the training pipeline.
"""
from typing import List

import pandas as pd

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.features import indicators


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
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.atr_period = atr_period
        self.roc_period = roc_period

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = data.copy()
        close = df["close"]

        df[f"ma_{self.ma_short}"] = indicators.moving_average(close, self.ma_short)
        df[f"ma_{self.ma_long}"] = indicators.moving_average(close, self.ma_long)
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

        # 'close' is already part of feature_names() and doubles as the target source.
        keep = ["timestamp"] + self.feature_names()
        keep = [c for c in keep if c in df.columns]
        df = df[keep].dropna().reset_index(drop=True)
        return df

    def feature_names(self) -> List[str]:
        return [
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
        ]
