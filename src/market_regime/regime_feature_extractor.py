"""
RegimeFeatureExtractor - wraps TechnicalExtractor and injects MA-based regime
features while preserving the existing interface (IFeatureExtractor).

This keeps regime knowledge in one place (`src/market_regime/`) and lets the
training / prediction pipelines switch to it without any other changes.
"""
from typing import List

import pandas as pd

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.features.technical_extractor import TechnicalExtractor
from src.market_regime.regime_detector import add_regime_features, REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS


class RegimeFeatureExtractor(IFeatureExtractor):
    """
    Feature extractor that adds MA-based market-regime columns on top of the
    standard TechnicalExtractor output.
    """

    def __init__(
        self,
        base_extractor: TechnicalExtractor | None = None,
        base_feature_set: str | None = None,
        short_window: int = 7,
        long_window: int = 30,
        slope_window: int = 3,
        onehot: bool = True,
        include_label: bool = False,
    ):
        """
        Args:
            base_extractor: underlying TechnicalExtractor instance.  Created
                with default settings if omitted.
            base_feature_set: convenience string to build a TechnicalExtractor
                with that feature set.  Ignored if `base_extractor` is provided.
            short_window: MA short window used for regime detection.
            long_window: MA long window used for regime detection.
            slope_window: window used to decide whether an MA is rising/falling.
            onehot: emit one-hot regime columns (`regime_bull`, ...).
            include_label: emit the categorical `ma_regime` column as well.
        """
        if base_extractor is not None:
            self.base_extractor = base_extractor
        elif base_feature_set is not None:
            self.base_extractor = TechnicalExtractor(feature_set=base_feature_set)
        else:
            self.base_extractor = TechnicalExtractor()
        self.short_window = short_window
        self.long_window = long_window
        self.slope_window = slope_window
        self.onehot = onehot
        self.include_label = include_label

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1. Build the standard technical feature set.
        df = self.base_extractor.extract(data)

        # 2. Append MA regime columns (uses close prices *after* alignment).
        df = add_regime_features(
            df,
            close_col="close",
            short_window=self.short_window,
            long_window=self.long_window,
            slope_window=self.slope_window,
            onehot=self.onehot,
        )

        # Drop the bare categorical label unless requested.
        if not self.include_label and "ma_regime" in df.columns:
            df = df.drop(columns=["ma_regime"])

        # Forward-fill any NaNs introduced at the start of the long MA window.
        # Using ffill/bfill is safe here because regime columns are labels / ratios.
        df = df.ffill().bfill().dropna().reset_index(drop=True)
        return df

    def feature_names(self) -> List[str]:
        base = self.base_extractor.feature_names()
        regime_cols: List[str] = []
        # MA levels used for regime detection may have different windows than the
        # base extractor (e.g. ma_30).  Include any that are not already present.
        short_col = f"ma_{self.short_window}"
        long_col = f"ma_{self.long_window}"
        for col in (short_col, long_col):
            if col not in base:
                regime_cols.append(col)
        if self.include_label:
            regime_cols.append("ma_regime")
        if self.onehot:
            regime_cols.extend([f"regime_{REGIME_BULL}", f"regime_{REGIME_BEAR}", f"regime_{REGIME_SIDEWAYS}"])
        return base + regime_cols
