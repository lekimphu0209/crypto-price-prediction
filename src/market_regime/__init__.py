"""
Market Regime module - MA-based market segmentation.

Provides pure functions and an extractor to label market conditions
(uptrend / downtrend / sideways) from moving-average relationships.
"""
from src.market_regime.regime_detector import (
    detect_regime_trend_ma,
    segment_regimes,
    add_regime_features,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
)
from src.market_regime.regime_feature_extractor import RegimeFeatureExtractor

__all__ = [
    "detect_regime_trend_ma",
    "segment_regimes",
    "add_regime_features",
    "RegimeFeatureExtractor",
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_SIDEWAYS",
]
