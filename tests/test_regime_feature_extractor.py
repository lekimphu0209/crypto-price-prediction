"""
Sanity test for RegimeFeatureExtractor integration.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.features.market_regime.regime_feature_extractor import RegimeFeatureExtractor


def test_regime_feature_extractor():
    n = 60
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": np.linspace(100, 140, n),
        "high": np.linspace(105, 145, n),
        "low": np.linspace(95, 135, n),
        "close": np.linspace(100, 140, n),
        "volume": np.ones(n) * 1000.0,
    })
    extractor = RegimeFeatureExtractor(base_feature_set="A")
    features = extractor.extract(df)
    names = extractor.feature_names()

    assert "regime_bull" in features.columns
    assert "regime_bear" in features.columns
    assert "regime_sideways" in features.columns
    assert "regime_bull" in names
    print(f"Extracted {len(names)} features: {names}")
    print(f"Rows: {len(features)}")
    print("RegimeFeatureExtractor test passed.")


if __name__ == "__main__":
    test_regime_feature_extractor()
