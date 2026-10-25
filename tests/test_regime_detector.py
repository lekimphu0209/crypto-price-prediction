"""
Quick sanity test for MA-based regime detector.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_regime.regime_detector import (
    detect_regime_trend_ma,
    segment_regimes,
    add_regime_features,
    regime_summary,
)


def test_basic_regime_detection():
    n = 120
    close = np.concatenate([
        np.linspace(100, 150, 60),   # strong uptrend
        np.linspace(150, 120, 30),    # downtrend
        np.linspace(120, 122, 30),    # roughly flat / sideways
    ])
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "close": close,
    })
    df = add_regime_features(df, short_window=7, long_window=30, onehot=True)

    assert "ma_regime" in df.columns
    assert "regime_bull" in df.columns
    assert "regime_bear" in df.columns
    assert "regime_sideways" in df.columns
    assert set(df["ma_regime"].unique()).issubset({"bull", "bear", "sideways"})

    segments = segment_regimes(df["ma_regime"])
    assert len(segments) >= 2
    labels = [s.label for s in segments]
    assert "bull" in labels, f"expected at least one bull segment, got {labels}"

    summary = regime_summary(df["ma_regime"])
    assert len(summary) == 3
    print("Regime summary:")
    print(summary.to_string(index=False))
    print("\nLast 5 segments:")
    for s in segments[-5:]:
        print(f"  {s.label}: idx {s.start_idx}-{s.end_idx} (len {s.length})")
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_basic_regime_detection()
