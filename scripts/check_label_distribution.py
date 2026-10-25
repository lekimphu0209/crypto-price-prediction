"""
Check UP/DOWN label distribution in train vs test sets.

Purpose: Verify if bias UP is due to imbalanced training data.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.training.composition import build_training_pipeline, DATA_DIR, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder


def main():
    print("=" * 80)
    print("CHECK LABEL DISTRIBUTION: UP/DOWN IN TRAIN VS TEST")
    print("=" * 80)

    SYMBOL = "BTCUSDT"
    FEATURE_SET = "B"

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)

    # Load data
    print(f"\nLoading data for {SYMBOL}...")
    df = pipeline.loader.load(SYMBOL)
    print(f"Raw data shape: {df.shape}")

    # Extract features
    features = pipeline.extractor.extract(df)
    print(f"Features shape: {features.shape}")

    # Build target (aligned with features index)
    target = pipeline.target_builder.build(
        base_close=df["close"].values,
        future_close=df["close"].shift(-HORIZON).values,
    )
    print(f"Target shape: {target.shape}")

    # Align target with features (features may have fewer rows due to NaN in extraction)
    target = target[:len(features)]
    print(f"Aligned target shape: {target.shape}")

    # Remove NaN rows
    mask = ~pd.isna(target)
    features = features[mask].values
    target = target[mask]
    print(f"After removing NaN: features={features.shape}, target={target.shape}")

    # Count UP/DOWN in full dataset
    up_count = (target == 1).sum()
    down_count = (target == 0).sum()
    total = len(target)
    up_pct = up_count / total * 100
    down_pct = down_count / total * 100

    print(f"\nFULL DATASET:")
    print(f"  UP: {up_count} ({up_pct:.2f}%)")
    print(f"  DOWN: {down_count} ({down_pct:.2f}%)")

    # Train/Test split (80/20, time-ordered)
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]

    print(f"\nSPLIT:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Count UP/DOWN in train
    train_up = (y_train == 1).sum()
    train_down = (y_train == 0).sum()
    train_total = len(y_train)
    train_up_pct = train_up / train_total * 100
    train_down_pct = train_down / train_total * 100

    print(f"\nTRAIN SET:")
    print(f"  UP: {train_up} ({train_up_pct:.2f}%)")
    print(f"  DOWN: {train_down} ({train_down_pct:.2f}%)")

    # Count UP/DOWN in test
    test_up = (y_test == 1).sum()
    test_down = (y_test == 0).sum()
    test_total = len(y_test)
    test_up_pct = test_up / test_total * 100
    test_down_pct = test_down / test_total * 100

    print(f"\nTEST SET:")
    print(f"  UP: {test_up} ({test_up_pct:.2f}%)")
    print(f"  DOWN: {test_down} ({test_down_pct:.2f}%)")

    # Conclusion
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if abs(train_up_pct - test_up_pct) < 5:
        print("✅ Label distribution is BALANCED between train and test.")
        print("   Bias UP is NOT due to imbalanced training data.")
    else:
        print("❌ Label distribution is IMBALANCED between train and test.")
        print(f"   Train UP: {train_up_pct:.2f}% vs Test UP: {test_up_pct:.2f}%")

    if train_up_pct > 60:
        print(f"⚠️  Train set has HIGH UP bias ({train_up_pct:.2f}%)")
        print("   Model may learn to predict UP more frequently.")
    else:
        print(f"✅ Train set has reasonable UP/DOWN ratio ({train_up_pct:.2f}% UP)")

    print("=" * 80)


if __name__ == "__main__":
    main()
