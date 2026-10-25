"""
Expanding-window retrain demo for Logistic Regression.

Moi fold:
- train_end = INITIAL_TRAIN + fold * STEP
- fit RobustScaler tren toan bo train data hien tai
- fit LinearForecaster tu dau tren train data
- du doan STEP mau tiep theo
Ket qua duoc luu vao results.json.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy import stats

from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
INITIAL_TRAIN_SIZE = 300
TEST_SIZE = 50
N_FOLDS = 5


def main():
    print("=" * 80)
    print("EXPANDING WINDOW RETRAIN DEMO - Logistic Regression")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    print(f"Raw data shape: {df.shape}")

    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()
    print(f"Features shape: {feat_df.shape}")

    values = feat_df[feature_names].values
    close = feat_df["close"].values

    # Build full windows/targets once (raw features)
    scaler_init = RobustScaler()
    scaler_init.fit(values[:INITIAL_TRAIN_SIZE])
    feature_matrix_init = scaler_init.transform(values)
    X_init, base_close, future_close = window.build(feature_matrix_init, close)
    y_init = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y_init)
    # Use index for splitting; actual scaling will be refit per fold
    X_raw = window.build(scaler_init.transform(values), close)[1]  # placeholder
    # Build raw windows directly without scaling to apply fold-specific scaling later
    X_raw, base_close, future_close = window.build(values, close)  # (samples, seq, feats) from raw
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X_raw = X_raw[mask]
    y = y[mask]
    print(f"Windows shape: {X_raw.shape}, Target shape: {y.shape}")

    results = []

    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if test_end >= len(X_raw):
            print(f"\nFold {fold + 1}: not enough data, stop")
            break

        print(f"\nFold {fold + 1}: train[0:{train_end}], test[{test_start}:{test_end}]")

        # Extract raw windows for this fold and refit scaler on train only
        X_train_raw = X_raw[:train_end]
        y_train = y[:train_end]
        X_test_raw = X_raw[test_start:test_end]
        y_test = y[test_start:test_end]

        # Flatten raw windows to (samples, seq*features) for scaler
        flat_train = X_train_raw.reshape(len(X_train_raw), -1)
        flat_test = X_test_raw.reshape(len(X_test_raw), -1)
        scaler = RobustScaler()
        scaler.fit(flat_train)
        X_train_flat = scaler.transform(flat_train)
        X_test_flat = scaler.transform(flat_test)

        # Reshape back to 3D for LinearForecaster (it expects 3D then flattens)
        X_train = X_train_flat.reshape(X_train_raw.shape)
        X_test = X_test_flat.reshape(X_test_raw.shape)

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        forecaster = LinearForecaster(seed=None, C=1.0, class_weight="balanced", max_iter=1000)

        start = time.time()
        forecaster.fit(X_train, y_train)
        proba = np.asarray(forecaster.predict(X_test))
        y_pred = (proba >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        metrics = {
            "fold": fold + 1,
            "train_end": int(train_end),
            "test_start": int(test_start),
            "test_end": int(test_end),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        results.append(metrics)
        elapsed = time.time() - start
        print(f"  -> Acc={metrics['accuracy']:.4f} BalAcc={metrics['balanced_accuracy']:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} ({elapsed:.1f}s)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if results:
        balacc_vals = np.array([m["balanced_accuracy"] for m in results])
        print("\n" + "=" * 80)
        print("T-TEST: Expanding Window Logistic vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n (folds): {len(balacc_vals)}")
        print(f"  Mean: {balacc_vals.mean():.4f}")
        print(f"  Std: {balacc_vals.std():.4f}")
        t_stat, p_value = stats.ttest_1samp(balacc_vals, 0.5)
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (alpha=0.05): {'YES' if p_value < 0.05 else 'NO'}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Expanding Window Logistic: Acc={np.mean([m['accuracy'] for m in results]):.4f} "
              f"BalAcc={balacc_vals.mean():.4f} ± {balacc_vals.std():.4f}")


if __name__ == "__main__":
    main()
