"""
Walk-forward: LSTM seed ensemble with 20 seeds on large train windows only.

Fold with train window < MIN_TRAIN_SIZE are skipped (small windows inflate
variance and are not representative of production where full history is used).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
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

from src.training.composition import build_training_pipeline, MODELS_DIR, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.utils.determinism import set_full_determinism

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50
MIN_TRAIN_SIZE = 300  # skip folds with train window < 300
N_SEEDS = 20
GROUP_SIZES = [5, 4]


def main():
    print("=" * 80)
    print(f"WALK-FORWARD: LSTM SEED ENSEMBLE {N_SEEDS} SEEDS (train >= {MIN_TRAIN_SIZE})")
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
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
    feature_matrix = scaler.transform(values)

    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)

    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    print(f"Windows shape: {X.shape}, Target shape: {y.shape}")

    full_ensemble_results = []
    group_results = {g: [] for g in GROUP_SIZES}

    total_runs = 0
    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        if train_end >= MIN_TRAIN_SIZE:
            total_runs += N_SEEDS

    run_idx = 0

    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if train_end < MIN_TRAIN_SIZE:
            print(f"\nFold {fold + 1}: train window = {train_end} < {MIN_TRAIN_SIZE}, skipping")
            continue

        if test_end >= len(X):
            print(f"Fold {fold + 1}: Not enough data, skipping")
            break

        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]

        print(f"\nFold {fold + 1}: Train={len(X_train)}, Test={len(X_test)}")
        print(f"Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        all_proba = np.zeros((N_SEEDS, len(X_test)))

        for seed in range(N_SEEDS):
            run_idx += 1
            set_full_determinism(seed)
            tf.keras.backend.clear_session()

            forecaster = KerasForecaster(
                name="LSTM",
                epochs=40,
                batch_size=16,
                units=32,
                dropout=0.1,
                rnn_dropout=0.0,
                recurrent_dropout=0.0,
                l2_reg=0.0,
                learning_rate=0.001,
                early_stopping_patience=0,
                validation_split=0.0,
                mode="classification",
            )

            start = time.time()
            forecaster.fit(X_train, y_train)
            proba = np.asarray(forecaster.predict(X_test))
            all_proba[seed] = proba
            elapsed = time.time() - start
            print(f"  [{run_idx}/{total_runs}] seed{seed}: ({elapsed:.1f}s)")

        # Full ensemble
        mean_proba = all_proba.mean(axis=0)
        y_pred_full = (mean_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_full, labels=[0, 1]).ravel()
        full_metrics = {
            "fold": fold + 1,
            "group_size": N_SEEDS,
            "group_index": 0,
            "accuracy": float(accuracy_score(y_test, y_pred_full)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_full)),
            "precision": float(precision_score(y_test, y_pred_full, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_full, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_full, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        full_ensemble_results.append(full_metrics)
        print(f"  -> FULL {N_SEEDS}-seed ensemble: BalAcc={full_metrics['balanced_accuracy']:.4f}")

        # Group ensembles
        for group_size in GROUP_SIZES:
            if N_SEEDS % group_size != 0:
                continue
            n_groups = N_SEEDS // group_size
            for g in range(n_groups):
                g_proba = all_proba[g * group_size:(g + 1) * group_size].mean(axis=0)
                y_pred_g = (g_proba >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_g, labels=[0, 1]).ravel()
                m = {
                    "fold": fold + 1,
                    "group_size": group_size,
                    "group_index": g,
                    "accuracy": float(accuracy_score(y_test, y_pred_g)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_g)),
                    "precision": float(precision_score(y_test, y_pred_g, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred_g, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred_g, zero_division=0)),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                }
                group_results[group_size].append(m)

    out_path = os.path.join(MODELS_DIR, "btc_walkforward_lstm_ensemble_filtered_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "min_train_size": MIN_TRAIN_SIZE,
            "full_ensemble": full_ensemble_results,
            "group_ensembles": group_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 80)
    print("T-TEST: Ensemble vs Baseline (0.5)")
    print("=" * 80)

    for name, vals in [("full 20-seed", full_ensemble_results)] + [(f"group-{g}", group_results[g]) for g in GROUP_SIZES]:
        balaccs = np.array([m["balanced_accuracy"] for m in vals])
        if len(balaccs) == 0:
            continue
        t, p = stats.ttest_1samp(balaccs, 0.5)
        print(f"  {name}: n={len(balaccs)}, mean={balaccs.mean():.4f}, std={balaccs.std():.4f}, t={t:.4f}, p={p:.4f}")


if __name__ == "__main__":
    main()
