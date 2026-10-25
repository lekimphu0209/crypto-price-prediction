"""
BiLSTM Seed Ensemble with Ensemble Filtered on more folds.

Uses the extended dataset (data/btc_1d_extended.csv) so that raising
MIN_TRAIN_SIZE still leaves more usable folds than the original 1000-day run.

Output: experiments/ensemble_filtered_more_folds/results_bilstm_ensemble_filtered_more_folds.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.utils.determinism import set_full_determinism

EXTENDED_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "btc_1d_extended.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_SET = "B"
SEQ_LEN = 20
HORIZON = 7

# Walk-forward / ensemble settings
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50
MIN_TRAIN_SIZE = 300
N_FOLDS = 7        # train windows 200..500 => 5 filtered folds (>=300)
N_SEEDS = 5        # full ensemble size
EPOCHS = 15        # can be raised to 40 for final comparison


def main():
    print("=" * 80)
    print(f"ENSEMBLE FILTERED (more folds): BiLSTM {N_SEEDS} seeds, train >= {MIN_TRAIN_SIZE}")
    print(f"N_FOLDS={N_FOLDS}, TEST_SIZE={TEST_SIZE}, EPOCHS={EPOCHS}")
    print(f"Extended data: {EXTENDED_CSV}")
    print("=" * 80)

    df = pd.read_csv(EXTENDED_CSV, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Raw data shape: {df.shape}")

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    feat_df = extractor.extract(df)
    feature_names = extractor.feature_names()
    print(f"Features shape: {feat_df.shape}")

    values = feat_df[feature_names].values
    close = feat_df["close"].values

    X_raw, base_close, future_close = window.build(values, close)
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X_raw = X_raw[mask]
    y = y[mask]
    print(f"Windows shape: {X_raw.shape}, Target shape: {y.shape}")

    full_ensemble_results = []

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

        if test_end >= len(X_raw):
            print(f"\nFold {fold + 1}: Not enough data, skipping")
            break

        X_train_raw = X_raw[:train_end]
        y_train = y[:train_end]
        X_test_raw = X_raw[test_start:test_end]
        y_test = y[test_start:test_end]

        # Scale per fold: fit only on train, transform both train and test
        flat_train = X_train_raw.reshape(len(X_train_raw), -1)
        flat_test = X_test_raw.reshape(len(X_test_raw), -1)
        scaler = RobustScaler()
        scaler.fit(flat_train)
        X_train = scaler.transform(flat_train).reshape(X_train_raw.shape)
        X_test = scaler.transform(flat_test).reshape(X_test_raw.shape)

        print(f"\nFold {fold + 1}: Train={len(X_train)}, Test={len(X_test)}")
        print(f"Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        all_proba = np.zeros((N_SEEDS, len(X_test)))
        seed_balacc = []

        for seed in range(N_SEEDS):
            run_idx += 1
            set_full_determinism(seed)
            tf.keras.backend.clear_session()

            forecaster = KerasForecaster(
                name="BiLSTM",
                epochs=EPOCHS,
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

            y_pred_single = (proba >= 0.5).astype(int)
            balacc = balanced_accuracy_score(y_test, y_pred_single)
            seed_balacc.append(balacc)

            elapsed = time.time() - start
            print(f"  [{run_idx}/{total_runs}] seed{seed}: BalAcc={balacc:.4f} ({elapsed:.1f}s)")

        # Full ensemble
        mean_proba = all_proba.mean(axis=0)
        y_pred_full = (mean_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_full, labels=[0, 1]).ravel()
        full_metrics = {
            "fold": fold + 1,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, y_pred_full)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_full)),
            "precision": float(precision_score(y_test, y_pred_full, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_full, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_full, zero_division=0)),
            "mean_seed_balacc": float(np.mean(seed_balacc)),
            "std_seed_balacc": float(np.std(seed_balacc)),
            "min_seed_balacc": float(np.min(seed_balacc)),
            "max_seed_balacc": float(np.max(seed_balacc)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        full_ensemble_results.append(full_metrics)
        print(f"  -> FULL {N_SEEDS}-seed ensemble: {full_metrics['balanced_accuracy']:.4f} "
              f"(seeds {np.mean(seed_balacc):.4f} ± {np.std(seed_balacc):.4f})")

    out_path = os.path.join(OUT_DIR, "results_bilstm_ensemble_filtered_more_folds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "initial_train_size": INITIAL_TRAIN_SIZE,
                "test_size": TEST_SIZE,
                "min_train_size": MIN_TRAIN_SIZE,
                "n_folds": N_FOLDS,
                "n_seeds": N_SEEDS,
                "epochs": EPOCHS,
                "seq_len": SEQ_LEN,
                "horizon": HORIZON,
                "feature_set": FEATURE_SET,
            },
            "full_ensemble": full_ensemble_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if full_ensemble_results:
        balaccs = np.array([m["balanced_accuracy"] for m in full_ensemble_results])
        print("\n" + "=" * 80)
        print("T-TEST: Ensemble Filtered (more folds) vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n (filtered folds): {len(balaccs)}")
        print(f"  Mean BalAcc: {balaccs.mean():.4f} ± {balaccs.std():.4f}")
        print(f"  Min: {balaccs.min():.4f}, Max: {balaccs.max():.4f}")
        t, p = stats.ttest_1samp(balaccs, 0.5)
        print(f"  t-statistic: {t:.4f}")
        print(f"  p-value: {p:.4f}")
        print(f"  Significant (α=0.05): {'YES' if p < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
