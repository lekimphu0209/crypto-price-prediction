"""
Walk-forward validation with BiLSTM seed ensemble + bagging.

For each fold:
- Generate N_SEEDS bootstrap samples of training data (different seed each).
- Train a BiLSTM on each sample.
- Average predicted probabilities on the test set.
- Threshold at 0.5.

Bagging increases diversity between ensemble members, reducing variance.
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
N_SEEDS = 10
BAG_FRAC = 0.85


def bootstrap_sample(X, y, rng, frac=BAG_FRAC):
    n = len(X)
    size = max(1, int(n * frac))
    idx = rng.choice(n, size=size, replace=True)
    return X[idx], y[idx]


def main():
    print("=" * 80)
    print(f"WALK-FORWARD: BiLSTM BAGGING ENSEMBLE ({N_FOLDS} FOLDS, {N_SEEDS} bags)")
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

    fold_metrics = []
    total_runs = N_FOLDS * N_SEEDS
    run_idx = 0

    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if test_end >= len(X):
            print(f"Fold {fold + 1}: Not enough data, skipping")
            break

        X_train_all, X_test = X[:train_end], X[test_start:test_end]
        y_train_all, y_test = y[:train_end], y[test_start:test_end]

        print(f"\nFold {fold + 1}: Train={len(X_train_all)}, Test={len(X_test)}")
        print(f"Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        all_test_proba = np.zeros((N_SEEDS, len(X_test)))
        seed_balacc = []

        for seed in range(N_SEEDS):
            run_idx += 1
            set_full_determinism(seed)
            tf.keras.backend.clear_session()

            # bootstrap sample with numpy random for this seed
            rng = np.random.RandomState(seed)
            X_train_bag, y_train_bag = bootstrap_sample(X_train_all, y_train_all, rng)

            forecaster = KerasForecaster(
                name="BiLSTM",
                epochs=40,
                batch_size=16,
                units=32,
                dropout=0.1,
                rnn_dropout=0.0,
                recurrent_dropout=0.0,
                l2_reg=0.0,
                learning_rate=0.001,
                early_stopping_patience=0,
                reduce_lr_patience=0,
                checkpoint_best=False,
                validation_split=0.0,
                mode="classification",
            )

            start = time.time()
            forecaster.fit(X_train_bag, y_train_bag)
            proba = np.asarray(forecaster.predict(X_test))
            all_test_proba[seed] = proba

            y_pred_single = (proba >= 0.5).astype(int)
            balacc = balanced_accuracy_score(y_test, y_pred_single)
            seed_balacc.append(balacc)
            elapsed = time.time() - start
            print(f"  [{run_idx}/{total_runs}] seed{seed}: BalAcc={balacc:.4f} ({elapsed:.1f}s)")

        # ensemble: average probability
        mean_proba = all_test_proba.mean(axis=0)
        y_pred_ensemble = (mean_proba >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble, labels=[0, 1]).ravel()
        metrics = {
            "fold": fold + 1,
            "accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_ensemble)),
            "precision": float(precision_score(y_test, y_pred_ensemble, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_ensemble, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_ensemble, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "mean_seed_balacc": float(np.mean(seed_balacc)),
            "std_seed_balacc": float(np.std(seed_balacc)),
            "min_seed_balacc": float(np.min(seed_balacc)),
            "max_seed_balacc": float(np.max(seed_balacc)),
        }
        fold_metrics.append(metrics)
        print(f"  -> BAGGING ENSEMBLE: Acc={metrics['accuracy']:.4f} "
              f"BalAcc={metrics['balanced_accuracy']:.4f} "
              f"(seed mean ± std: {metrics['mean_seed_balacc']:.4f} ± {metrics['std_seed_balacc']:.4f})")

    out_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_bagging_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if fold_metrics:
        balacc_vals = np.array([m["balanced_accuracy"] for m in fold_metrics])
        print("\n" + "=" * 80)
        print("T-TEST: BiLSTM Bagging Ensemble vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n (folds): {len(balacc_vals)}")
        print(f"  Mean: {balacc_vals.mean():.4f}")
        print(f"  Std: {balacc_vals.std():.4f}")
        t, p = stats.ttest_1samp(balacc_vals, 0.5)
        print(f"  t: {t:.4f}, p: {p:.4f}")
        print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
