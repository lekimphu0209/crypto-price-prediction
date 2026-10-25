"""
Walk-forward validation with BiLSTM validation-weighted seed ensemble.

For each fold:
- Split train into train/val (last 10% of train as validation, time-based).
- Train N_SEEDS BiLSTM models with different seeds.
- Compute each seed's validation BalAcc.
- Weight test predictions by validation BalAcc (or use top-K).
- Threshold weighted average probability at 0.5.

This reduces seed variance by prioritizing seeds that generalize on the val window.
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
SEED_START = 0
VAL_RATIO = 0.1


def main():
    print("=" * 80)
    print(f"WALK-FORWARD: BiLSTM VALIDATION-WEIGHTED SEED ENSEMBLE "
          f"({N_FOLDS} FOLDS, {N_SEEDS} seeds/fold)")
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

        # time-based validation split: last 10% of train_all
        val_size = max(1, int(len(X_train_all) * VAL_RATIO))
        X_train, X_val = X_train_all[:-val_size], X_train_all[-val_size:]
        y_train, y_val = y_train_all[:-val_size], y_train_all[-val_size:]

        print(f"\nFold {fold + 1}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        all_test_proba = np.zeros((N_SEEDS, len(X_test)))
        val_balacc = np.zeros(N_SEEDS)

        for seed_offset in range(N_SEEDS):
            seed = SEED_START + seed_offset
            run_idx += 1
            set_full_determinism(seed)
            tf.keras.backend.clear_session()

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
            forecaster.fit(X_train, y_train)
            val_proba = np.asarray(forecaster.predict(X_val))
            val_pred = (val_proba >= 0.5).astype(int)
            val_balacc[seed_offset] = balanced_accuracy_score(y_val, val_pred)

            test_proba = np.asarray(forecaster.predict(X_test))
            all_test_proba[seed_offset] = test_proba
            elapsed = time.time() - start
            print(f"  [{run_idx}/{total_runs}] seed{seed}: val_BalAcc={val_balacc[seed_offset]:.4f} ({elapsed:.1f}s)")

        # weight predictions by val BalAcc, shift to [0, 1]
        weights = val_balacc - val_balacc.min() + 1e-6
        weights = weights / weights.sum()
        weighted_proba = np.average(all_test_proba, axis=0, weights=weights)
        y_pred = (weighted_proba >= 0.5).astype(int)

        # top-K (3) ensemble
        top_k = 3
        top_idx = np.argsort(val_balacc)[-top_k:]
        top_proba = all_test_proba[top_idx].mean(axis=0)
        y_pred_topk = (top_proba >= 0.5).astype(int)

        # unweighted mean ensemble
        mean_proba = all_test_proba.mean(axis=0)
        y_pred_mean = (mean_proba >= 0.5).astype(int)

        for name, y_pred_ in [
            ("weighted", y_pred),
            ("top3", y_pred_topk),
            ("mean", y_pred_mean),
        ]:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_, labels=[0, 1]).ravel()
            metrics = {
                "fold": fold + 1,
                "ensemble": name,
                "accuracy": float(accuracy_score(y_test, y_pred_)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_)),
                "precision": float(precision_score(y_test, y_pred_, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred_, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "val_balacc_mean": float(val_balacc.mean()),
                "val_balacc_std": float(val_balacc.std()),
                "weights": weights.tolist(),
                "top3_indices": top_idx.tolist(),
                "val_balacc": val_balacc.tolist(),
            }
            fold_metrics.append(metrics)
            print(f"  -> {name}: Acc={metrics['accuracy']:.4f} "
                  f"BalAcc={metrics['balanced_accuracy']:.4f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    out_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_ensemble_weighted_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 80)
    print("T-TEST: Ensemble vs Baseline (0.5)")
    print("=" * 80)
    for name in ["weighted", "top3", "mean"]:
        vals = np.array([m["balanced_accuracy"] for m in fold_metrics if m["ensemble"] == name])
        t, p = stats.ttest_1samp(vals, 0.5)
        print(f"  {name}: mean={vals.mean():.4f} std={vals.std():.4f} t={t:.4f} p={p:.4f}")


if __name__ == "__main__":
    main()
