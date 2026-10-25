"""
Walk-forward validation with BiLSTM using stable training config.

Near-original capacity, only adding checkpoint + reduce_lr for stability:
- units=64
- dropout=0.15
- recurrent_dropout=0.1
- ReduceLROnPlateau
- ModelCheckpoint(best val_loss)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
SEEDS = list(range(10))


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )


def main():
    print("=" * 80)
    print(f"WALK-FORWARD: BiLSTM STABLE ({N_FOLDS} FOLDS, {len(SEEDS)} SEEDS)")
    print("=" * 80)
    print("Config: units=64, dropout=0.15, recurrent_dropout=0.1, checkpoint, fixed LR")

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

    results = []
    total_runs = len(SEEDS) * N_FOLDS
    run_idx = 0

    all_balacc = []

    for seed in SEEDS:
        set_full_determinism(seed)
        fold_metrics = []

        for fold in range(N_FOLDS):
            run_idx += 1
            train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
            test_start = train_end
            test_end = test_start + TEST_SIZE

            if test_end >= len(X):
                print(f"  Fold {fold + 1}: Not enough data, skipping")
                break

            X_train, X_test = X[:train_end], X[test_start:test_end]
            y_train, y_test = y[:train_end], y[test_start:test_end]

            print(f"\n[{run_idx}/{total_runs}] BiLSTM-stable seed{seed} Fold{fold + 1}")
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

            test_up = (y_test == 1).sum()
            test_down = (y_test == 0).sum()
            print(f"  Test: UP={test_up}, DOWN={test_down}")

            forecaster = KerasForecaster(
                name="BiLSTM",
                epochs=50,
                batch_size=16,
                units=64,
                dropout=0.15,
                recurrent_dropout=0.1,
                l2_reg=0.0,
                learning_rate=0.001,
                early_stopping_patience=10,
                reduce_lr_patience=0,
                reduce_lr_factor=0.5,
                checkpoint_best=True,
                validation_split=0.1,
                mode="classification",
            )

            start = time.time()
            try:
                forecaster.fit(X_train, y_train)
                proba = np.asarray(forecaster.predict(X_test))
                y_pred = (proba >= 0.5).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                metrics = {
                    "fold": fold + 1,
                    "seed": seed,
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
                fold_metrics.append(metrics)
                all_balacc.append(metrics["balanced_accuracy"])
                elapsed = time.time() - start
                print(f"  -> {format_metrics(metrics)} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                print(f"  -> ERROR: {e}")
                fold_metrics.append(None)

        valid_folds = [m for m in fold_metrics if m is not None]
        if valid_folds:
            agg = {
                "model": "BiLSTM_stable",
                "seed": seed,
                "n_folds": len(valid_folds),
                "mean_accuracy": float(np.mean([m["accuracy"] for m in valid_folds])),
                "std_accuracy": float(np.std([m["accuracy"] for m in valid_folds])),
                "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in valid_folds])),
                "std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in valid_folds])),
                "mean_precision": float(np.mean([m["precision"] for m in valid_folds])),
                "std_precision": float(np.std([m["precision"] for m in valid_folds])),
                "mean_recall": float(np.mean([m["recall"] for m in valid_folds])),
                "std_recall": float(np.std([m["recall"] for m in valid_folds])),
                "mean_f1": float(np.mean([m["f1"] for m in valid_folds])),
                "std_f1": float(np.std([m["f1"] for m in valid_folds])),
                "fold_metrics": valid_folds,
            }
            results.append(agg)
            print(f"\n  Aggregated: Acc={agg['mean_accuracy']:.4f}±{agg['std_accuracy']:.4f} "
                  f"BalAcc={agg['mean_balanced_accuracy']:.4f}±{agg['std_balanced_accuracy']:.4f}")

    out_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_stable_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if all_balacc:
        all_balacc = np.array(all_balacc)
        print("\n" + "=" * 80)
        print("T-TEST: BiLSTM-stable vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n: {len(all_balacc)}")
        print(f"  Mean: {all_balacc.mean():.4f}")
        print(f"  Std: {all_balacc.std():.4f}")
        t_stat, p_value = stats.ttest_1samp(all_balacc, 0.5)
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (α=0.05): {'YES' if p_value < 0.05 else 'NO'}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if results:
        all_mean_acc = [r["mean_accuracy"] for r in results]
        all_mean_balacc = [r["mean_balanced_accuracy"] for r in results]
        print(f"BiLSTM-stable: Acc={np.mean(all_mean_acc):.4f}±{np.std(all_mean_acc):.4f} "
              f"BalAcc={np.mean(all_mean_balacc):.4f}±{np.std(all_mean_balacc):.4f}")


if __name__ == "__main__":
    main()
