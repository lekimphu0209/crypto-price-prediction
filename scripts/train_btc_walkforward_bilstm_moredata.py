"""
Walk-forward validation with BiLSTM using larger training windows.

Original: INITIAL_TRAIN_SIZE=200, 5 folds, TEST_SIZE=50
This script: INITIAL_TRAIN_SIZE=400, 5 folds, TEST_SIZE=50

This tests whether more training data reduces variance and improves BiLSTM.
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
INITIAL_TRAIN_SIZE = 400
TEST_SIZE = 50
SEEDS = list(range(10))


def main():
    print("=" * 80)
    print(f"WALK-FORWARD VALIDATION: BiLSTM MORE DATA ({N_FOLDS} FOLDS, {len(SEEDS)} SEEDS)")
    print(f"Initial train size: {INITIAL_TRAIN_SIZE}, Test size: {TEST_SIZE}")
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

    results = []
    total_runs = len(SEEDS) * N_FOLDS
    run_idx = 0

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

            print(f"\n[{run_idx}/{total_runs}] BiLSTM seed{seed} Fold{fold + 1}")
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

            test_up = (y_test == 1).sum()
            test_down = (y_test == 0).sum()
            print(f"  Test: UP={test_up}, DOWN={test_down}")

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
                validation_split=0.0,
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
                print(f"  -> {metrics}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if fold_metrics:
            accs = [m["accuracy"] for m in fold_metrics]
            balaccs = [m["balanced_accuracy"] for m in fold_metrics]
            precs = [m["precision"] for m in fold_metrics]
            recs = [m["recall"] for m in fold_metrics]
            f1s = [m["f1"] for m in fold_metrics]
            results.append({
                "model": "BiLSTM",
                "seed": seed,
                "n_folds": len(fold_metrics),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_balanced_accuracy": float(np.mean(balaccs)),
                "std_balanced_accuracy": float(np.std(balaccs)),
                "mean_precision": float(np.mean(precs)),
                "std_precision": float(np.std(precs)),
                "mean_recall": float(np.mean(recs)),
                "std_recall": float(np.std(recs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "fold_metrics": fold_metrics,
            })

    out_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_moredata_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if results:
        all_balacc = [f["balanced_accuracy"] for r in results for f in r["fold_metrics"]]
        all_balacc = np.array(all_balacc)
        print("\n" + "=" * 80)
        print("T-TEST: BiLSTM More Data vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n: {len(all_balacc)}")
        print(f"  Mean BalAcc: {all_balacc.mean():.4f}")
        print(f"  Std BalAcc: {all_balacc.std():.4f}")
        t, p = stats.ttest_1samp(all_balacc, 0.5)
        print(f"  t: {t:.4f}, p: {p:.4f}")
        print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
