"""
Walk-forward validation with 5 folds for BTCUSDT.

Instead of a single train/test split, this uses time-series walk-forward:
- Fold 1: Train[0:200], Test[200:250]
- Fold 2: Train[0:250], Test[250:300]
- Fold 3: Train[0:300], Test[300:350]
- Fold 4: Train[0:350], Test[350:400]
- Fold 5: Train[0:400], Test[400:450]

Reports mean/std of metrics across folds.

Usage:
    python scripts/train_btc_walkforward.py
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

from src.training.composition import build_training_pipeline, MODELS_DIR, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.utils.determinism import set_full_determinism

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200  # Initial training window size
TEST_SIZE = 50  # Test window size per fold
MODELS = ["RNN", "LSTM"]
SEEDS = list(range(10))


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )


def set_seed(seed: int):
    """Set full determinism for reproducible results."""
    set_full_determinism(seed)


def main():
    print("=" * 80)
    print(f"WALK-FORWARD VALIDATION: {N_FOLDS} FOLDS")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    # Load and prepare data
    from src.training.composition import build_training_pipeline
    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    print(f"Raw data shape: {df.shape}")

    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()
    print(f"Features shape: {feat_df.shape}")

    # Scale features
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])  # Fit on initial window
    feature_matrix = scaler.transform(values)

    # Build windows
    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)

    # Remove NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    print(f"Windows shape: {X.shape}, Target shape: {y.shape}")

    # Walk-forward folds
    results = []
    total_runs = len(MODELS) * len(SEEDS) * N_FOLDS
    run_idx = 0

    for model_name in MODELS:
        for seed in SEEDS:
            set_seed(seed)
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

                print(f"\n[{run_idx}/{total_runs}] {model_name} seed{seed} Fold{fold + 1}")
                print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

                # Count UP/DOWN in test
                test_up = (y_test == 1).sum()
                test_down = (y_test == 0).sum()
                print(f"  Test: UP={test_up}, DOWN={test_down}")

                forecaster = KerasForecaster(
                    name=model_name,
                    epochs=40,
                    batch_size=16,
                    units=32,
                    dropout=0.1,
                    learning_rate=0.001,
                    early_stopping_patience=0,  # Disable early stopping (no validation data)
                    validation_split=0.0,  # No validation split, use all for training
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
                    elapsed = time.time() - start
                    print(f"  -> {format_metrics(metrics)} ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"  -> ERROR: {e}")
                    fold_metrics.append(None)

            # Aggregate fold metrics
            valid_folds = [m for m in fold_metrics if m is not None]
            if valid_folds:
                agg = {
                    "model": model_name,
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
                    "fold_metrics": valid_folds,  # Keep detailed per-fold metrics
                }
                results.append(agg)
                print(f"\n  Aggregated: Acc={agg['mean_accuracy']:.4f}±{agg['std_accuracy']:.4f} "
                      f"BalAcc={agg['mean_balanced_accuracy']:.4f}±{agg['std_balanced_accuracy']:.4f}")

    # Save results with detailed fold metrics
    out_path = os.path.join(MODELS_DIR, "btc_walkforward_results_detailed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Statistical significance on all fold values (not mean-per-seed only)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from analyze_walkforward_ttest import run_ttest_report, print_report

        models = sorted({r["model"] for r in results})
        ttest_report = run_ttest_report(results, models=models)
        ttest_path = os.path.join(MODELS_DIR, "btc_walkforward_ttest_results.json")
        with open(ttest_path, "w", encoding="utf-8") as f:
            json.dump(ttest_report, f, indent=2)
        print(f"T-test report saved to: {ttest_path}")
        print_report(ttest_report)
    except Exception as e:
        print(f"T-test analysis skipped: {e}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY (Mean ± Std across folds)")
        print("=" * 80)
        for r in results:
            print(f"{r['model']} seed{r['seed']:d}: "
                  f"Acc={r['mean_accuracy']:.4f}±{r['std_accuracy']:.4f} "
                  f"BalAcc={r['mean_balanced_accuracy']:.4f}±{r['std_balanced_accuracy']:.4f}")

        print("\nBest by Balanced Accuracy:")
        best = max(results, key=lambda x: x["mean_balanced_accuracy"])
        print(f"  {best['model']} seed{best['seed']}: "
              f"BalAcc={best['mean_balanced_accuracy']:.4f}±{best['std_balanced_accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
