"""
Parallel Walk-forward validation with 5 folds for BTCUSDT using ProcessPoolExecutor.
Optimized for multi-core CPUs.
"""
import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit TensorFlow threading to avoid contention during parallel runs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.training.composition import build_training_pipeline, MODELS_DIR, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200  # Initial training window size
TEST_SIZE = 50  # Test window size per fold
MODELS = ["RNN", "LSTM"]
SEEDS = list(range(10))  # 10 seeds (0 to 9)


def run_model_seed_wrapper(args):
    """Wrapper function to unpack arguments in child process."""
    return run_model_seed(*args)


def run_model_seed(model_name, seed, X, y, INITIAL_TRAIN_SIZE, TEST_SIZE, N_FOLDS):
    # Set seed inside process
    from src.utils.determinism import set_full_determinism
    set_full_determinism(seed)
    
    # Imports must be inside the process to avoid TensorFlow issues
    import os
    # Ensure TF threads are restricted inside the process too
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
    
    fold_metrics = []
    
    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE
        
        if test_end >= len(X):
            break
            
        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]
        
        forecaster = KerasForecaster(
            name=model_name,
            epochs=40,
            batch_size=16,
            units=32,
            dropout=0.1,
            learning_rate=0.001,
            early_stopping_patience=0,
            validation_split=0.0,
            mode="classification",
        )
        
        try:
            start_time = time.time()
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
            duration = time.time() - start_time
            print(f"  [Worker] {model_name} seed{seed} Fold{fold + 1} completed in {duration:.1f}s - BalAcc={metrics['balanced_accuracy']:.4f}")
        except Exception as e:
            print(f"  [Worker] ERROR in {model_name} seed{seed} Fold{fold + 1}: {e}")
            fold_metrics.append(None)
            
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
            "fold_metrics": valid_folds,
        }
        return agg
    return None


def main():
    print("=" * 80)
    print(f"PARALLEL WALK-FORWARD VALIDATION: {N_FOLDS} FOLDS")
    print(f"Models: {MODELS}, Seeds: {SEEDS}")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()

    # Scale features
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
    feature_matrix = scaler.transform(values)

    # Build windows
    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)

    # Remove NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    print(f"Windows shape: {X.shape}, Target shape: {y.shape}")

    # Prepare parallel execution tasks
    tasks = []
    for model_name in MODELS:
        for seed in SEEDS:
            tasks.append((model_name, seed, X, y, INITIAL_TRAIN_SIZE, TEST_SIZE, N_FOLDS))

    print(f"Starting ProcessPoolExecutor with 4 workers for {len(tasks)} tasks...")
    results = []
    
    start_total = time.time()
    # Use 4 workers to run seeds in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_model_seed_wrapper, task): (task[0], task[1]) for task in tasks}
        
        for future in as_completed(futures):
            model_name, seed = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
                    print(f"✓ Completed task for {model_name} seed{seed} - Mean BalAcc: {res['mean_balanced_accuracy']:.4f}")
            except Exception as e:
                print(f"✗ Failed task for {model_name} seed{seed}: {e}")

    total_duration = time.time() - start_total
    print(f"\nAll parallel runs completed in {total_duration:.1f}s.")

    # Save results
    out_path = os.path.join(MODELS_DIR, "btc_walkforward_results_detailed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    # Run t-test
    try:
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

    print("Done.")


if __name__ == "__main__":
    main()
