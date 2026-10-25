"""
Variance analysis test for walk-forward results.

Checks:
- std_between_folds vs std_between_seeds
- observed std vs theoretical SE (n=50)
- std between seeds
- std by fold size (small vs large train windows)
- feature stats by fold
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import RobustScaler
from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON
from src.features.extractors.technical import TechnicalExtractor
from src.training.targets.trend_target import TrendTargetBuilder
from src.training.datasets.window_dataset import WindowDataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_FILE = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")
SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def extract_fold_matrix(results):
    """Return matrix (n_seeds, n_folds) of balanced_accuracy."""
    return np.array([[f["balanced_accuracy"] for f in r["fold_metrics"]] for r in results])


def variance_analysis():
    print("=" * 80)
    print("VARIANCE ANALYSIS: BiLSTM walk-forward (10 seeds x 5 folds)")
    print("=" * 80)

    results = load_results()
    M = extract_fold_matrix(results)  # (10, 5)
    n_seeds, n_folds = M.shape
    all_vals = M.ravel()

    # 1. std_between_folds vs std_between_seeds
    mean_per_seed = M.mean(axis=1)
    mean_per_fold = M.mean(axis=0)

    std_between_seeds = np.std(mean_per_seed, ddof=1)
    std_between_folds = np.std(mean_per_fold, ddof=1)

    print("\n1. STD BETWEEN SEEDS vs STD BETWEEN FOLDS")
    print(f"  std_between_seeds:  {std_between_seeds:.4f}")
    print(f"  std_between_folds:  {std_between_folds:.4f}")
    print(f"  ratio (folds/seeds): {std_between_folds / std_between_seeds:.3f}")

    # 2. observed std vs theoretical SE (n=50)
    observed_std = np.std(all_vals, ddof=1)
    observed_se = observed_std / np.sqrt(len(all_vals))
    theoretical_se = observed_std / np.sqrt(50)  # same as n=50

    print("\n2. OBSERVED STD vs THEORETICAL SE (n=50)")
    print(f"  observed std:     {observed_std:.4f}")
    print(f"  observed SE:      {observed_se:.4f}")
    print(f"  theoretical SE:   {theoretical_se:.4f}")
    print(f"  mean:             {np.mean(all_vals):.4f}")
    print(f"  95% CI (mean ± 1.96*SE): [{np.mean(all_vals) - 1.96*observed_se:.4f}, {np.mean(all_vals) + 1.96*observed_se:.4f}]")

    # 3. std between seeds (per fold and overall)
    std_per_fold = np.std(M, axis=0, ddof=1)
    print("\n3. STD BETWEEN SEEDS (per fold)")
    for i, s in enumerate(std_per_fold, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean across folds: {np.mean(std_per_fold):.4f}")

    # 4. std by fold size (small vs large train windows)
    print("\n4. STD BY FOLD SIZE (train window size)")
    small_folds = [0, 1]  # Fold 1 (train=200), Fold 2 (train=250)
    large_folds = [3, 4]  # Fold 4 (train=350), Fold 5 (train=400)
    small_vals = M[:, small_folds].ravel()
    large_vals = M[:, large_folds].ravel()
    print(f"  Small train windows (folds 1-2): {np.std(small_vals, ddof=1):.4f}")
    print(f"  Large train windows (folds 4-5): {np.std(large_vals, ddof=1):.4f}")
    print(f"  Mean small: {np.mean(small_vals):.4f}")
    print(f"  Mean large: {np.mean(large_vals):.4f}")

    # 5. Feature stats by fold
    print("\n5. FEATURE STATS BY FOLD")
    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)
    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
    feature_matrix = scaler.transform(values)
    X, base_close, future_close = window.build(feature_matrix, close)
    y = TrendTargetBuilder(threshold=0.0).build(base_close, future_close)
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask].astype(int)

    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        X_train = X[:train_end]
        feat_flat = X_train.reshape(-1, X_train.shape[-1])
        print(f"\n  Fold {fold+1} (train_end={train_end}):")
        print(f"    n_samples: {len(X_train)}")
        print(f"    mean feature: {feat_flat.mean():.4f}")
        print(f"    std feature:  {feat_flat.std():.4f}")
        print(f"    min feature:  {feat_flat.min():.4f}")
        print(f"    max feature:  {feat_flat.max():.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(all_vals)}")
    print(f"Overall mean BalAcc: {np.mean(all_vals):.4f}")
    print(f"Overall std BalAcc:  {observed_std:.4f}")
    print(f"Between-seeds std:   {std_between_seeds:.4f}")
    print(f"Between-folds std:   {std_between_folds:.4f}")


def main():
    variance_analysis()


if __name__ == "__main__":
    main()
