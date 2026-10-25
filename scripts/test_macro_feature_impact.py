"""
Kiểm tra tác động của macro features (gold, DXY, interest rate)
đến balanced accuracy trên walk-forward validation với Logistic Regression.

So sánh 2 feature set:
  - Feature Set B (có macro): gold, dxy, usd_rate, gold_change, dxy_change, rate_change
  - Feature Set B không macro: exclude toàn bộ macro features

Dùng 3 seeds x 5 folds để nhanh.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score

from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster
from src.utils.determinism import set_full_determinism

SYMBOL = "BTCUSDT"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50
SEEDS = [0, 1, 2, 3, 4]

MACRO_FEATURES = ["gold", "dxy", "usd_rate", "gold_change", "dxy_change", "rate_change"]


def run_walkforward(feature_names, values, close, label):
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
    feature_matrix = scaler.transform(values)

    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)
    target_builder = TrendTargetBuilder()
    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X, y = X[mask], y[mask]

    all_balacc = []
    for seed in SEEDS:
        set_full_determinism(seed)
        for fold in range(N_FOLDS):
            train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
            test_start = train_end
            test_end = test_start + TEST_SIZE
            if test_end >= len(X):
                break
            X_train, X_test = X[:train_end], X[test_start:test_end]
            y_train, y_test = y[:train_end], y[test_start:test_end]

            try:
                forecaster = LinearForecaster(seed=seed, C=1.0, class_weight="balanced", max_iter=1000)
                forecaster.fit(X_train, y_train)
                proba = np.asarray(forecaster.predict(X_test))
                y_pred = (proba >= 0.5).astype(int)
                balacc = balanced_accuracy_score(y_test, y_pred)
                all_balacc.append(balacc)
            except Exception as e:
                print(f"  ERROR seed={seed} fold={fold+1}: {e}")

    arr = np.array(all_balacc)
    return arr


def main():
    print("=" * 70)
    print("MACRO FEATURE IMPACT TEST (Gold / DXY / Interest Rate)")
    print("Logistic Regression, Feature Set B, 5 seeds x 5 folds")
    print("=" * 70)

    # Load data once
    extractor_with = TechnicalExtractor(feature_set="B")
    pipeline = build_training_pipeline(extractor=extractor_with, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    print(f"Data loaded: {df.shape}")

    # --- Feature set WITH macro ---
    print("\n[1/2] Extracting features WITH macro (gold, dxy, rate)...")
    feat_with = extractor_with.extract(df)
    feature_names_with = extractor_with.feature_names()
    values_with = feat_with[feature_names_with].values
    close = feat_with["close"].values
    print(f"  Features: {len(feature_names_with)} -> {feature_names_with}")

    t0 = time.time()
    balacc_with = run_walkforward(feature_names_with, values_with, close, "WITH macro")
    print(f"  Done in {time.time()-t0:.1f}s | n={len(balacc_with)} | Mean={balacc_with.mean():.4f} ± {balacc_with.std():.4f}")

    # --- Feature set WITHOUT macro ---
    print("\n[2/2] Extracting features WITHOUT macro...")
    extractor_without = TechnicalExtractor(feature_set="B", exclude_features=MACRO_FEATURES)
    feat_without = extractor_without.extract(df)
    feature_names_without = extractor_without.feature_names()
    values_without = feat_without[feature_names_without].values
    print(f"  Features: {len(feature_names_without)} -> {feature_names_without}")

    t0 = time.time()
    balacc_without = run_walkforward(feature_names_without, values_without, close, "WITHOUT macro")
    print(f"  Done in {time.time()-t0:.1f}s | n={len(balacc_without)} | Mean={balacc_without.mean():.4f} ± {balacc_without.std():.4f}")

    # --- So sánh ---
    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH")
    print("=" * 70)
    print(f"  WITH macro:    n={len(balacc_with):2d}, Mean={balacc_with.mean():.4f} ± {balacc_with.std():.4f}")
    print(f"  WITHOUT macro: n={len(balacc_without):2d}, Mean={balacc_without.mean():.4f} ± {balacc_without.std():.4f}")
    diff = balacc_with.mean() - balacc_without.mean()
    print(f"  Difference (with - without): {diff:+.4f}")

    # Paired t-test
    n = min(len(balacc_with), len(balacc_without))
    t_stat, p_val = stats.ttest_rel(balacc_with[:n], balacc_without[:n])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"\n  Paired t-test: t={t_stat:.4f}, p={p_val:.4f} {sig}")

    # One-sample t-tests vs 0.5
    t_w, p_w = stats.ttest_1samp(balacc_with, 0.5)
    t_wo, p_wo = stats.ttest_1samp(balacc_without, 0.5)
    print(f"\n  vs baseline=0.5:")
    print(f"    WITH macro:    t={t_w:.4f}, p={p_w:.4f} {'(significant)' if p_w < 0.05 else '(ns)'}")
    print(f"    WITHOUT macro: t={t_wo:.4f}, p={p_wo:.4f} {'(significant)' if p_wo < 0.05 else '(ns)'}")

    print("\n" + "=" * 70)
    print("KẾT LUẬN")
    print("=" * 70)
    if p_val < 0.05:
        if diff > 0:
            print(f"  → Macro features GIÚP CẢI THIỆN Balanced Accuracy (+{diff:.4f}, p={p_val:.4f})")
        else:
            print(f"  → Macro features LÀM GIẢM Balanced Accuracy ({diff:.4f}, p={p_val:.4f})")
    else:
        print(f"  → Macro features KHÔNG có ảnh hưởng có ý nghĩa thống kê (diff={diff:+.4f}, p={p_val:.4f} ns)")


if __name__ == "__main__":
    main()
