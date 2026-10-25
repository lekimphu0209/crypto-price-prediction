"""
Test 5: Majority class baseline for each fold.

Purpose: Establish a realistic baseline - predicting the majority class from train.
This is a more meaningful threshold than 50% random, especially when classes are imbalanced.
If model can't beat this baseline, it's not learning anything useful.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.training.composition import build_training_pipeline
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def create_test_data(n_samples=200, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    close = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + np.random.randn(n_samples) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
        "low": close * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
        "close": close,
        "volume": np.random.randint(1000, 10000, n_samples),
    })
    
    return df


def majority_class_baseline(y_train, y_test):
    """
    Predict majority class from train for all test samples.
    Returns metrics for this baseline.
    """
    # Find majority class in train
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    
    # Predict majority class for all test samples
    y_pred = np.full(len(y_test), majority_class)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
        "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
        "f1": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "majority_class": int(majority_class),
        "train_up_count": int(counts[classes == 1][0]) if 1 in classes else 0,
        "train_down_count": int(counts[classes == 0][0]) if 0 in classes else 0,
    }
    
    return metrics


def test_single_split_baseline():
    """Test majority class baseline on single train/test split."""
    print("=" * 80)
    print("TEST 5: Majority Class Baseline - Single Split")
    print("=" * 80)
    
    # Create data
    df = create_test_data(200)
    
    # Build pipeline
    extractor = TechnicalExtractor(feature_set="B")
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=20, target_horizon=7)
    
    # Extract features
    feat_df = extractor.extract(df)
    feature_names = extractor.feature_names()
    
    # Scale
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:160])
    feature_matrix = scaler.transform(values)
    
    # Build windows
    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)
    
    # Remove NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Calculate baseline
    baseline = majority_class_baseline(y_train, y_test)
    
    print(f"Train: UP={baseline['train_up_count']}, DOWN={baseline['train_down_count']}")
    print(f"Majority class: {baseline['majority_class']} ({'UP' if baseline['majority_class'] == 1 else 'DOWN'})")
    print(f"Test: TN={baseline['tn']}, FP={baseline['fp']}, FN={baseline['fn']}, TP={baseline['tp']}")
    print(f"Baseline Accuracy: {baseline['accuracy']:.4f}")
    print(f"Baseline Balanced Accuracy: {baseline['balanced_accuracy']:.4f}")
    
    return True


def test_walkforward_baseline():
    """Test majority class baseline on walk-forward folds."""
    print("\n" + "=" * 80)
    print("TEST 5: Majority Class Baseline - Walk-Forward (5 Folds)")
    print("=" * 80)
    
    # Create data
    df = create_test_data(300)
    
    # Build pipeline
    extractor = TechnicalExtractor(feature_set="B")
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=20, target_horizon=7)
    
    # Extract features
    feat_df = extractor.extract(df)
    feature_names = extractor.feature_names()
    
    # Scale
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:200])
    feature_matrix = scaler.transform(values)
    
    # Build windows
    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)
    
    # Remove NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    
    # Walk-forward folds
    initial_train_size = 100
    test_size = 30
    n_folds = 5
    
    fold_baselines = []
    
    for fold in range(n_folds):
        train_end = initial_train_size + fold * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end >= len(X):
            print(f"  Fold {fold + 1}: Not enough data, skipping")
            break
        
        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]
        
        baseline = majority_class_baseline(y_train, y_test)
        fold_baselines.append(baseline)
        
        print(f"\nFold {fold + 1}:")
        print(f"  Train: UP={baseline['train_up_count']}, DOWN={baseline['train_down_count']}")
        print(f"  Majority: {baseline['majority_class']} ({'UP' if baseline['majority_class'] == 1 else 'DOWN'})")
        print(f"  Test: Acc={baseline['accuracy']:.4f}, BalAcc={baseline['balanced_accuracy']:.4f}")
    
    # Aggregate
    if fold_baselines:
        mean_acc = np.mean([b['accuracy'] for b in fold_baselines])
        std_acc = np.std([b['accuracy'] for b in fold_baselines])
        mean_bal_acc = np.mean([b['balanced_accuracy'] for b in fold_baselines])
        std_bal_acc = np.std([b['balanced_accuracy'] for b in fold_baselines])
        
        print(f"\nAggregated Baseline:")
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
        
        return True
    else:
        print("❌ No folds completed")
        return False


def test_real_data_baseline():
    """Test majority class baseline on real BTCUSDT data."""
    print("\n" + "=" * 80)
    print("TEST 5: Majority Class Baseline - Real BTCUSDT Data")
    print("=" * 80)
    
    # Build pipeline with real data
    extractor = TechnicalExtractor(feature_set="B")
    pipeline = build_training_pipeline(extractor=extractor, seq_len=30, horizon=7)
    
    # Load real data
    df = pipeline.loader.load("BTCUSDT")
    print(f"Real data shape: {df.shape}")
    
    # Extract features
    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()
    
    # Scale
    from sklearn.preprocessing import RobustScaler
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    scaler = RobustScaler()
    scaler.fit(values[:800])
    feature_matrix = scaler.transform(values)
    
    # Build windows
    from src.training.window_dataset import WindowDataset
    window = WindowDataset(seq_len=30, target_horizon=7)
    X, base_close, future_close = window.build(feature_matrix, close)
    y = pipeline.target_builder.build(base_close, future_close)
    
    # Remove NaN
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Calculate baseline
    baseline = majority_class_baseline(y_train, y_test)
    
    print(f"Train: UP={baseline['train_up_count']}, DOWN={baseline['train_down_count']}")
    print(f"Majority class: {baseline['majority_class']} ({'UP' if baseline['majority_class'] == 1 else 'DOWN'})")
    print(f"Test: TN={baseline['tn']}, FP={baseline['fp']}, FN={baseline['fn']}, TP={baseline['tp']}")
    print(f"Baseline Accuracy: {baseline['accuracy']:.4f}")
    print(f"Baseline Balanced Accuracy: {baseline['balanced_accuracy']:.4f}")
    
    # Compare with actual model results
    print(f"\nComparison with previous model results:")
    print(f"  LSTM seed0 walk-forward: BalAcc ~0.56")
    print(f"  Majority baseline: BalAcc {baseline['balanced_accuracy']:.4f}")
    
    if baseline['balanced_accuracy'] > 0.56:
        print(f"⚠️  Majority baseline BEATS LSTM model!")
        print(f"   Model is not learning useful patterns")
    else:
        print(f"✅ LSTM model beats majority baseline")
    
    return True


def main():
    print("\n" + "=" * 80)
    print("MAJORITY CLASS BASELINE CHECK")
    print("=" * 80)
    
    results = []
    results.append(("Single Split Baseline", test_single_split_baseline()))
    results.append(("Walk-Forward Baseline", test_walkforward_baseline()))
    results.append(("Real Data Baseline", test_real_data_baseline()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅ All baseline checks completed")
        print("   Compare model performance against these baselines")
    else:
        print("\n❌ Some baseline checks failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
