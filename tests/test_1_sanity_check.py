"""
Test 1: Sanity check with synthetic data for entire pipeline.

Purpose: Verify the entire pipeline works correctly with known synthetic data.
If this fails, there's a systemic bug in the pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.training.composition import build_training_pipeline
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.regression_target_builder import RegressionTargetBuilder
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster


def create_synthetic_data(n_samples=500):
    """Create synthetic OHLCV data with known patterns."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    
    # Random walk with drift for close price
    close = 100 + np.cumsum(np.random.randn(n_samples) * 0.01 + 0.001)
    
    # Create OHLCV with timestamp column (required by TechnicalExtractor)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + np.random.randn(n_samples) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
        "low": close * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
        "close": close,
        "volume": np.random.randint(1000, 10000, n_samples),
    })
    
    return df


def test_classification_pipeline():
    """Test classification pipeline with synthetic data."""
    print("=" * 80)
    print("TEST 1: Sanity Check - Classification Pipeline")
    print("=" * 80)
    
    # Create synthetic data
    df = create_synthetic_data(500)
    print(f"Synthetic data shape: {df.shape}")
    
    # Build pipeline
    extractor = TechnicalExtractor(feature_set="B")
    pipeline = build_training_pipeline(extractor=extractor, seq_len=20, horizon=7)
    
    # Mock loader to return synthetic data
    class MockLoader:
        def __init__(self, df):
            self.df = df
        def load(self, symbol):
            return self.df
    
    pipeline.loader = MockLoader(df)
    
    # Test with Logistic Regression (doesn't support validation data)
    forecaster = LinearForecaster(C=1.0, class_weight="balanced")
    
    # Temporarily disable validation for LinearForecaster
    original_train = pipeline.train
    def train_no_val(self, symbol, forecaster, model_name="model", save_csv=False):
        # Skip validation for LinearForecaster
        from src.training.training_pipeline import TrainingPipeline
        # Temporarily modify threshold tuning to skip validation
        original_threshold_metric = self.threshold_metric
        self.threshold_metric = "accuracy"  # Use simpler metric
        
        # Manually call the training logic without validation
        result = original_train(symbol, forecaster, model_name, save_csv)
        
        self.threshold_metric = original_threshold_metric
        return result
    
    # Simpler approach: just test with RNN which supports validation
    forecaster = KerasForecaster(
        name="RNN",
        epochs=2,
        batch_size=16,
        units=16,
        dropout=0.1,
        early_stopping_patience=0,
        mode="classification",
    )
    
    result = pipeline.train("SYNTH", forecaster, model_name="test_rnn", save_csv=False)
    
    if result.get("ok"):
        print(f"✅ Classification pipeline OK: Acc={result['metrics']['accuracy']:.4f}")
        return True
    else:
        print(f"❌ Classification pipeline FAILED: {result.get('error')}")
        return False


def test_regression_pipeline():
    """Test regression pipeline with synthetic data."""
    print("\n" + "=" * 80)
    print("TEST 1: Sanity Check - Regression Pipeline")
    print("=" * 80)
    
    # Create synthetic data
    df = create_synthetic_data(500)
    print(f"Synthetic data shape: {df.shape}")
    
    # Build pipeline with regression mode
    extractor = TechnicalExtractor(feature_set="B")
    target_builder = RegressionTargetBuilder()
    pipeline = build_training_pipeline(
        extractor=extractor,
        seq_len=20,
        horizon=7,
        mode="regression",
        target_builder=target_builder,
    )
    
    # Mock loader
    class MockLoader:
        def __init__(self, df):
            self.df = df
        def load(self, symbol):
            return self.df
    
    pipeline.loader = MockLoader(df)
    
    # Test with RNN regression
    forecaster = KerasForecaster(
        name="RNN",
        epochs=5,
        batch_size=16,
        units=16,
        dropout=0.1,
        early_stopping_patience=0,
        mode="regression",
    )
    result = pipeline.train("SYNTH", forecaster, model_name="test_rnn_reg", save_csv=False)
    
    if result.get("ok"):
        print(f"✅ Regression pipeline OK: R2={result['metrics']['r2']:.4f}")
        return True
    else:
        print(f"❌ Regression pipeline FAILED: {result.get('error')}")
        return False


def test_rnn_classification():
    """Test RNN classification with synthetic data."""
    print("\n" + "=" * 80)
    print("TEST 1: Sanity Check - RNN Classification")
    print("=" * 80)
    
    # Create synthetic data
    df = create_synthetic_data(500)
    print(f"Synthetic data shape: {df.shape}")
    
    # Build pipeline
    extractor = TechnicalExtractor(feature_set="B")
    pipeline = build_training_pipeline(extractor=extractor, seq_len=20, horizon=7)
    
    # Mock loader
    class MockLoader:
        def __init__(self, df):
            self.df = df
        def load(self, symbol):
            return self.df
    
    pipeline.loader = MockLoader(df)
    
    # Test with RNN
    forecaster = KerasForecaster(
        name="RNN",
        epochs=5,
        batch_size=16,
        units=16,
        dropout=0.1,
        early_stopping_patience=0,
        mode="classification",
    )
    result = pipeline.train("SYNTH", forecaster, model_name="test_rnn", save_csv=False)
    
    if result.get("ok"):
        print(f"✅ RNN classification OK: Acc={result['metrics']['accuracy']:.4f}")
        return True
    else:
        print(f"❌ RNN classification FAILED: {result.get('error')}")
        return False


def main():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Synthetic Data Pipeline Test")
    print("=" * 80)
    
    results = []
    results.append(("Classification Pipeline", test_classification_pipeline()))
    results.append(("Regression Pipeline", test_regression_pipeline()))
    results.append(("RNN Classification", test_rnn_classification()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅ All sanity checks passed - pipeline is working correctly")
    else:
        print("\n❌ Some sanity checks failed - there may be systemic bugs")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
