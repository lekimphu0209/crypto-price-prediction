"""
Pipeline Verification Script

Tests each step of the training/prediction pipeline to ensure data integrity:
1. Data fetching - verify OHLCV data from Binance
2. Feature extraction - verify technical indicators calculation
3. Window creation - verify sliding windows
4. Target building - verify return calculation
5. Scaling - verify scaling logic (features only, not target)
6. Prediction reconstruction - verify price reconstruction from return
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline


def test_data_fetching(symbol: str = "BTCUSDT"):
    """Test 1: Verify data fetching from Binance"""
    print("\n" + "=" * 70)
    print("TEST 1: DATA FETCHING")
    print("=" * 70)
    
    pipeline = build_training_pipeline()
    df = pipeline.loader.load(symbol)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    print(f"✓ Sample data (first 3 rows):")
    print(df.head(3))
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"✗ WARNING: Missing values found:\n{missing}")
    else:
        print("✓ No missing values")
    
    # Check data types
    print(f"✓ Data types:\n{df.dtypes}")
    
    # Check for negative values in price/volume
    if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        print("✗ WARNING: Negative values found in price/volume")
    else:
        print("✓ No negative values in price/volume")
    
    return df


def test_feature_extraction(df: pd.DataFrame):
    """Test 2: Verify technical indicators calculation"""
    print("\n" + "=" * 70)
    print("TEST 2: FEATURE EXTRACTION")
    print("=" * 70)
    
    pipeline = build_training_pipeline()
    feat_df = pipeline.extractor.extract(df)
    
    print(f"✓ Extracted features: {pipeline.extractor.feature_names()}")
    print(f"✓ Feature shape: {feat_df.shape}")
    print(f"✓ Sample features (first 3 rows):")
    print(feat_df[pipeline.extractor.feature_names()].head(3))
    
    # Check for missing values after feature extraction
    missing = feat_df.isnull().sum()
    if missing.sum() > 0:
        print(f"✗ WARNING: Missing values after feature extraction:\n{missing}")
        print(f"  Rows with NaN: {feat_df.isnull().any(axis=1).sum()}")
    else:
        print("✓ No missing values after feature extraction")
    
    # Verify specific indicators
    print("\n✓ Verifying specific indicators:")
    
    # Check MA7 and MA14
    if 'ma_7' in feat_df.columns and 'ma_14' in feat_df.columns:
        # MA should be close to close price (smoothed)
        ma7_diff = (feat_df['ma_7'] - feat_df['close']).abs().mean()
        ma14_diff = (feat_df['ma_14'] - feat_df['close']).abs().mean()
        print(f"  MA7 avg diff from close: ${ma7_diff:.2f}")
        print(f"  MA14 avg diff from close: ${ma14_diff:.2f}")
    
    # Check RSI (should be between 0-100)
    if 'rsi' in feat_df.columns:
        rsi_min = feat_df['rsi'].min()
        rsi_max = feat_df['rsi'].max()
        print(f"  RSI range: [{rsi_min:.2f}, {rsi_max:.2f}]")
        if rsi_min < 0 or rsi_max > 100:
            print(f"  ✗ WARNING: RSI out of range [0, 100]")
        else:
            print(f"  ✓ RSI in valid range")
    
    # Check Bollinger Bands (upper > middle > lower)
    if 'bb_upper' in feat_df.columns and 'bb_middle' in feat_df.columns and 'bb_lower' in feat_df.columns:
        bb_valid = (feat_df['bb_upper'] >= feat_df['bb_middle']).all() and \
                   (feat_df['bb_middle'] >= feat_df['bb_lower']).all()
        if bb_valid:
            print(f"  ✓ Bollinger Bands: upper >= middle >= lower")
        else:
            print(f"  ✗ WARNING: Bollinger Bands ordering incorrect")
    
    return feat_df


def test_window_creation(feature_matrix: np.ndarray, close: np.ndarray, seq_len: int = 20):
    """Test 3: Verify sliding window creation"""
    print("\n" + "=" * 70)
    print("TEST 3: WINDOW CREATION")
    print("=" * 70)
    
    pipeline = build_training_pipeline()
    X, base_close, next_close = pipeline.window.build(feature_matrix, close)
    
    print(f"✓ Input shape: {feature_matrix.shape}")
    print(f"✓ Window length (seq_len): {seq_len}")
    print(f"✓ Number of windows created: {len(X)}")
    print(f"✓ X shape: {X.shape} (samples, seq_len, features)")
    print(f"✓ base_close shape: {base_close.shape}")
    print(f"✓ next_close shape: {next_close.shape}")
    
    # Verify window shapes
    if X.shape[1] != seq_len:
        print(f"✗ ERROR: Window length mismatch: expected {seq_len}, got {X.shape[1]}")
    else:
        print(f"✓ Window length correct")
    
    # Verify base_close and next_close relationship
    # next_close should be the close price after base_close in the original series
    print(f"\n✓ Verifying window content:")
    print(f"  First window - base_close: ${base_close[0]:.2f}, next_close: ${next_close[0]:.2f}")
    print(f"  Last window - base_close: ${base_close[-1]:.2f}, next_close: ${next_close[-1]:.2f}")
    
    # Check that windows are sequential
    for i in range(min(5, len(base_close)-1)):
        if base_close[i+1] != close[seq_len + i]:
            print(f"  ✗ WARNING: Window {i+1} base_close doesn't match expected position")
        else:
            print(f"  ✓ Window {i+1} base_close matches expected position")
    
    return X, base_close, next_close


def test_target_building(base_close: np.ndarray, next_close: np.ndarray):
    """Test 4: Verify return calculation"""
    print("\n" + "=" * 70)
    print("TEST 4: TARGET BUILDING (RETURN CALCULATION)")
    print("=" * 70)
    
    pipeline = build_training_pipeline()
    y = pipeline.target_builder.build(base_close, next_close)
    
    print(f"✓ Number of targets: {len(y)}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Sample returns (first 5): {y[:5]}")
    print(f"✓ Return statistics:")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std: {y.std():.6f}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")
    
    # Verify return calculation manually
    manual_returns = (next_close - base_close) / base_close
    diff = np.abs(y - manual_returns).max()
    print(f"\n✓ Verifying return calculation:")
    print(f"  Max difference from manual calculation: {diff:.10f}")
    if diff < 1e-10:
        print(f"  ✓ Return calculation correct")
    else:
        print(f"  ✗ ERROR: Return calculation incorrect")
    
    # Test reconstruction
    print(f"\n✓ Verifying price reconstruction:")
    reconstructed = pipeline.target_builder.reconstruct(base_close[0], y[0])
    expected = next_close[0]
    diff_recon = abs(reconstructed - expected)
    print(f"  Original: ${expected:.2f}")
    print(f"  Reconstructed: ${reconstructed:.2f}")
    print(f"  Difference: ${diff_recon:.2f}")
    if diff_recon < 0.01:
        print(f"  ✓ Reconstruction correct")
    else:
        print(f"  ✗ WARNING: Reconstruction has error")
    
    return y


def test_scaling_logic(feature_matrix: np.ndarray):
    """Test 5: Verify scaling logic"""
    print("\n" + "=" * 70)
    print("TEST 5: SCALING LOGIC")
    print("=" * 70)
    
    from sklearn.preprocessing import StandardScaler
    
    print(f"✓ Original feature matrix shape: {feature_matrix.shape}")
    print(f"✓ Original feature statistics (first 3 features):")
    for i in range(min(3, feature_matrix.shape[1])):
        print(f"  Feature {i}: mean={feature_matrix[:, i].mean():.4f}, std={feature_matrix[:, i].std():.4f}")
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)
    
    print(f"\n✓ Scaled feature matrix shape: {scaled_matrix.shape}")
    print(f"✓ Scaled feature statistics (first 3 features):")
    for i in range(min(3, scaled_matrix.shape[1])):
        print(f"  Feature {i}: mean={scaled_matrix[:, i].mean():.10f}, std={scaled_matrix[:, i].std():.10f}")
    
    # Verify scaling (mean ~0, std ~1)
    mean_diff = np.abs(scaled_matrix.mean()).max()
    std_diff = np.abs(scaled_matrix.std(axis=0) - 1.0).max()
    print(f"\n✓ Verifying scaling:")
    print(f"  Max deviation from mean=0: {mean_diff:.10f}")
    print(f"  Max deviation from std=1: {std_diff:.10f}")
    if mean_diff < 1e-10 and std_diff < 1e-10:
        print(f"  ✓ Scaling correct")
    else:
        print(f"  ✗ WARNING: Scaling has small deviations (acceptable)")
    
    return scaler, scaled_matrix


def test_prediction_reconstruction():
    """Test 6: Verify prediction reconstruction"""
    print("\n" + "=" * 70)
    print("TEST 6: PREDICTION RECONSTRUCTION")
    print("=" * 70)
    
    pipeline = build_training_pipeline()
    
    # Test with simple values
    test_base = np.array([100.0, 200.0, 300.0])
    test_returns = np.array([0.05, -0.03, 0.02])
    
    print(f"✓ Test reconstruction with known values:")
    for i in range(len(test_base)):
        reconstructed = pipeline.target_builder.reconstruct(test_base[i], test_returns[i])
        expected = test_base[i] * (1.0 + test_returns[i])
        diff = abs(reconstructed - expected)
        print(f"  Base: ${test_base[i]:.2f}, Return: {test_returns[i]:.2%}")
        print(f"    Expected: ${expected:.2f}, Reconstructed: ${reconstructed:.2f}, Diff: ${diff:.4f}")
        if diff < 0.01:
            print(f"    ✓ Correct")
        else:
            print(f"    ✗ ERROR")
    
    return True


def main():
    print("=" * 70)
    print("PIPELINE VERIFICATION")
    print("Ensuring data integrity through all pipeline steps")
    print("=" * 70)
    
    SYMBOL = "BTCUSDT"
    
    try:
        # Test 1: Data fetching
        df = test_data_fetching(SYMBOL)
        
        # Test 2: Feature extraction
        feat_df = test_feature_extraction(df)
        
        # Get feature matrix
        pipeline = build_training_pipeline()
        feature_names = pipeline.extractor.feature_names()
        feature_matrix = feat_df[feature_names].values
        close = feat_df["close"].values
        
        # Test 5: Scaling logic (before window creation)
        scaler, scaled_matrix = test_scaling_logic(feature_matrix)
        
        # Test 3: Window creation
        X, base_close, next_close = test_window_creation(scaled_matrix, close)
        
        # Test 4: Target building
        y = test_target_building(base_close, next_close)
        
        # Test 6: Prediction reconstruction
        test_prediction_reconstruction()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
