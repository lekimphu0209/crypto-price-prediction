"""
Test 2: Check for look-ahead bias in feature engineering.

Purpose: Verify that features at time t do NOT use information from time > t.
Look-ahead bias is the most dangerous bug in financial projects - it creates
artificially good backtest results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.features.technical_extractor import TechnicalExtractor
from src.features.indicators import moving_average, rsi, macd, bollinger_bands


def test_moving_average_no_lookahead():
    """Test that moving average doesn't use future data."""
    print("=" * 80)
    print("TEST 2: Look-ahead Bias - Moving Average")
    print("=" * 80)
    
    # Create simple price series as pandas Series
    prices = pd.Series([100, 101, 102, 103, 104, 105])
    ma_3 = moving_average(prices, window=3)
    
    # MA at index 2 should use prices[0:3] = [100, 101, 102]
    # NOT prices[2:5] = [102, 103, 104]
    expected_at_2 = (100 + 101 + 102) / 3
    actual_at_2 = ma_3[2]
    
    if abs(actual_at_2 - expected_at_2) < 1e-6:
        print(f"✅ Moving average OK: MA[2]={actual_at_2:.2f} (expected {expected_at_2:.2f})")
        return True
    else:
        print(f"❌ Moving average FAILED: MA[2]={actual_at_2:.2f} (expected {expected_at_2:.2f})")
        return False


def test_rsi_no_lookahead():
    """Test that RSI doesn't use future data."""
    print("\n" + "=" * 80)
    print("TEST 2: Look-ahead Bias - RSI")
    print("=" * 80)
    
    # Create price series as pandas Series
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    rsi_values = rsi(prices, period=14)
    
    # RSI at index 14 should use prices[0:15]
    # Verify that changing future prices doesn't affect past RSI
    prices_copy = prices.copy()
    prices_copy[-1] = 999  # Change last price drastically
    rsi_copy = rsi(prices_copy, period=14)
    
    # RSI at index 13 should be the same (not affected by future change)
    if abs(rsi_values[13] - rsi_copy[13]) < 1e-6:
        print(f"✅ RSI OK: RSI[13] not affected by future price change")
        return True
    else:
        print(f"❌ RSI FAILED: RSI[13] changed when future price changed")
        print(f"   Original: {rsi_values[13]:.4f}, After change: {rsi_copy[13]:.4f}")
        return False


def test_macd_no_lookahead():
    """Test that MACD doesn't use future data."""
    print("\n" + "=" * 80)
    print("TEST 2: Look-ahead Bias - MACD")
    print("=" * 80)
    
    # Create price series as pandas Series
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120])
    macd_line = macd(prices)
    
    # Verify that changing future prices doesn't affect past MACD
    prices_copy = prices.copy()
    prices_copy.iloc[-1] = 999
    macd_copy = macd(prices_copy)
    
    # MACD at index -2 should be the same
    if abs(macd_line.iloc[-2] - macd_copy.iloc[-2]) < 1e-6:
        print(f"✅ MACD OK: MACD[-2] not affected by future price change")
        return True
    else:
        print(f"❌ MACD FAILED: MACD[-2] changed when future price changed")
        print(f"   Original: {macd_line.iloc[-2]:.4f}, After change: {macd_copy.iloc[-2]:.4f}")
        return False


def test_bollinger_bands_no_lookahead():
    """Test that Bollinger Bands don't use future data."""
    print("\n" + "=" * 80)
    print("TEST 2: Look-ahead Bias - Bollinger Bands")
    print("=" * 80)
    
    # Create price series as pandas Series
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120])
    upper, middle, lower = bollinger_bands(prices, window=20, num_std=2)
    
    # Verify that changing future prices doesn't affect past bands
    prices_copy = prices.copy()
    prices_copy.iloc[-1] = 999
    upper_copy, middle_copy, lower_copy = bollinger_bands(prices_copy, window=20, num_std=2)
    
    # Bands at index -2 should be the same
    if abs(upper.iloc[-2] - upper_copy.iloc[-2]) < 1e-6:
        print(f"✅ Bollinger Bands OK: Upper[-2] not affected by future price change")
        return True
    else:
        print(f"❌ Bollinger Bands FAILED: Upper[-2] changed when future price changed")
        print(f"   Original: {upper.iloc[-2]:.4f}, After change: {upper_copy.iloc[-2]:.4f}")
        return False


def test_technical_extractor_no_lookahead():
    """Test that TechnicalExtractor doesn't introduce look-ahead bias."""
    print("\n" + "=" * 80)
    print("TEST 2: Look-ahead Bias - TechnicalExtractor")
    print("=" * 80)
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n_samples = 100
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
    
    # Extract features
    extractor = TechnicalExtractor(feature_set="B")
    features = extractor.extract(df)
    
    # For each feature, check if changing future data affects past values
    feature_names = extractor.feature_names()
    has_lookahead = False
    
    for feature in feature_names:
        if feature == "close":
            continue  # Skip close price itself
        
        # Get feature value at index -3 (to have buffer for rolling windows)
        original_value = features[feature].iloc[-3]
        
        # Change ONLY future data (last 2 rows, well outside any reasonable window)
        df_copy = df.copy()
        df_copy.iloc[-2:, df_copy.columns.get_loc("close")] = 999
        
        # Re-extract features
        features_copy = extractor.extract(df_copy)
        new_value = features_copy[feature].iloc[-3]
        
        # Check if value changed
        if abs(original_value - new_value) > 1e-6:
            print(f"❌ Feature {feature} has look-ahead bias")
            print(f"   Original: {original_value:.4f}, After change: {new_value:.4f}")
            has_lookahead = True
    
    if not has_lookahead:
        print(f"✅ All features OK: No look-ahead bias detected")
        return True
    else:
        print(f"❌ Some features have look-ahead bias")
        return False


def main():
    print("\n" + "=" * 80)
    print("LOOK-AHEAD BIAS CHECK")
    print("=" * 80)
    
    results = []
    results.append(("Moving Average", test_moving_average_no_lookahead()))
    results.append(("RSI", test_rsi_no_lookahead()))
    results.append(("MACD", test_macd_no_lookahead()))
    results.append(("Bollinger Bands", test_bollinger_bands_no_lookahead()))
    results.append(("TechnicalExtractor", test_technical_extractor_no_lookahead()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅ No look-ahead bias detected - features are safe")
    else:
        print("\n❌ Look-ahead bias detected - FIX IMMEDIATELY")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
