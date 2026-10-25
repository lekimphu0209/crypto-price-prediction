"""
Walk-forward Validation Script (FIXED - NO DATA LEAKAGE)

Performs rolling window backtesting with proper leakage prevention:
- Split train/test BEFORE scaling
- Fit scaler ONLY on train data
- Transform train/test separately

This simulates real-time trading with proper data handling.

Results are saved to data/{SYMBOL}_walkforward_fixed_{timestamp}.csv
"""
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import ReturnTargetBuilder
from src.training.market_data_loader import MarketDataLoader
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.factory import create_forecaster, SUPPORTED_MODELS

DATA_DIR = "data"


def walk_forward_validation(symbol: str, model_name: str,
                           train_window: int = 300,
                           test_window: int = 50,
                           step_size: int = 50,
                           total_data: int = 1000) -> pd.DataFrame:
    """
    Perform walk-forward validation for a single model - NO DATA LEAKAGE.

    Args:
        symbol: Binance symbol (e.g., BTCUSDT)
        model_name: Model name
        train_window: Number of days to train on each fold
        test_window: Number of days to test on each fold
        step_size: Step size for rolling window
        total_data: Total data points to use

    Returns:
        DataFrame with validation results
    """
    # Build components
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=1000)
    extractor = TechnicalExtractor()
    target_builder = ReturnTargetBuilder()
    window = WindowDataset(seq_len=20)

    # Load all data once
    df = loader.load(symbol)
    if len(df) < total_data:
        total_data = len(df)

    df = df.iloc[-total_data:].reset_index(drop=True)

    results = []
    fold = 0

    # Walk-forward: start from train_window, step forward by step_size
    for test_start in range(train_window, total_data - test_window, step_size):
        test_end = test_start + test_window
        if test_end > total_data:
            break

        # Train data: from (test_start - train_window) to test_start
        train_start = test_start - train_window

        print(f"  Fold {fold + 1}: train [{train_start}:{test_start}], test [{test_start}:{test_end}]")

        # Create temporary data for this fold
        train_df = df.iloc[train_start:test_start].copy()
        test_df = df.iloc[test_start:test_end].copy()

        # Extract features
        feat_train = extractor.extract(train_df)
        feat_test = extractor.extract(test_df)

        feature_names = extractor.feature_names()

        # Scale features - FIT ONLY ON TRAIN (FIX: no leakage)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(feat_train[feature_names].values)

        train_features = scaler.transform(feat_train[feature_names].values)
        test_features = scaler.transform(feat_test[feature_names].values)

        train_close = feat_train["close"].values
        test_close = feat_test["close"].values

        # Build windows
        X_train, base_train, next_train = window.build(train_features, train_close)
        y_train = target_builder.build(base_train, next_train)

        X_test, base_test, next_test = window.build(test_features, test_close)

        if len(X_test) == 0:
            print(f"  Fold {fold + 1}: Skipped (no test windows)")
            fold += 1
            continue

        # Train model
        forecaster = create_forecaster(model_name)
        forecaster.fit(X_train, y_train)

        # Predict (model outputs next-day LOG RETURN)
        ret_pred = forecaster.predict(X_test)
        price_pred = np.array([
            target_builder.reconstruct(b, r) for b, r in zip(base_test, ret_pred)
        ])

        # Actual next-day log return (ground truth)
        ret_actual = np.log(next_test / base_test)

        # --- PRICE-space metrics (inflated by base_close autocorrelation) ---
        mae = float(mean_absolute_error(next_test, price_pred))
        r2 = float(r2_score(next_test, price_pred))

        # --- RETURN-space metrics (honest signal quality) ---
        return_mae = float(mean_absolute_error(ret_actual, ret_pred))
        return_r2 = float(r2_score(ret_actual, ret_pred)) if len(ret_actual) > 1 else float("nan")

        # --- Naive baseline: predict = today's price (return = 0) ---
        # price_pred_naive == base_test, so error is just |next - base|
        naive_mae = float(mean_absolute_error(next_test, base_test))
        # Skill score vs naive (MASE-like): <1 means model beats naive
        mase = float(mae / naive_mae) if naive_mae > 0 else float("nan")
        beats_naive = bool(mae < naive_mae)

        # --- Directional accuracy: predicted return sign vs actual return sign ---
        # (per-sample, not consecutive price diff)
        nonzero = ret_actual != 0
        if nonzero.sum() > 0:
            direction_correct = (np.sign(ret_pred[nonzero]) == np.sign(ret_actual[nonzero])).sum()
            directional_acc = float(direction_correct / len(nonzero))
        else:
            directional_acc = float("nan")

        results.append({
            "fold": fold + 1,
            "mae": mae,
            "r2": r2,
            "return_mae": return_mae,
            "return_r2": return_r2,
            "naive_mae": naive_mae,
            "mase": mase,
            "beats_naive": beats_naive,
            "directional_acc": directional_acc,
        })

        fold += 1

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION (FIXED - NO DATA LEAKAGE)")
    print("Rolling window backtesting with proper data handling")
    print("=" * 70)

    SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    train_window = 300
    test_window = 50
    step_size = 50

    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        all_results[symbol] = {}

        for model_name in SUPPORTED_MODELS:
            print(f"\n{model_name}:")
            df = walk_forward_validation(
                symbol=symbol,
                model_name=model_name,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                total_data=1000,
            )

            if len(df) == 0:
                print("  No valid folds")
                continue

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(DATA_DIR, f"{symbol}_{model_name}_walkforward_fixed_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved to: {csv_path}")

            # Summary
            avg_mae = df["mae"].mean()
            avg_r2 = df["r2"].mean()
            avg_return_mae = df["return_mae"].mean()
            avg_return_r2 = df["return_r2"].mean()
            avg_naive_mae = df["naive_mae"].mean()
            avg_mase = df["mase"].mean()
            beats_naive_pct = df["beats_naive"].mean() * 100
            avg_dir_acc = df["directional_acc"].mean()

            all_results[symbol][model_name] = {
                "avg_mae": avg_mae,
                "avg_r2": avg_r2,
                "avg_return_mae": avg_return_mae,
                "avg_return_r2": avg_return_r2,
                "avg_naive_mae": avg_naive_mae,
                "avg_mase": avg_mase,
                "beats_naive_pct": beats_naive_pct,
                "avg_directional_acc": avg_dir_acc,
                "num_folds": len(df),
            }

            print(f"\n  [PRICE]  Avg MAE: ${avg_mae:,.2f}  |  R²: {avg_r2:.4f}")
            print(f"  [RETURN] Avg MAE: {avg_return_mae:.5f}  |  R²: {avg_return_r2:.4f}")
            print(f"  [NAIVE]  Avg MAE: ${avg_naive_mae:,.2f}  (predict = today's price)")
            print(f"  [SKILL]  MASE: {avg_mase:.3f}  (<1 = beats naive)  |  Beats naive: {beats_naive_pct:.1f}% of folds")
            print(f"  [DIR]    Direction Accuracy: {avg_dir_acc:.2%}  (>50% = useful)")
            print(f"  Folds: {len(df)}")

    # Save overall summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(DATA_DIR, f"walkforward_fixed_summary_{timestamp}.json")
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
