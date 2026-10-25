"""
Walk-forward Validation Script

Performs rolling window backtesting:
- Train on t1-t300, test t301-t320
- Train on t21-t320, test t321-t340
- And so on...

This simulates real-time trading better than simple train/test split.

Results are saved to data/{SYMBOL}_walkforward_{timestamp}.csv
"""
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR, DATA_DIR
from src.infrastructure.forecasters.factory import create_forecaster, SUPPORTED_MODELS


def walk_forward_validation(symbol: str, model_name: str,
                           train_window: int = 300,
                           test_window: int = 20,
                           step_size: int = 20,
                           total_data: int = 1000) -> pd.DataFrame:
    """
    Perform walk-forward validation for a single model.
    
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
    pipeline = build_training_pipeline()
    
    # Load all data once
    df = pipeline.loader.load(symbol)
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
        
        # We need to mock the loader to return only this fold's data
        # For simplicity, we'll use the pipeline's internal methods
        
        # Extract features
        feat_train = pipeline.extractor.extract(train_df)
        feat_test = pipeline.extractor.extract(test_df)
        
        feature_names = pipeline.extractor.feature_names()
        
        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(feat_train[feature_names].values)
        
        train_features = scaler.transform(feat_train[feature_names].values)
        test_features = scaler.transform(feat_test[feature_names].values)
        
        train_close = feat_train["close"].values
        test_close = feat_test["close"].values
        
        # Build windows
        X_train, base_train, next_train = pipeline.window.build(train_features, train_close)
        y_train = pipeline.target_builder.build(base_train, next_train)
        
        X_test, base_test, next_test = pipeline.window.build(test_features, test_close)
        
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
            pipeline.target_builder.reconstruct(b, r) for b, r in zip(base_test, ret_pred)
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
            direction_accuracy = float(
                np.mean(np.sign(ret_pred[nonzero]) == np.sign(ret_actual[nonzero]))
            )
        else:
            direction_accuracy = float("nan")

        results.append({
            "fold": fold + 1,
            "train_start": train_start,
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
            "mae": mae,
            "r2": r2,
            "return_mae": return_mae,
            "return_r2": return_r2,
            "naive_mae": naive_mae,
            "mase": mase,
            "beats_naive": beats_naive,
            "direction_accuracy": direction_accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        fold += 1
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("Rolling window backtesting to simulate real-time trading")
    print("=" * 70)
    
    SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    TRAIN_WINDOW = 250  # Reduced to allow more folds
    TEST_WINDOW = 50    # Increased for more test samples
    STEP_SIZE = 50      # Step size for rolling window
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {}
    
    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        all_results[symbol] = {}
        
        for model_name in SUPPORTED_MODELS:
            print(f"\n{model_name}:")
            try:
                df = walk_forward_validation(
                    symbol, 
                    model_name, 
                    train_window=TRAIN_WINDOW,
                    test_window=TEST_WINDOW,
                    step_size=STEP_SIZE,
                )
                
                if len(df) > 0:
                    # Save individual model results
                    csv_path = os.path.join(DATA_DIR, f"{symbol}_{model_name}_walkforward_{timestamp}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"  Saved to: {csv_path}")
                    
                    # Summary stats
                    avg_mae = df['mae'].mean()
                    avg_naive_mae = df['naive_mae'].mean()
                    avg_mase = df['mase'].mean()
                    beats_naive_pct = df['beats_naive'].mean() * 100
                    print(f"  [PRICE]  Avg MAE: ${avg_mae:,.2f}  |  R²: {df['r2'].mean():.4f}")
                    print(f"  [RETURN] Avg MAE: {df['return_mae'].mean():.5f}  |  R²: {df['return_r2'].mean():.4f}")
                    print(f"  [NAIVE]  Avg MAE: ${avg_naive_mae:,.2f}  (predict = today's price)")
                    print(f"  [SKILL]  MASE: {avg_mase:.3f}  (<1 = beats naive)  |  Beats naive: {beats_naive_pct:.0f}% of folds")
                    print(f"  [DIR]    Direction Accuracy: {df['direction_accuracy'].mean()*100:.2f}%  (>50% = useful)")
                    print(f"  Folds: {len(df)}")

                    all_results[symbol][model_name] = {
                        "avg_mae": float(avg_mae),
                        "avg_r2": float(df['r2'].mean()),
                        "avg_return_mae": float(df['return_mae'].mean()),
                        "avg_return_r2": float(df['return_r2'].mean()),
                        "avg_naive_mae": float(avg_naive_mae),
                        "avg_mase": float(avg_mase),
                        "beats_naive_pct": float(beats_naive_pct),
                        "avg_direction_accuracy": float(df['direction_accuracy'].mean()),
                        "folds": len(df),
                    }
                else:
                    print(f"  No folds generated")
                    all_results[symbol][model_name] = {"error": "No folds generated"}
                    
            except Exception as e:
                print(f"  Error: {e}")
                all_results[symbol][model_name] = {"error": str(e)}
    
    # Save summary
    summary_path = os.path.join(DATA_DIR, f"walkforward_summary_{timestamp}.json")
    import json
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
