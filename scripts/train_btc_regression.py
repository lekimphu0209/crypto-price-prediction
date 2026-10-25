"""
Train RNN/LSTM regression models for BTCUSDT 7-day log return.

Predicts the continuous log return over 7 days, then converts the predicted
sign to an Up/Down direction. Evaluates both regression metrics (MSE, MAE, R2)
and direction accuracy.

Usage:
    python scripts/train_btc_regression.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.features.technical_extractor import TechnicalExtractor
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
SEQ_LEN = 30
HORIZON = 7
SEEDS = [0, 1, 2]
MODELS = ["RNN", "LSTM"]


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"R2={m['r2']:.4f} MAE={m['mae']:.4f} MSE={m['mse']:.4f} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    print("=" * 80)
    print("BTCUSDT 7-DAY REGRESSION (RNN + LSTM)")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    pipeline = build_training_pipeline(
        extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON, mode="regression"
    )

    results = []
    total = len(MODELS) * len(SEEDS)
    run_idx = 0

    for model_name in MODELS:
        for seed in SEEDS:
            run_idx += 1
            label = f"Reg_{model_name}_seed{seed}"
            print(f"\n[{run_idx}/{total}] {label}")
            set_seed(seed)

            forecaster = KerasForecaster(
                name=model_name,
                epochs=40,
                batch_size=16,
                units=32,
                dropout=0.1,
                learning_rate=0.001,
                early_stopping_patience=10,
                validation_split=0.1,
                mode="regression",
            )

            start = time.time()
            try:
                result = pipeline.train(
                    SYMBOL,
                    forecaster,
                    model_name=label,
                    save_csv=(run_idx == 1),
                )
                elapsed = time.time() - start
                if result.get("ok"):
                    m = result["metrics"]
                    row = {
                        "model": model_name,
                        "seed": seed,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        **m,
                    }
                    results.append(row)
                    print(f"  -> {format_metrics(m)} ({elapsed:.1f}s)")
                else:
                    err = result.get("error", "unknown")
                    results.append({
                        "model": model_name,
                        "seed": seed,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        "error": err,
                    })
                    print(f"  -> FAILED: {err}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "label": label,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_regression_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    successful = [r for r in results if "error" not in r]
    if successful:
        print("\nTop 3 by Direction Accuracy:")
        top = sorted(successful, key=lambda x: x["accuracy"], reverse=True)
        for r in top[:3]:
            print(f"  {r['label']:<22} -> Acc={r['accuracy']:.4f} BalAcc={r['balanced_accuracy']:.4f}")

        print("\nTop 3 by R2:")
        top_r2 = sorted(successful, key=lambda x: x["r2"], reverse=True)
        for r in top_r2[:3]:
            print(f"  {r['label']:<22} -> R2={r['r2']:.4f} Acc={r['accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
