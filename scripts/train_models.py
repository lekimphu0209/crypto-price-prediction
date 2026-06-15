"""
Train Models (Clean Architecture pipeline)

Trains all models for BTC and ETH using the decoupled training pipeline:
    data -> features (close, volume, MA7, MA14, RSI, MACD, Bollinger)
         -> window -> next-day return -> forecaster -> registry

Models are saved under models/{SYMBOL}/{model_slug}/.

Usage:
    python scripts/train_models.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.infrastructure.forecasters.factory import create_forecaster, SUPPORTED_MODELS

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def main():
    print("=" * 70)
    print("TRAINING MODELS (Clean Architecture)")
    print("Features: close, volume, MA7, MA14, RSI, MACD, Bollinger Bands")
    print("=" * 70)

    pipeline = build_training_pipeline()
    summary = {}

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        summary[symbol] = {}
        for i, model_name in enumerate(SUPPORTED_MODELS):
            forecaster = create_forecaster(model_name)
            # Only save CSV for first model of each symbol
            save_csv = (i == 0)
            result = pipeline.train(symbol, forecaster, model_name=model_name, save_csv=save_csv)
            if result.get("ok"):
                m = result["metrics"]
                summary[symbol][model_name] = m
                print(f"  {model_name:<18} -> R2={m['r2']:.4f}  MAE=${m['mae']:,.2f}")
            else:
                err = result.get("error", "unknown")
                summary[symbol][model_name] = {"error": err}
                print(f"  {model_name:<18} -> FAILED: {err}")

    out_path = os.path.join(MODELS_DIR, "_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
