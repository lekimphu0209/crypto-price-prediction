"""
Train a single model for a single symbol (fast iteration / demos).

Unlike train_models.py (which loops every registered model), this trains
exactly one {symbol, model} pair through the same Clean Architecture
pipeline - handy when validating a newly registered model.

Usage (from repo root or backend/):
    python backend/scripts/train_one.py --symbol BTCUSDT --model LightGBM
    python backend/scripts/train_one.py -s ETHUSDT -m "CNN-LSTM" --horizon 14
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.training.composition import build_training_pipeline
from src.models.registry import create_forecaster, registered_models


def main():
    cfg = load_config("default")

    parser = argparse.ArgumentParser(description="Train one model for one symbol")
    parser.add_argument("--symbol", "-s", default="BTCUSDT")
    parser.add_argument("--model", "-m", required=True,
                        help=f"Registered model name: {registered_models()}")
    parser.add_argument("--horizon", type=int, default=int(cfg.get("horizon", 7)))
    parser.add_argument("--seq-len", type=int, default=int(cfg.get("seq_len", 20)))
    parser.add_argument("--mode", default="classification",
                        choices=["classification", "regression"])
    args = parser.parse_args()

    print(f"Training {args.model} on {args.symbol} "
          f"(mode={args.mode}, horizon={args.horizon}, seq_len={args.seq_len})")

    pipeline = build_training_pipeline(
        seq_len=args.seq_len, horizon=args.horizon, mode=args.mode,
    )
    forecaster = create_forecaster(args.model, mode=args.mode)
    result = pipeline.train(args.symbol, forecaster, model_name=args.model, save_csv=True)

    if not result.get("ok"):
        print(f"FAILED: {result.get('error', 'unknown')}")
        sys.exit(1)

    m = result["metrics"]
    print(f"\n{args.model} on {args.symbol}:")
    for key in ("accuracy", "balanced_accuracy", "precision", "recall",
                "f1", "threshold", "mse", "mae", "r2"):
        if key in m:
            print(f"  {key:<18} {m[key]:.4f}" if isinstance(m[key], float) else f"  {key:<18} {m[key]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
