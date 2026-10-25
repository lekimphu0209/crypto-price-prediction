"""
Train a single BiLSTM model WITHOUT regime features (baseline).

Usage:
    python scripts/train_bilstm_baseline.py [--symbol BTCUSDT] [--epochs 20]
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM baseline.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to train on.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--output", default=None, help="Path to save result JSON.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING BiLSTM BASELINE (no regime) | {args.symbol} | epochs={args.epochs}")
    print("=" * 70)

    pipeline = build_training_pipeline()
    forecaster = KerasForecaster("BiLSTM", epochs=args.epochs)

    result = pipeline.train(args.symbol, forecaster, model_name="BiLSTM_Baseline", save_csv=False)

    if result.get("ok"):
        m = result["metrics"]
        print("\n--- RESULTS ---")
        print(
            f"Accuracy          = {m['accuracy']:.4f}\n"
            f"Balanced Accuracy = {m['balanced_accuracy']:.4f}\n"
            f"Precision         = {m['precision']:.4f}\n"
            f"Recall            = {m['recall']:.4f}\n"
            f"F1                = {m['f1']:.4f}\n"
            f"Threshold         = {m['threshold']:.2f}\n"
            f"Confusion Matrix  = TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}"
        )

        out_path = args.output or os.path.join(MODELS_DIR, f"_bilstm_baseline_{args.symbol.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({args.symbol: {"BiLSTM_Baseline": m}}, f, indent=2)
        print(f"\nSaved results to: {out_path}")
    else:
        print(f"Training failed: {result.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
