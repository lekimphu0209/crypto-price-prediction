"""
Train models with MA-based regime features.

This is a drop-in variant of `scripts/train_models.py` that injects one-hot
regime features (`regime_bull`, `regime_bear`, `regime_sideways`) derived from
short/long moving averages.

Usage:
    python scripts/train_models_with_regime.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import (
    build_training_pipeline_with_regime,
    MODELS_DIR,
)
from src.models.registry import create_forecaster, SUPPORTED_MODELS

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def main():
    print("=" * 70)
    print("TRAINING WITH MA-BASED REGIME FEATURES")
    print("Regimes: bull / sideways / bear from short/long MA")
    print("=" * 70)

    pipeline = build_training_pipeline_with_regime(base_feature_set="E")
    summary = {}

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        summary[symbol] = {}
        for i, model_name in enumerate(SUPPORTED_MODELS):
            forecaster = create_forecaster(model_name)
            save_csv = (i == 0)
            result = pipeline.train(symbol, forecaster, model_name=model_name, save_csv=save_csv)
            if result.get("ok"):
                m = result["metrics"]
                summary[symbol][model_name] = m
                print(
                    f"  {model_name:<18} -> Acc={m['accuracy']:.4f}  "
                    f"BalAcc={m['balanced_accuracy']:.4f}  Thr={m['threshold']:.2f}  "
                    f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                    f"[TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
                )
            else:
                err = result.get("error", "unknown")
                summary[symbol][model_name] = {"error": err}
                print(f"  {model_name:<18} -> FAILED: {err}")

    out_path = os.path.join(MODELS_DIR, "_summary_regime.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
