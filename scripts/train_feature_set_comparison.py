"""
Train models across 3 feature sets for comparison.

Trains Logistic Regression, RNN, and LSTM on:
    Feature Set A: original features
    Feature Set B: original + stationary features
    Feature Set C: only stationary features

Results are saved to models/_feature_set_comparison.json.
Models are overwritten per feature set because the goal is to compare metrics,
not to keep every trained model.

Usage:
    python scripts/train_feature_set_comparison.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.features.technical_extractor import TechnicalExtractor
from src.infrastructure.forecasters.factory import create_forecaster

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FEATURE_SETS = ["A", "B", "C"]
MODELS = ["Logistic Regression", "RNN", "LSTM"]


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f}  "
        f"BalAcc={m['balanced_accuracy']:.4f}  "
        f"Thr={m['threshold']:.2f}  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
        f"[TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
    )


def main():
    print("=" * 90)
    print("FEATURE SET COMPARISON")
    print("A: original  |  B: original + stationary  |  C: only stationary")
    print("Models: Logistic Regression, RNN, LSTM")
    print("=" * 90)

    summary = {}

    for symbol in SYMBOLS:
        print(f"\n### {symbol} ###")
        summary[symbol] = {}

        for fs_idx, feature_set in enumerate(FEATURE_SETS):
            print(f"\n--- Feature Set {feature_set} ---")
            extractor = TechnicalExtractor(feature_set=feature_set)
            pipeline = build_training_pipeline(extractor=extractor)
            summary[symbol][feature_set] = {}

            # Only save raw/features CSV once per symbol (first feature set)
            save_csv = (fs_idx == 0)

            for model_name in MODELS:
                forecaster = create_forecaster(model_name)
                result = pipeline.train(
                    symbol, forecaster, model_name=model_name, save_csv=save_csv
                )
                save_csv = False  # only first model of first feature set

                if result.get("ok"):
                    metrics = result["metrics"]
                    summary[symbol][feature_set][model_name] = metrics
                    print(f"  {model_name:<18} | Set {feature_set} -> {format_metrics(metrics)}")
                else:
                    err = result.get("error", "unknown")
                    summary[symbol][feature_set][model_name] = {"error": err}
                    print(f"  {model_name:<18} | Set {feature_set} -> FAILED: {err}")

    out_path = os.path.join(MODELS_DIR, "_feature_set_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison summary saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
