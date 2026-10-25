"""
Hyperparameter tuning for BTCUSDT only.

Tests Feature Set B (baseline) and Feature Set E (enhanced) with different
sequence lengths and model hyperparameters. Results are saved to
models/btc_tuning_results.json.

Usage:
    python scripts/train_btc_tuning.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.features.technical_extractor import TechnicalExtractor
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster

SYMBOL = "BTCUSDT"
FEATURE_SETS = ["B", "E"]
SEQ_LENS = [20, 30]

HYPERPARAMS = {
    "Logistic Regression": [
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 0.1, "class_weight": "balanced"},
    ],
    "RNN": [
        {"units": 32, "dropout": 0.1, "epochs": 80, "batch_size": 16},
        {"units": 64, "dropout": 0.2, "epochs": 80, "batch_size": 16},
    ],
    "LSTM": [
        {"units": 32, "dropout": 0.1, "epochs": 80, "batch_size": 16},
        {"units": 64, "dropout": 0.2, "epochs": 80, "batch_size": 16},
    ],
}


def create_forecaster(model_name, hp):
    if model_name == "Logistic Regression":
        return LinearForecaster(
            C=hp.get("C", 1.0),
            class_weight=hp.get("class_weight", "balanced"),
            max_iter=5000,
        )
    return KerasForecaster(
        name=model_name,
        epochs=hp.get("epochs", 40),
        batch_size=hp.get("batch_size", 16),
        units=hp.get("units", 32),
        dropout=hp.get("dropout", 0.1),
        learning_rate=hp.get("learning_rate", 0.001),
        early_stopping_patience=hp.get("early_stopping_patience", 10),
        validation_split=hp.get("validation_split", 0.1),
    )


def make_label(model_name, feature_set, seq_len, hp):
    parts = [model_name.replace(" ", "_"), f"fs{feature_set}", f"sl{seq_len}"]
    if model_name == "Logistic Regression":
        parts.append(f"C{hp['C']}")
        if isinstance(hp.get("class_weight"), dict):
            parts.append(f"w{hp['class_weight'][0]}_{hp['class_weight'][1]}")
    else:
        parts.append(f"u{hp['units']}_d{hp['dropout']}_e{hp['epochs']}")
    return "_".join(parts)


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"Thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} [TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
    )


def main():
    print("=" * 80)
    print("BTCUSDT HYPERPARAMETER TUNING")
    print(f"Feature sets: {FEATURE_SETS}")
    print(f"Seq lengths: {SEQ_LENS}")
    print("=" * 80)

    results = []
    total = len(FEATURE_SETS) * len(SEQ_LENS) * sum(len(v) for v in HYPERPARAMS.values())
    run_idx = 0

    for feature_set in FEATURE_SETS:
        for seq_len in SEQ_LENS:
            extractor = TechnicalExtractor(feature_set=feature_set)
            pipeline = build_training_pipeline(extractor=extractor, seq_len=seq_len)

            for model_name, hp_list in HYPERPARAMS.items():
                for hp in hp_list:
                    run_idx += 1
                    label = make_label(model_name, feature_set, seq_len, hp)
                    print(f"\n[{run_idx}/{total}] {label}")

                    start = time.time()
                    forecaster = create_forecaster(model_name, hp)
                    try:
                        result = pipeline.train(
                            SYMBOL, forecaster, model_name=label, save_csv=(run_idx == 1)
                        )
                        elapsed = time.time() - start
                        if result.get("ok"):
                            m = result["metrics"]
                            row = {
                                "model_name": model_name,
                                "feature_set": feature_set,
                                "seq_len": seq_len,
                                "hyperparams": hp,
                                "label": label,
                                "elapsed_seconds": elapsed,
                                **m,
                            }
                            results.append(row)
                            print(f"  -> {format_metrics(m)} ({elapsed:.1f}s)")
                        else:
                            err = result.get("error", "unknown")
                            results.append({
                                "model_name": model_name,
                                "feature_set": feature_set,
                                "seq_len": seq_len,
                                "hyperparams": hp,
                                "label": label,
                                "elapsed_seconds": elapsed,
                                "error": err,
                            })
                            print(f"  -> FAILED: {err}")
                    except Exception as e:
                        elapsed = time.time() - start
                        results.append({
                            "model_name": model_name,
                            "feature_set": feature_set,
                            "seq_len": seq_len,
                            "hyperparams": hp,
                            "label": label,
                            "elapsed_seconds": elapsed,
                            "error": str(e),
                        })
                        print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_tuning_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Save best config
    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda x: x["balanced_accuracy"])
        best_config = {
            "symbol": SYMBOL,
            "model_name": best["model_name"],
            "feature_set": best["feature_set"],
            "seq_len": best["seq_len"],
            "hyperparams": best["hyperparams"],
            "label": best["label"],
            "balanced_accuracy": best["balanced_accuracy"],
            "accuracy": best["accuracy"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "threshold": best["threshold"],
        }
        best_path = os.path.join(MODELS_DIR, "btc_best_config.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best config saved to: {best_path}")

    # Print top 5 by balanced accuracy
    successful = [r for r in results if "error" not in r]
    if successful:
        print("\nTop 5 by Balanced Accuracy:")
        top = sorted(successful, key=lambda x: x["balanced_accuracy"], reverse=True)[:5]
        for r in top:
            print(f"  {r['label']:<40} -> BalAcc={r['balanced_accuracy']:.4f} Acc={r['accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
