"""
Feature selection for RNN on BTCUSDT using Feature Set B.

Tests different subsets of Set B to find the most informative features.
Uses the best RNN hyperparameters from the previous tuning round.

Usage:
    python scripts/train_btc_rnn_feature_selection.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.features.technical_extractor import TechnicalExtractor
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
SEQ_LENS = [20, 30]

# RNN with the best hyperparameters from the previous tuning round.
RNN_HP = {
    "units": 32,
    "dropout": 0.1,
    "epochs": 80,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
}

# Different feature subsets to test on Feature Set B.
SUBSETS = [
    {"name": "full_B", "exclude": []},
    {"name": "no_abs_price", "exclude": ["close", "ma_7", "ma_14", "bb_upper", "bb_middle", "bb_lower"]},
    {"name": "no_abs_macro", "exclude": ["gold", "dxy", "usd_rate"]},
    {"name": "no_log_return_1", "exclude": ["log_return_1"]},
    {"name": "no_volume", "exclude": ["volume"]},
    {"name": "no_roc", "exclude": ["roc"]},
    {"name": "stationary_only", "exclude": [
        "close", "volume", "ma_7", "ma_14",
        "bb_upper", "bb_middle", "bb_lower",
        "gold", "dxy", "usd_rate",
    ]},
    {"name": "lean_stationary", "exclude": [
        "close", "volume", "ma_7", "ma_14",
        "bb_upper", "bb_middle", "bb_lower",
        "gold", "dxy", "usd_rate", "log_return_1",
    ]},
]


def make_label(subset_name, seq_len):
    return f"RNN_{subset_name}_sl{seq_len}"


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"Thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} [TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
    )


def main():
    print("=" * 80)
    print("RNN + FEATURE SET B + FEATURE SELECTION")
    print("=" * 80)

    results = []
    total = len(SEQ_LENS) * len(SUBSETS)
    run_idx = 0

    for seq_len in SEQ_LENS:
        for subset in SUBSETS:
            run_idx += 1
            label = make_label(subset["name"], seq_len)
            print(f"\n[{run_idx}/{total}] {label}")

            extractor = TechnicalExtractor(
                feature_set=FEATURE_SET,
                exclude_features=subset["exclude"],
            )
            n_features = len(extractor.feature_names())
            print(f"  Features used: {n_features}")

            pipeline = build_training_pipeline(extractor=extractor, seq_len=seq_len)
            forecaster = KerasForecaster(name="RNN", **RNN_HP)

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
                        "subset": subset["name"],
                        "exclude": subset["exclude"],
                        "n_features": n_features,
                        "seq_len": seq_len,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        **m,
                    }
                    results.append(row)
                    print(f"  -> {format_metrics(m)} ({elapsed:.1f}s)")
                else:
                    err = result.get("error", "unknown")
                    results.append({
                        "subset": subset["name"],
                        "exclude": subset["exclude"],
                        "n_features": n_features,
                        "seq_len": seq_len,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        "error": err,
                    })
                    print(f"  -> FAILED: {err}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "subset": subset["name"],
                    "exclude": subset["exclude"],
                    "n_features": n_features,
                    "seq_len": seq_len,
                    "label": label,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_rnn_feature_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda x: x["balanced_accuracy"])
        best_config = {
            "symbol": SYMBOL,
            "model_name": "RNN",
            "feature_set": FEATURE_SET,
            "subset": best["subset"],
            "exclude": best["exclude"],
            "n_features": best["n_features"],
            "seq_len": best["seq_len"],
            "rnn_hyperparams": RNN_HP,
            "label": best["label"],
            "balanced_accuracy": best["balanced_accuracy"],
            "accuracy": best["accuracy"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "threshold": best["threshold"],
        }
        best_path = os.path.join(MODELS_DIR, "btc_rnn_best_feature_subset.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best feature subset saved to: {best_path}")

        print("\nTop 5 by Balanced Accuracy:")
        top = sorted(successful, key=lambda x: x["balanced_accuracy"], reverse=True)[:5]
        for r in top:
            print(f"  {r['label']:<42} n={r['n_features']:2d} -> BalAcc={r['balanced_accuracy']:.4f} Acc={r['accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
