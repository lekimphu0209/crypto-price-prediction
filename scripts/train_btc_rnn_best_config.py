"""
Run the best RNN configuration multiple times with different seeds and epochs.

Best config from feature selection:
    RNN, Feature Set B full, seq_len=30, units=32, dropout=0.1, batch=16, lr=0.001

Usage:
    python scripts/train_btc_rnn_best_config.py
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
RNN_HP = {
    "units": 32,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
}
SEEDS = [0, 1, 2, 3, 4]
EPOCHS_LIST = [80, 120]


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_label(epochs, seed):
    return f"RNN_best_e{epochs}_s{seed}"


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"Thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} [TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
    )


def main():
    print("=" * 80)
    print("BEST RNN CONFIG - MULTIPLE SEEDS & EPOCHS")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN)

    results = []
    total = len(SEEDS) * len(EPOCHS_LIST)
    run_idx = 0

    for epochs in EPOCHS_LIST:
        for seed in SEEDS:
            run_idx += 1
            label = make_label(epochs, seed)
            print(f"\n[{run_idx}/{total}] {label}")

            set_seed(seed)
            forecaster = KerasForecaster(name="RNN", epochs=epochs, **RNN_HP)

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
                        "epochs": epochs,
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
                        "epochs": epochs,
                        "seed": seed,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        "error": err,
                    })
                    print(f"  -> FAILED: {err}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "epochs": epochs,
                    "seed": seed,
                    "label": label,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_rnn_best_config_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda x: x["balanced_accuracy"])
        best_path = os.path.join(MODELS_DIR, "btc_best_config.json")
        best_config = {
            "symbol": SYMBOL,
            "model_name": "RNN",
            "feature_set": FEATURE_SET,
            "subset": "full_B",
            "exclude": [],
            "seq_len": SEQ_LEN,
            "epochs": best["epochs"],
            "seed": best["seed"],
            "rnn_hyperparams": RNN_HP,
            "label": best["label"],
            "balanced_accuracy": best["balanced_accuracy"],
            "accuracy": best["accuracy"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "threshold": best["threshold"],
        }
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best config updated: {best_path}")

        print("\nTop 5 by Balanced Accuracy:")
        top = sorted(successful, key=lambda x: x["balanced_accuracy"], reverse=True)[:5]
        for r in top:
            print(f"  {r['label']:<22} -> BalAcc={r['balanced_accuracy']:.4f} Acc={r['accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
