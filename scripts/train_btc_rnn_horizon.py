"""
Test shorter prediction horizons for RNN on BTCUSDT.

Compares horizons 1, 3, and 7 days using the best feature set (B full)
and best seq_len (30) from previous experiments.

Usage:
    python scripts/train_btc_rnn_horizon.py
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
HORIZONS = [1, 3, 7]
SEEDS = [0, 1]

RNN_HP = {
    "units": 32,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
}
EPOCHS = 80


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_label(horizon, seed):
    return f"RNN_h{horizon}_s{seed}"


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"Thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f}"
    )


def main():
    print("=" * 80)
    print("RNN + SET B + SEQ_LEN=30 + HORIZON COMPARISON")
    print("=" * 80)

    results = []
    total = len(HORIZONS) * len(SEEDS)
    run_idx = 0

    for horizon in HORIZONS:
        extractor = TechnicalExtractor(feature_set=FEATURE_SET)
        pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=horizon)

        for seed in SEEDS:
            run_idx += 1
            label = make_label(horizon, seed)
            print(f"\n[{run_idx}/{total}] {label}")

            set_seed(seed)
            forecaster = KerasForecaster(name="RNN", epochs=EPOCHS, **RNN_HP)

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
                        "horizon": horizon,
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
                        "horizon": horizon,
                        "seed": seed,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        "error": err,
                    })
                    print(f"  -> FAILED: {err}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "horizon": horizon,
                    "seed": seed,
                    "label": label,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_rnn_horizon_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    successful = [r for r in results if "error" not in r]
    if successful:
        print("\nTop results by Balanced Accuracy:")
        top = sorted(successful, key=lambda x: x["balanced_accuracy"], reverse=True)
        for r in top:
            print(f"  {r['label']:<18} -> BalAcc={r['balanced_accuracy']:.4f} Acc={r['accuracy']:.4f}")

        # Average per horizon
        print("\nAverage per horizon:")
        for h in HORIZONS:
            rows = [r for r in successful if r["horizon"] == h]
            if rows:
                avg_bal = sum(r["balanced_accuracy"] for r in rows) / len(rows)
                avg_acc = sum(r["accuracy"] for r in rows) / len(rows)
                print(f"  horizon={h}: Avg BalAcc={avg_bal:.4f} Avg Acc={avg_acc:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
