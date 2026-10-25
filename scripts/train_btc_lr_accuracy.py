"""
Optimize Logistic Regression accuracy for BTCUSDT 7-day direction.

Grid searches C and class_weight on Feature Set B, uses the training pipeline
with threshold_metric="accuracy" to tune the decision threshold for maximum
accuracy, and reports test-set metrics.

Usage:
    python scripts/train_btc_lr_accuracy.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, MODELS_DIR
from src.features.technical_extractor import TechnicalExtractor
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
SEQ_LEN = 30
HORIZON = 7

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = ["balanced", None]


def format_metrics(m):
    return (
        f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
        f"Thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} [TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]"
    )


def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION + SET B + ACCURACY OPTIMIZATION")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    pipeline = build_training_pipeline(
        extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON, threshold_metric="accuracy"
    )

    results = []
    total = len(C_VALUES) * len(CLASS_WEIGHTS)
    run_idx = 0

    for class_weight in CLASS_WEIGHTS:
        for C in C_VALUES:
            run_idx += 1
            cw_label = "none" if class_weight is None else str(class_weight)
            label = f"LR_C{C}_cw{cw_label}"
            print(f"\n[{run_idx}/{total}] {label}")

            forecaster = LinearForecaster(C=C, class_weight=class_weight)

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
                        "C": C,
                        "class_weight": cw_label,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        **m,
                    }
                    results.append(row)
                    print(f"  -> {format_metrics(m)} ({elapsed:.1f}s)")
                else:
                    err = result.get("error", "unknown")
                    results.append({
                        "C": C,
                        "class_weight": cw_label,
                        "label": label,
                        "elapsed_seconds": elapsed,
                        "error": err,
                    })
                    print(f"  -> FAILED: {err}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "C": C,
                    "class_weight": cw_label,
                    "label": label,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                print(f"  -> ERROR: {e}")

    out_path = os.path.join(MODELS_DIR, "btc_lr_accuracy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    successful = [r for r in results if "error" not in r]
    if successful:
        print("\nTop 3 by test Accuracy:")
        top = sorted(successful, key=lambda x: x["accuracy"], reverse=True)
        for r in top[:3]:
            print(f"  {r['label']:<18} -> Acc={r['accuracy']:.4f} BalAcc={r['balanced_accuracy']:.4f}")

        print("\nTop 3 by test Balanced Accuracy:")
        top_bal = sorted(successful, key=lambda x: x["balanced_accuracy"], reverse=True)
        for r in top_bal[:3]:
            print(f"  {r['label']:<18} -> BalAcc={r['balanced_accuracy']:.4f} Acc={r['accuracy']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
