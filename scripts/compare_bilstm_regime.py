"""
Compare BiLSTM baseline vs BiLSTM with regime features from saved JSON results.

Usage:
    python scripts/compare_bilstm_regime.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import MODELS_DIR


def main():
    base_path = os.path.join(MODELS_DIR, "_bilstm_baseline_btcusdt.json")
    regime_path = os.path.join(MODELS_DIR, "_bilstm_regime_btcusdt.json")

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(regime_path, "r", encoding="utf-8") as f:
        regime = json.load(f)

    symbol = "BTCUSDT"
    b = base[symbol]["BiLSTM_Baseline"]
    r = regime[symbol]["BiLSTM_Regime"]

    print("=" * 70)
    print("BiLSTM RESULTS COMPARISON (BTCUSDT, 20 epochs)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':>14} {'Regime':>14} {'Delta':>12}")
    print("-" * 70)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "threshold"]
    for m in metrics:
        bv = b[m]
        rv = r[m]
        delta = rv - bv
        print(f"{m:<20} {bv:>14.4f} {rv:>14.4f} {delta:>+12.4f}")
    print("-" * 70)
    print(f"{'Confusion':<20} Baseline: TN={b['tn']} FP={b['fp']} FN={b['fn']} TP={b['tp']}")
    print(f"{'':20} Regime:   TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
