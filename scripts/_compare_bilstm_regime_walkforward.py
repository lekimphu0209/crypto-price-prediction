import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def flat_balanced_accuracy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array([f["balanced_accuracy"] for r in data for f in r["fold_metrics"]])


def main():
    orig_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")
    reg_path = os.path.join(MODELS_DIR, "btc_walkforward_bilstm_regime_results.json")

    orig = flat_balanced_accuracy(orig_path)
    reg = flat_balanced_accuracy(reg_path)

    print("=" * 70)
    print("WALK-FORWARD BiLSTM COMPARISON: ORIGINAL vs +REGIME")
    print("=" * 70)
    print(f"n (runs): {len(orig)} original, {len(reg)} regime")
    print()
    print("Metric: balanced_accuracy")
    print(f"  Original BiLSTM   -> mean={orig.mean():.4f}  std={orig.std():.4f}  min={orig.min():.4f}  max={orig.max():.4f}")
    print(f"  BiLSTM + Regime   -> mean={reg.mean():.4f}  std={reg.std():.4f}  min={reg.min():.4f}  max={reg.max():.4f}")
    print()
    print("One-sample t-test vs 0.5")
    t, p = stats.ttest_1samp(orig, 0.5)
    print(f"  Original  : t={t:.4f}, p={p:.4f}, significant={'YES' if p < 0.05 else 'NO'}")
    t, p = stats.ttest_1samp(reg, 0.5)
    print(f"  Regime    : t={t:.4f}, p={p:.4f}, significant={'YES' if p < 0.05 else 'NO'}")
    print()
    print("Paired t-test (Regime - Original)")
    if len(orig) == len(reg):
        t, p = stats.ttest_rel(reg, orig)
        diff = reg.mean() - orig.mean()
        print(f"  Mean diff = {diff:+.4f}")
        print(f"  t={t:.4f}, p={p:.4f}, significant={'YES' if p < 0.05 else 'NO'}")
    else:
        print("  Cannot pair: different n")


if __name__ == "__main__":
    main()
