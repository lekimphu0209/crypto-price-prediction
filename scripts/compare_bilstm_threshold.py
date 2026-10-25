"""Compare BiLSTM with threshold labels vs original BiLSTM."""
import json
import numpy as np
from scipy import stats
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")) as f:
    bilstm_data = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_threshold_results.json")) as f:
    threshold_data = json.load(f)

bilstm = np.array([f["balanced_accuracy"] for r in bilstm_data for f in r["fold_metrics"]])
threshold = np.array([f["balanced_accuracy"] for r in threshold_data for f in r["fold_metrics"]])

print("=" * 80)
print("BiLSTM ORIGINAL vs BiLSTM + 1% THRESHOLD LABEL")
print("=" * 80)
print(f"\nBiLSTM original:  {bilstm.mean():.4f} ± {bilstm.std():.4f} (n={len(bilstm)})")
print(f"BiLSTM threshold: {threshold.mean():.4f} ± {threshold.std():.4f} (n={len(threshold)})")

t, p = stats.ttest_rel(bilstm, threshold)
print(f"\nPaired t-test (original vs threshold):")
print(f"  diff: {bilstm.mean() - threshold.mean():.4f}")
print(f"  t: {t:.4f}")
print(f"  p: {p:.6f}")
print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")

# vs baseline
for name, vals in [("BiLSTM original", bilstm), ("BiLSTM threshold", threshold)]:
    t, p = stats.ttest_1samp(vals, 0.5)
    print(f"\n{name} vs baseline (0.5):")
    print(f"  mean: {vals.mean():.4f}")
    print(f"  t: {t:.4f}, p: {p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if bilstm.mean() > threshold.mean() and p < 0.05:
    print("BiLSTM original tốt hơn đáng kể so với threshold label 1%.")
elif bilstm.mean() > threshold.mean():
    print("BiLSTM original tốt hơn nhưng chưa đạt ý nghĩa thống kê.")
else:
    print("BiLSTM threshold tốt hơn.")
