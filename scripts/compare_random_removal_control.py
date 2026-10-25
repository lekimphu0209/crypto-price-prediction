"""Compare BiLSTM original vs threshold vs random-removal control."""
import json
import numpy as np
from scipy import stats
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")) as f:
    orig = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_threshold_results.json")) as f:
    thresh = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_random_removal_results.json")) as f:
    rand = json.load(f)

original = np.array([f["balanced_accuracy"] for r in orig for f in r["fold_metrics"]])
threshold = np.array([f["balanced_accuracy"] for r in thresh for f in r["fold_metrics"]])
random_removal = np.array([f["balanced_accuracy"] for r in rand for f in r["fold_metrics"]])

print("=" * 80)
print("CONTROL EXPERIMENT: Random removal vs Threshold vs Original")
print("=" * 80)
print(f"\nBiLSTM original:      {original.mean():.4f} ± {original.std():.4f} (n={len(original)})")
print(f"BiLSTM random removal: {random_removal.mean():.4f} ± {random_removal.std():.4f} (n={len(random_removal)})")
print(f"BiLSTM threshold 1%:  {threshold.mean():.4f} ± {threshold.std():.4f} (n={len(threshold)})")

# Paired comparisons
comparisons = [
    ("Original", "Random removal", original, random_removal),
    ("Original", "Threshold", original, threshold),
    ("Random removal", "Threshold", random_removal, threshold),
]

print("\n" + "=" * 80)
print("Paired t-tests")
print("=" * 80)
for a_name, b_name, a, b in comparisons:
    t, p = stats.ttest_rel(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    winner = a_name if a.mean() > b.mean() else b_name
    diff = abs(a.mean() - b.mean())
    print(f"{a_name} vs {b_name}: diff={diff:.4f}, t={t:.4f}, p={p:.6f} {sig} → winner: {winner}")

# vs baseline
print("\n" + "=" * 80)
print("One-sample t-test vs Baseline (0.5)")
print("=" * 80)
for name, vals in [
    ("Original", original),
    ("Random removal", random_removal),
    ("Threshold", threshold),
]:
    t, p = stats.ttest_1samp(vals, 0.5)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{name:20s}: {vals.mean():.4f} t={t:.4f}, p={p:.6f} {sig}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("• Random removal (0.5568) tốt hơn threshold (0.5435).")
print("• Điều này nghĩa là việc loại bỏ mẫu NGẪU NHIÊN gây ít hại hơn")
print("  so với việc loại bỏ cụ thể các mẫu ở vùng trung lập (threshold).")
print("• Tuy nhiên, cả random removal và threshold đều kém hơn original (0.5730).")
print("  → Data loss là một phần nguyên nhân, nhưng threshold còn loại bỏ")
print("    những mẫu có ích cho mô hình.")
print("\nQuyết định: Không nên dùng threshold 1% hoặc random removal.")
print("Nên tập trung vào weighted loss hoặc cải thiện feature/model thay vì")
print("loại bỏ nhãn trung lập.")
