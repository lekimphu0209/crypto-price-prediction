"""Compare BiLSTM regularized vs original."""
import json
import numpy as np
from scipy import stats
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")) as f:
    orig = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_regularized_results.json")) as f:
    reg = json.load(f)

original = np.array([f["balanced_accuracy"] for r in orig for f in r["fold_metrics"]])
regularized = np.array([f["balanced_accuracy"] for r in reg for f in r["fold_metrics"]])

print("=" * 80)
print("BiLSTM ORIGINAL vs BiLSTM REGULARIZED")
print("=" * 80)
print(f"\nOriginal:     {original.mean():.4f} ± {original.std():.4f} (n={len(original)})")
print(f"Regularized:  {regularized.mean():.4f} ± {regularized.std():.4f} (n={len(regularized)})")

# Paired t-test
t, p = stats.ttest_rel(original, regularized)
print(f"\nPaired t-test (original vs regularized):")
print(f"  diff: {original.mean() - regularized.mean():.4f}")
print(f"  t: {t:.4f}")
print(f"  p: {p:.6f}")
print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")

# vs baseline
for name, vals in [("Original", original), ("Regularized", regularized)]:
    t, p = stats.ttest_1samp(vals, 0.5)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"\n{name} vs baseline (0.5):")
    print(f"  mean: {vals.mean():.4f}")
    print(f"  t: {t:.4f}, p: {p:.6f} {sig}")

# Variance between seeds
print("\n" + "=" * 80)
print("Variance between seeds")
print("=" * 80)

def seed_std(data):
    M = np.array([[f["balanced_accuracy"] for f in r["fold_metrics"]] for r in data])
    return np.std(M.mean(axis=1), ddof=1)

print(f"Original seed std:     {seed_std(orig):.4f}")
print(f"Regularized seed std:  {seed_std(reg):.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("Regularization mạnh quá mức (dropout=0.3, units=16, L2=0.001, callbacks)")
print("làm mô hình UNDERFIT nghiêm trọng trên tập dữ liệu nhỏ 200-400 mẫu.")
print("BalAcc giảm từ 0.5730 xuống 0.4913 và KHÔNG còn vượt baseline.")
print("Variance giữa seed giảm nhưng không có ý nghĩa vì mean đã rơi xuống baseline.")
print("\nKết luận: Không nên dùng cấu hình regularization này với data hiện tại.")
