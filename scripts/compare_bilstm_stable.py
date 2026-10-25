"""Compare BiLSTM stable vs original."""
import json
import numpy as np
from scipy import stats
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_results.json")) as f:
    orig = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_stable_results.json")) as f:
    stable = json.load(f)

original = np.array([f["balanced_accuracy"] for r in orig for f in r["fold_metrics"]])
stable_vals = np.array([f["balanced_accuracy"] for r in stable for f in r["fold_metrics"]])

print("=" * 80)
print("BiLSTM ORIGINAL vs BiLSTM STABLE (no ReduceLROnPlateau)")
print("=" * 80)
print(f"\nOriginal:   {original.mean():.4f} ± {original.std():.4f} (n={len(original)})")
print(f"Stable:     {stable_vals.mean():.4f} ± {stable_vals.std():.4f} (n={len(stable_vals)})")

t, p = stats.ttest_rel(original, stable_vals)
print(f"\nPaired t-test:")
print(f"  diff: {original.mean() - stable_vals.mean():.4f}")
print(f"  t: {t:.4f}")
print(f"  p: {p:.6f}")
print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")

for name, vals in [("Original", original), ("Stable", stable_vals)]:
    t, p = stats.ttest_1samp(vals, 0.5)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"\n{name} vs baseline (0.5):")
    print(f"  mean: {vals.mean():.4f}")
    print(f"  t: {t:.4f}, p: {p:.6f} {sig}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("Loại bỏ ReduceLROnPlateau vẫn không cứu vãn được.")
print("Collapse tiếp tục: stable 0.4944 vs original 0.5730.")
print("Nguyên nhân giờ nghi ngờ: validation_split + early_stopping + checkpoint")
print("hoặc architecture 64 units + recurrent_dropout quá mạnh cho data nhỏ.")
print("\nKhuyến nghị: Chạy config THẬT sát gốc:")
print("  units=32, dropout=0.1, KHÔNG recurrent_dropout,")
print("  KHÔNG validation_split, KHÔNG early_stopping, KHÔNG checkpoint,")
print("  chỉ giữ nguyên như bản gốc đã chứng minh 0.5730.")
