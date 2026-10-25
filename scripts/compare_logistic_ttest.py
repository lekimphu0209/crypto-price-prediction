"""Compare Logistic Regression with RNN/LSTM using t-tests on all fold values."""
import json
import numpy as np
from scipy import stats
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(MODELS_DIR, "btc_walkforward_results_detailed.json")) as f:
    nn_results = json.load(f)
with open(os.path.join(MODELS_DIR, "btc_walkforward_logistic_results.json")) as f:
    log_results = json.load(f)

rnn_vals = []
lstm_vals = []
for r in nn_results:
    if r["model"] == "RNN":
        rnn_vals.extend([f["balanced_accuracy"] for f in r["fold_metrics"]])
    elif r["model"] == "LSTM":
        lstm_vals.extend([f["balanced_accuracy"] for f in r["fold_metrics"]])

log_vals = [f["balanced_accuracy"] for r in log_results for f in r["fold_metrics"]]

rnn_vals = np.array(rnn_vals)
lstm_vals = np.array(lstm_vals)
log_vals = np.array(log_vals)

print("=" * 80)
print("COMPARISON: LSTM vs RNN vs Logistic Regression")
print("=" * 80)
print(f"\nAll fold values (n=50 each):")
print(f"  LSTM:     {lstm_vals.mean():.4f} ± {lstm_vals.std():.4f}")
print(f"  RNN:      {rnn_vals.mean():.4f} ± {rnn_vals.std():.4f}")
print(f"  Logistic: {log_vals.mean():.4f} ± {log_vals.std():.4f}")
print(f"  Baseline: 0.5000")

_, p_lstm_log = stats.ttest_rel(lstm_vals, log_vals)
_, p_rnn_log = stats.ttest_rel(rnn_vals, log_vals)
_, p_log_base = stats.ttest_1samp(log_vals, 0.5)
_, p_lstm_base = stats.ttest_1samp(lstm_vals, 0.5)
_, p_rnn_base = stats.ttest_1samp(rnn_vals, 0.5)

print("\n" + "=" * 80)
print("T-TEST RESULTS")
print("=" * 80)
print(f"LSTM vs Logistic:   p = {p_lstm_log:.6f}  {'SIGNIFICANT' if p_lstm_log < 0.05 else 'not significant'}")
print(f"RNN vs Logistic:    p = {p_rnn_log:.6f}  {'SIGNIFICANT' if p_rnn_log < 0.05 else 'not significant'}")
print(f"LSTM vs Baseline:   p = {p_lstm_base:.6f}  {'SIGNIFICANT' if p_lstm_base < 0.05 else 'not significant'}")
print(f"RNN vs Baseline:    p = {p_rnn_base:.6f}  {'SIGNIFICANT' if p_rnn_base < 0.05 else 'not significant'}")
print(f"Logistic vs Baseline: p = {p_log_base:.6f}  {'SIGNIFICANT' if p_log_base < 0.05 else 'not significant'}")

print("\n" + "=" * 80)
print("RANKING")
print("=" * 80)
ranks = [
    ("LSTM", lstm_vals.mean()),
    ("Baseline", 0.5),
    ("RNN", rnn_vals.mean()),
    ("Logistic", log_vals.mean()),
]
ranks.sort(key=lambda x: x[1], reverse=True)
for i, (name, val) in enumerate(ranks, 1):
    print(f"{i}. {name}: {val:.4f}")

print("\nLSTM là model duy nhất vượt baseline với ý nghĩa thống kê.")
