import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_all_balanced(path, model_name=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array([f["balanced_accuracy"] for r in data for f in r["fold_metrics"]])


def main():
    configs = [
        ("BiLSTM", "btc_walkforward_bilstm_results.json"),
        ("BiLSTM+Regime", "btc_walkforward_bilstm_regime_results.json"),
        ("LSTM", "btc_walkforward_results_detailed.json"),
        ("LSTM+Regime", "btc_walkforward_lstm_regime_results.json"),
        ("RNN", "btc_walkforward_results_detailed.json"),
        ("RNN+Regime", "btc_walkforward_rnn_regime_results.json"),
        ("Logistic", "btc_walkforward_logistic_results.json"),
        ("Logistic+Regime", "btc_walkforward_logistic_regression_regime_results.json"),
    ]

    # For combined original file, need to filter by model name
    def load(name, file):
        vals = load_all_balanced(os.path.join(MODELS_DIR, file))
        if name in ("LSTM", "RNN"):
            with open(os.path.join(MODELS_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            vals = np.array([f["balanced_accuracy"] for r in data if r.get("model") == name for f in r["fold_metrics"]])
        return vals

    print("=" * 90)
    print("WALK-FORWARD BALANCED ACCURACY COMPARISON (n=50)")
    print("=" * 90)
    print(f"{'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'vs 0.5 p':>10} {'n':>5}")
    print("-" * 90)

    results = {}
    for name, file in configs:
        vals = load(name, file)
        if len(vals) == 0:
            print(f"{name:<20} NO DATA")
            continue
        t, p = stats.ttest_1samp(vals, 0.5)
        print(f"{name:<20} {vals.mean():.4f} {vals.std():.4f} {vals.min():.4f} {vals.max():.4f} {p:>10.4f} {len(vals):>5}")
        results[name] = vals

    print("\n" + "=" * 90)
    print("PAIRED T-TEST: Regime vs Original (same seed/fold)")
    print("=" * 90)
    pairs = [("BiLSTM", "BiLSTM+Regime"), ("LSTM", "LSTM+Regime"), ("RNN", "RNN+Regime"), ("Logistic", "Logistic+Regime")]
    for orig, reg in pairs:
        if orig not in results or reg not in results:
            continue
        if len(results[orig]) != len(results[reg]):
            continue
        diff = results[reg].mean() - results[orig].mean()
        t, p = stats.ttest_rel(results[reg], results[orig])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{orig} → {reg}: mean diff = {diff:+.4f}, t = {t:.4f}, p = {p:.4f} {sig}")


if __name__ == "__main__":
    main()
