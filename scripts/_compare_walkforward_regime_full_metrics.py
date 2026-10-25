import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]

CONFIGS = [
    ("BiLSTM", "btc_walkforward_bilstm_results.json", "BiLSTM"),
    ("BiLSTM", "btc_walkforward_bilstm_regime_results.json", "BiLSTM_Regime"),
    ("LSTM", "btc_walkforward_results_detailed.json", "LSTM"),
    ("LSTM", "btc_walkforward_lstm_regime_results.json", "LSTM_Regime"),
    ("RNN", "btc_walkforward_results_detailed.json", "RNN"),
    ("RNN", "btc_walkforward_rnn_regime_results.json", "RNN_Regime"),
    ("Logistic", "btc_walkforward_logistic_results.json", "LogisticRegression"),
    ("Logistic", "btc_walkforward_logistic_regression_regime_results.json", "Logistic Regression_Regime"),
]


def load_values(path, model_name, metric):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = []
    for r in data:
        if r.get("model") != model_name:
            continue
        for f in r["fold_metrics"]:
            values.append((r["seed"], f["fold"], f[metric]))
    df = pd.DataFrame(values, columns=["seed", "fold", metric])
    return df


def main():
    models = ["BiLSTM", "LSTM", "RNN", "Logistic"]
    records = []
    ttest_records = []

    for model in models:
        orig_rows = [c for c in CONFIGS if c[0] == model and "_Regime" not in c[2]]
        reg_rows = [c for c in CONFIGS if c[0] == model and "_Regime" in c[2]]
        if not orig_rows or not reg_rows:
            continue

        orig_path = os.path.join(MODELS_DIR, orig_rows[0][1])
        reg_path = os.path.join(MODELS_DIR, reg_rows[0][1])

        for metric in METRICS:
            orig_df = load_values(orig_path, orig_rows[0][2], metric)
            reg_df = load_values(reg_path, reg_rows[0][2], metric)

            merged = pd.merge(orig_df, reg_df, on=["seed", "fold"], suffixes=("_orig", "_reg"))

            orig_vals = merged[metric + "_orig"].values
            reg_vals = merged[metric + "_reg"].values

            t, p = stats.ttest_rel(reg_vals, orig_vals) if len(merged) > 1 else (np.nan, np.nan)

            records.append({
                "model": model,
                "metric": metric,
                "orig_mean": orig_vals.mean(),
                "orig_std": orig_vals.std(),
                "orig_min": orig_vals.min(),
                "orig_max": orig_vals.max(),
                "reg_mean": reg_vals.mean(),
                "reg_std": reg_vals.std(),
                "reg_min": reg_vals.min(),
                "reg_max": reg_vals.max(),
                "delta_mean": reg_vals.mean() - orig_vals.mean(),
                "delta_std": reg_vals.std() - orig_vals.std(),
                "p_paired": p,
                "n": len(merged),
            })

            ttest_records.append({
                "model": model,
                "metric": metric,
                "delta_mean": reg_vals.mean() - orig_vals.mean(),
                "t_stat": t,
                "p_value": p,
                "significant": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns",
            })

    summary = pd.DataFrame(records)
    ttest = pd.DataFrame(ttest_records)

    print("\n" + "=" * 100)
    print("FULL METRICS COMPARISON: ORIGINAL vs +REGIME (n=50)")
    print("=" * 100)

    for model in models:
        print(f"\n{model}")
        print("-" * 100)
        sub = summary[summary["model"] == model]
        for _, row in sub.iterrows():
            metric = row["metric"]
            print(f"  {metric:18s}: Gốc = {row['orig_mean']:.4f}±{row['orig_std']:.4f}  "
                  f"Regime = {row['reg_mean']:.4f}±{row['reg_std']:.4f}  "
                  f"Δ = {row['delta_mean']:+.4f}  "
                  f"p = {row['p_paired']:.4f} ({'***' if row['p_paired'] < 0.001 else '**' if row['p_paired'] < 0.01 else '*' if row['p_paired'] < 0.05 else 'ns'})")

    # Save
    out_dir = os.path.join(MODELS_DIR, "..")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    csv_path = os.path.join(out_dir, "reports", "regime_metrics_comparison.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    json_path = os.path.join(out_dir, "reports", "regime_metrics_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary.to_dict(orient="records"),
            "ttest": ttest.to_dict(orient="records"),
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Pretty table per metric
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (Mean ± Std)")
    print("=" * 100)
    for metric in METRICS:
        print(f"\n[{metric.upper()}]")
        sub = summary[summary["metric"] == metric].sort_values("model")
        print(f"{'Model':<12} {'Gốc':<18} {'Regime':<18} {'Δ mean':<10} {'p-value':<10} {'Sig':<5}")
        print("-" * 90)
        for _, row in sub.iterrows():
            sig = '***' if row['p_paired'] < 0.001 else '**' if row['p_paired'] < 0.01 else '*' if row['p_paired'] < 0.05 else 'ns'
            print(f"{row['model']:<12} {row['orig_mean']:.4f}±{row['orig_std']:.4f}    {row['reg_mean']:.4f}±{row['reg_std']:.4f}    {row['delta_mean']:+.4f}     {row['p_paired']:.4f}     {sig:<5}")


if __name__ == "__main__":
    main()
