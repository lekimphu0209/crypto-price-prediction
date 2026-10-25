"""
Compute and visualize experimental results for section 4.2 of the report.

Outputs:
- docs/figures/model_results.png (bar charts for BalAcc, Acc, Precision, Recall, F1)
- prints markdown tables to stdout
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def load_model(name, file_name):
    with open(os.path.join(MODELS_DIR, file_name)) as f:
        data = json.load(f)
    records = [f for r in data if r.get("model") == name for f in r["fold_metrics"]]
    if not records:
        records = [f for r in data for f in r["fold_metrics"]]
    return records


def stat_dict(records):
    metrics = ["balanced_accuracy", "accuracy", "precision", "recall", "f1"]
    out = {}
    for m in metrics:
        vals = np.array([r[m] for r in records])
        out[m] = {
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
        }
    return out


def main():
    models = {
        "BiLSTM": load_model("BiLSTM", "btc_walkforward_bilstm_results.json"),
        "LSTM": load_model("LSTM", "btc_walkforward_results_detailed.json"),
        "RNN": load_model("RNN", "btc_walkforward_results_detailed.json"),
        "Logistic": load_model("Logistic", "btc_walkforward_logistic_results.json"),
    }

    stats = {name: stat_dict(recs) for name, recs in models.items()}

    # Print markdown tables
    print("# 4.2. Kết quả thực nghiệm\n")

    print("## 4.2.1. Balanced Accuracy")
    print("| Model | Mean | Std | Min | Max |")
    print("|---|---|---|---|---|")
    for name in models:
        s = stats[name]["balanced_accuracy"]
        print(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

    print("\n## 4.2.2. Accuracy")
    print("| Model | Mean | Std | Min | Max |")
    print("|---|---|---|---|---|")
    for name in models:
        s = stats[name]["accuracy"]
        print(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

    print("\n## 4.2.3. Precision, Recall, F1-score")
    print("| Model | Precision | Recall | F1 |")
    print("|---|---|---|---|")
    for name in models:
        p = stats[name]["precision"]["mean"]
        r = stats[name]["recall"]["mean"]
        f = stats[name]["f1"]["mean"]
        print(f"| {name} | {p:.4f} | {r:.4f} | {f:.4f} |")

    # Figure
    names = list(models.keys())
    metrics = ["balanced_accuracy", "accuracy", "precision", "recall", "f1"]
    titles = {
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    }
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = [stats[n][metric]["mean"] for n in names]
        stds = [stats[n][metric]["std"] for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.75, edgecolor="black")
        ax.set_title(titles[metric])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    # Hide the 6th unused subplot
    axes[5].axis("off")

    fig.suptitle("Model Performance Comparison (Mean ± Std over 50 fold × seed)", fontsize=14, weight="bold")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "model_results.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
