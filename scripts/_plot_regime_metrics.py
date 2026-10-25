import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)


def sig_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_grouped_bars(summary, metrics, title, ylabel, out_path, ylim=None):
    models = ["BiLSTM", "LSTM", "RNN", "Logistic"]
    sub = summary[summary["metric"].isin(metrics)].copy()

    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.35 / n_metrics
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_models * n_metrics), 5.5))

    colors_orig = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors_reg = ["#87c0e8", "#ffcc8a", "#9ee89e", "#f58e8e", "#c5a6d6"]

    for i, metric in enumerate(metrics):
        df = sub[sub["metric"] == metric]
        df = df.set_index("model").reindex(models)

        offset = (i - (n_metrics - 1) / 2) * bar_width

        orig_means = df["orig_mean"].values
        orig_stds = df["orig_std"].values
        reg_means = df["reg_mean"].values
        reg_stds = df["reg_std"].values
        ps = df["p_paired"].values

        xpos = x + offset

        bars1 = ax.bar(
            xpos - bar_width / 2,
            orig_means,
            bar_width,
            yerr=orig_stds,
            capsize=3,
            label=f"{metric} Gốc",
            color=colors_orig[i % len(colors_orig)],
            edgecolor="black",
            linewidth=0.4,
        )
        bars2 = ax.bar(
            xpos + bar_width / 2,
            reg_means,
            bar_width,
            yerr=reg_stds,
            capsize=3,
            label=f"{metric} +Regime",
            color=colors_reg[i % len(colors_reg)],
            edgecolor="black",
            linewidth=0.4,
        )

        # Add significance stars on top of the higher bar
        for j in range(n_models):
            star = sig_star(ps[j])
            if star == "ns":
                continue
            top = max(orig_means[j] + orig_stds[j], reg_means[j] + reg_stds[j])
            ax.text(
                xpos[j],
                top + 0.015,
                star,
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="upper right", ncols=n_metrics, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    json_path = os.path.join(REPORTS_DIR, "regime_metrics_comparison.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = pd.DataFrame(data["summary"])

    os.makedirs(REPORTS_DIR, exist_ok=True)

    plot_grouped_bars(
        summary,
        metrics=["accuracy"],
        title="Accuracy: Gốc vs +Regime",
        ylabel="Accuracy",
        out_path=os.path.join(REPORTS_DIR, "regime_accuracy_comparison.png"),
        ylim=(0.35, 0.75),
    )

    plot_grouped_bars(
        summary,
        metrics=["balanced_accuracy"],
        title="Balanced Accuracy: Gốc vs +Regime",
        ylabel="Balanced Accuracy",
        out_path=os.path.join(REPORTS_DIR, "regime_balanced_accuracy_comparison.png"),
        ylim=(0.35, 0.75),
    )

    plot_grouped_bars(
        summary,
        metrics=["accuracy", "balanced_accuracy"],
        title="Accuracy & Balanced Accuracy: Gốc vs +Regime",
        ylabel="Score",
        out_path=os.path.join(REPORTS_DIR, "regime_accuracy_balacc_comparison.png"),
        ylim=(0.35, 0.75),
    )

    plot_grouped_bars(
        summary,
        metrics=["precision", "recall", "f1"],
        title="Precision, Recall & F1-score: Gốc vs +Regime",
        ylabel="Score",
        out_path=os.path.join(REPORTS_DIR, "regime_precision_recall_f1_comparison.png"),
        ylim=(0.1, 0.9),
    )


if __name__ == "__main__":
    main()
