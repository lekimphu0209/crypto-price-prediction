"""
Illustration of Balanced Accuracy results (Walk-Forward Validation, n=50).

Bar chart with std error bars, min-max range whiskers, and baseline line.
Output: docs/figures/balanced_accuracy_comparison.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")

MODELS = ["BiLSTM", "LSTM", "RNN", "Logistic\nRegression"]
MEAN = np.array([0.5730, 0.5524, 0.4727, 0.4619])
STD = np.array([0.0801, 0.0556, 0.0941, 0.0854])
MIN = np.array([0.3815, 0.3815, 0.2791, 0.3036])
MAX = np.array([0.7597, 0.7241, 0.6715, 0.6051])
BASELINE = 0.5


def main():
    x = np.arange(len(MODELS))
    # Color: green if above baseline, red if below
    colors = ["#2ca02c" if m >= BASELINE else "#d62728" for m in MEAN]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Min-max range as light whiskers
    for i in range(len(MODELS)):
        ax.plot([x[i], x[i]], [MIN[i], MAX[i]], color="gray", linewidth=1.2, alpha=0.6, zorder=1)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [MIN[i], MIN[i]], color="gray", linewidth=1.2, alpha=0.6)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [MAX[i], MAX[i]], color="gray", linewidth=1.2, alpha=0.6)

    bars = ax.bar(x, MEAN, width=0.5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=1, yerr=STD, capsize=6,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"}, zorder=2)

    # Baseline line
    ax.axhline(BASELINE, color="black", linestyle="--", linewidth=1.5, label="Baseline (0.5)", zorder=0)

    # Value labels on bars
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, MEAN[i] + STD[i] + 0.015,
                f"{MEAN[i]:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Balanced Accuracy theo mô hình — Walk-Forward Validation (n=50)\n"
                 "Cột = Mean ± Std, thanh xám = khoảng Min–Max", fontsize=12)
    ax.set_ylim(0.25, 0.80)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "balanced_accuracy_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
