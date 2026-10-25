"""
Illustration of Accuracy results (Walk-Forward Validation, n=50) - section 4.1.2.

Bar chart with std error bars, min-max range whiskers, random baseline (0.5)
and majority-class baseline (~0.55, UP proportion).
Output: docs/figures/accuracy_comparison.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")

MODELS = ["BiLSTM", "LSTM", "Logistic\nRegression", "RNN"]
MEAN = np.array([0.5188, 0.4956, 0.4232, 0.4204])
STD = np.array([0.0980, 0.0811, 0.0922, 0.1002])
MIN = np.array([0.3600, 0.3600, 0.2800, 0.2400])
MAX = np.array([0.7800, 0.6800, 0.5800, 0.6800])
RANDOM_BASELINE = 0.5
MAJORITY_BASELINE = 0.55  # UP class proportion


def main():
    x = np.arange(len(MODELS))
    colors = ["#2ca02c" if m >= RANDOM_BASELINE else "#d62728" for m in MEAN]

    fig, ax = plt.subplots(figsize=(9, 6))

    for i in range(len(MODELS)):
        ax.plot([x[i], x[i]], [MIN[i], MAX[i]], color="gray", linewidth=1.2, alpha=0.6, zorder=1)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [MIN[i], MIN[i]], color="gray", linewidth=1.2, alpha=0.6)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [MAX[i], MAX[i]], color="gray", linewidth=1.2, alpha=0.6)

    bars = ax.bar(x, MEAN, width=0.5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=1, yerr=STD, capsize=6,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"}, zorder=2)

    ax.axhline(RANDOM_BASELINE, color="black", linestyle="--", linewidth=1.5,
               label="Baseline ngẫu nhiên (0.5)", zorder=0)
    ax.axhline(MAJORITY_BASELINE, color="#ff7f0e", linestyle=":", linewidth=1.8,
               label="Baseline lớp đa số (~0.55)", zorder=0)

    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, MEAN[i] + STD[i] + 0.015,
                f"{MEAN[i]:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy theo mô hình — Walk-Forward Validation (n=50)\n"
                 "Cột = Mean ± Std, thanh xám = khoảng Min–Max", fontsize=12)
    ax.set_ylim(0.20, 0.85)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "accuracy_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
