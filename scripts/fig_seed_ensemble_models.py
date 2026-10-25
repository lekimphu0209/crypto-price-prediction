"""
Seed Ensemble comparison across BiLSTM, LSTM, RNN, and Logistic Regression.

Uses 5-fold, 10-seed seed ensemble results for each model.
Output: docs/figures/seed_ensemble_models.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    models = ["BiLSTM", "LSTM", "RNN", "Logistic"]
    means = np.array([0.5443, 0.5262, 0.4654, 0.4403])
    stds = np.array([0.0434, 0.0483, 0.0462, 0.0986])
    mins = np.array([0.4870, 0.4471, 0.4219, 0.2979])
    maxs = np.array([0.6034, 0.5865, 0.5548, 0.5689])
    p_values = [0.1108, 0.3397, 0.2089, 0.2925]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=1.2, error_kw={"elinewidth": 1.5})

    # Add min-max whiskers
    for i in range(len(models)):
        ax.plot([x[i], x[i]], [mins[i], maxs[i]], color="black", linewidth=1.0)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [mins[i], mins[i]], color="black", linewidth=1.0)
        ax.plot([x[i] - 0.08, x[i] + 0.08], [maxs[i], maxs[i]], color="black", linewidth=1.0)

    # Baseline
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Random baseline (0.5)")

    # Labels and value annotations
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Seed Ensemble (5 folds × 10 seeds) across Models", fontsize=14, weight="bold")
    ax.set_ylim(0.25, 0.70)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)

    for i, (m, s, p) in enumerate(zip(means, stds, p_values)):
        sig = "*" if p < 0.05 else "ns"
        ax.text(i, m + s + 0.02, f"{m:.3f}\n(p={p:.3f} {sig})",
                ha="center", va="bottom", fontsize=9, weight="bold")

    ax.text(0.5, -0.12, "Error bars: ±1 std; whiskers: min-max over 5 folds. "
                         "'ns' = not significant vs baseline 0.5 (α=0.05).",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "seed_ensemble_models.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
