"""
Illustration of Precision / Recall / F1-score per model (Walk-Forward, n=50).

Grouped bar chart.
Output: docs/figures/precision_recall_f1.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")

MODELS = ["BiLSTM", "LSTM", "RNN", "Logistic\nRegression"]
PRECISION = np.array([0.7408, 0.6619, 0.4707, 0.3866])
RECALL = np.array([0.4336, 0.3629, 0.3071, 0.2787])
F1 = np.array([0.4214, 0.3597, 0.3132, 0.2845])


def main():
    x = np.arange(len(MODELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    b1 = ax.bar(x - width, PRECISION, width, label="Precision", color="#1f77b4", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x, RECALL, width, label="Recall", color="#ff7f0e", alpha=0.85, edgecolor="black")
    b3 = ax.bar(x + width, F1, width, label="F1-score", color="#2ca02c", alpha=0.85, edgecolor="black")

    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1-score theo mô hình (lớp UP, Walk-Forward n=50)", fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "precision_recall_f1.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
