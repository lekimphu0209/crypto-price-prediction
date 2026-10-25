"""
Conceptual illustration of 5-fold Seed Ensemble.

Shows the expanding-window walk-forward splits, 10 seeds trained per fold,
their probability outputs averaged, and thresholding to a final prediction.

Output: docs/figures/seed_ensemble_conceptual.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(50, 9.5, "5-Fold Seed Ensemble Walk-Forward", fontsize=16, ha="center", weight="bold")
    ax.text(50, 8.9, "Each fold trains 10 models (different seeds); probabilities are averaged and thresholded at 0.5",
            fontsize=10, ha="center", color="#555555")

    # Timeline parameters
    y_fold = 7.2
    fold_height = 0.9
    seed_box_size = 0.22
    n_folds = 5
    initial_train = 18
    test_width = 10
    gap = 2
    seed_start_y = 5.0

    # Draw timeline base
    ax.plot([2, 95], [y_fold - 0.8, y_fold - 0.8], color="black", linewidth=1.5)
    ax.text(95, y_fold - 1.0, "Time", fontsize=10, ha="right", style="italic")

    colors = {"train": "#4C78A8", "test": "#F58518", "seed": "#54A24B", "avg": "#E45756"}

    for fold in range(n_folds):
        train_end = initial_train + fold * test_width
        x_train_start = 5
        x_test_start = 5 + train_end

        # Training window
        rect_train = patches.FancyBboxPatch(
            (x_train_start, y_fold - fold_height / 2),
            train_end, fold_height,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=colors["train"], edgecolor="black", alpha=0.7
        )
        ax.add_patch(rect_train)

        # Test window
        rect_test = patches.FancyBboxPatch(
            (x_test_start, y_fold - fold_height / 2),
            test_width, fold_height,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=colors["test"], edgecolor="black", alpha=0.8
        )
        ax.add_patch(rect_test)

        # Fold label
        ax.text(x_test_start + test_width / 2, y_fold + fold_height / 2 + 0.3,
                f"Fold {fold + 1}", fontsize=9, ha="center", weight="bold")

        # Seed boxes for this fold
        seed_x = x_test_start + 1.5
        seed_y = seed_start_y - fold * 0.9
        for seed in range(10):
            sx = seed_x + (seed % 5) * 1.1
            sy = seed_y - (seed // 5) * 0.35
            rect = patches.Rectangle((sx, sy), seed_box_size, seed_box_size,
                                     facecolor=colors["seed"], edgecolor="black", alpha=0.6)
            ax.add_patch(rect)

        # Arrow from test window to seeds
        ax.annotate("", xy=(seed_x + 2.2, seed_y + 0.25),
                    xytext=(x_test_start + test_width / 2, y_fold - fold_height / 2 - 0.1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

        # Average box
        avg_x = seed_x + 7.5
        avg_y = seed_y - 0.05
        rect_avg = patches.FancyBboxPatch(
            (avg_x, avg_y), 2.2, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=colors["avg"], edgecolor="black", alpha=0.8
        )
        ax.add_patch(rect_avg)
        ax.text(avg_x + 1.1, avg_y + 0.3, "Avg", fontsize=8, ha="center", va="center", color="white", weight="bold")

        # Arrows from seeds to avg
        for seed in range(10):
            sx = seed_x + (seed % 5) * 1.1 + seed_box_size / 2
            sy = seed_y - (seed // 5) * 0.35 + seed_box_size / 2
            ax.plot([sx, avg_x], [sy, avg_y + 0.3], color="#888888", linewidth=0.4, alpha=0.5)

        # Threshold and final prediction
        pred_x = avg_x + 4.5
        ax.annotate("", xy=(pred_x, avg_y + 0.3),
                    xytext=(avg_x + 2.2, avg_y + 0.3),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

        pred_label = "UP" if fold % 2 == 0 else "DOWN"
        pred_color = "#2CA02C" if pred_label == "UP" else "#D62728"
        ax.text(pred_x + 0.2, avg_y + 0.3, f"→ {pred_label}", fontsize=9,
                va="center", color=pred_color, weight="bold")

    # Legend
    legend_elements = [
        patches.Patch(facecolor=colors["train"], edgecolor="black", label="Training window"),
        patches.Patch(facecolor=colors["test"], edgecolor="black", label="Test window"),
        patches.Patch(facecolor=colors["seed"], edgecolor="black", label="Seed model"),
        patches.Patch(facecolor=colors["avg"], edgecolor="black", label="Average probability"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.05), frameon=True)

    # Bottom note
    ax.text(50, 0.3, "Walk-forward: training set expands each fold; the model is retrained from scratch. "
                     "Seed ensemble averages predicted probabilities across 10 independent initializations.",
            fontsize=9, ha="center", color="#555555")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "seed_ensemble_conceptual.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
