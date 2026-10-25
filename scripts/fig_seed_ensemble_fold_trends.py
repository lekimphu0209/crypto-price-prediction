"""
Fold-by-fold Balanced Accuracy trends for Seed Ensemble models.

Output: docs/figures/seed_ensemble_fold_trends.png
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def load_balacc(path, key=None):
    with open(path) as f:
        data = json.load(f)
    if key:
        data = data[key]
    return np.array([m["balanced_accuracy"] for m in data])


def main():
    bilstm = load_balacc(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_ensemble_results.json"))
    lstm = load_balacc(os.path.join(EXP_DIR, "seed_ensemble_lstm", "results_lstm_seed_ensemble.json"), "full_ensemble")
    rnn = load_balacc(os.path.join(EXP_DIR, "seed_ensemble_rnn", "results_rnn_seed_ensemble.json"), "full_ensemble")
    logistic = load_balacc(os.path.join(EXP_DIR, "seed_ensemble_logistic", "results_logistic_seed_ensemble.json"), "full_ensemble")

    models = {
        "BiLSTM": (bilstm, "#2E86AB", "o"),
        "LSTM": (lstm, "#A23B72", "s"),
        "RNN": (rnn, "#F18F01", "^"),
        "Logistic": (logistic, "#C73E1D", "d"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    folds = np.arange(1, 6)

    for name, (balacc, color, marker) in models.items():
        ax.plot(folds, balacc, marker=marker, markersize=8, linewidth=2.0,
                label=name, color=color, markeredgecolor="black", markeredgewidth=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Baseline (0.5)")

    ax.set_xticks(folds)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Fold-by-Fold Balanced Accuracy: Seed Ensemble (10 seeds) per Model", fontsize=14, weight="bold")
    ax.set_ylim(0.25, 0.70)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, loc="best")

    ax.text(0.5, -0.12, "Each point is the 10-seed ensemble BalAcc for that fold. "
                         "Lines show how model performance varies across walk-forward folds.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "seed_ensemble_fold_trends.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
