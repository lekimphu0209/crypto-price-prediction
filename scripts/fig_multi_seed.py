"""
Generate a multi-seed training illustration for the report.

Output: docs/figures/multi_seed_balacc.png
Shows the distribution of Balanced Accuracy across 10 seeds x 5 folds for each model.
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
    vals = [f["balanced_accuracy"] for r in data if r.get("model") == name for f in r["fold_metrics"]]
    if len(vals) == 0:
        vals = [f["balanced_accuracy"] for r in data for f in r["fold_metrics"]]
    return np.array(vals)


def main():
    lstm = load_model("LSTM", "btc_walkforward_results_detailed.json")
    rnn = load_model("RNN", "btc_walkforward_results_detailed.json")
    bilstm = load_model("BiLSTM", "btc_walkforward_bilstm_results.json")
    logistic = load_model("Logistic", "btc_walkforward_logistic_results.json")

    models = {
        "BiLSTM": bilstm,
        "LSTM": lstm,
        "RNN": rnn,
        "Logistic": logistic,
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(models))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for pos, (name, vals), color in zip(positions, models.items(), colors):
        # jittered scatter
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.5, s=30, color=color, edgecolors="none")
        # boxplot
        bp = ax.boxplot(
            vals,
            positions=[pos],
            widths=0.4,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={"color": "black"},
            boxprops={"facecolor": color, "alpha": 0.3},
            meanprops={"color": "red", "linestyle": "--"},
        )
        # annotate mean
        ax.text(pos, vals.mean() + 0.015, f"{vals.mean():.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (0.5)")
    ax.set_xticks(positions)
    ax.set_xticklabels(models.keys())
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Multi-seed Training Stability (10 seeds × 5 folds = 50 samples)")
    ax.set_ylim(0.25, 0.85)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    out_path = os.path.join(OUT_DIR, "multi_seed_balacc.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
