"""
Generate figures for the evaluation section of the report.

Output: docs/figures/evaluation_significance.png
- Subplot 1: mean Balanced Accuracy ± std per model with significance vs baseline
- Subplot 2: paired t-test p-value matrix with significance stars
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def load_model(name, file_name):
    with open(os.path.join(MODELS_DIR, file_name)) as f:
        data = json.load(f)
    vals = [f["balanced_accuracy"] for r in data if r.get("model") == name for f in r["fold_metrics"]]
    if len(vals) == 0:
        vals = [f["balanced_accuracy"] for r in data for f in r["fold_metrics"]]
    return np.array(vals)


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


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

    names = list(models.keys())
    means = [models[n].mean() for n in names]
    stds = [models[n].std() for n in names]

    # One-sample t-test vs 0.5
    baseline_p = {n: stats.ttest_1samp(models[n], 0.5)[1] for n in names}

    # Paired p-values matrix
    p_matrix = np.zeros((len(names), len(names)))
    t_matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                p_matrix[i, j] = np.nan
                t_matrix[i, j] = np.nan
            elif i < j:
                t, p = stats.ttest_rel(models[a], models[b])
                p_matrix[i, j] = p
                p_matrix[j, i] = p
                t_matrix[i, j] = t
                t_matrix[j, i] = -t

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean BalAcc with error bars
    ax = axes[0]
    x = np.arange(len(names))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Baseline 0.5")
    for i, (m, s, n) in enumerate(zip(means, stds, names)):
        stars = sig_stars(baseline_p[n])
        ax.text(i, m + s + 0.015, f"{m:.3f}\n{stars}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("One-sample t-test vs Baseline (0.5)")
    ax.set_ylim(0.35, 0.7)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Right: paired t-test significance matrix
    ax = axes[1]
    # Use -log10(p) for color, masking diagonal
    logp = -np.log10(np.clip(p_matrix, 1e-10, 1))
    im = ax.imshow(logp, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_title("Paired t-test Significance (-log10 p)")

    # Annotate stars
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                text = "—"
            else:
                text = sig_stars(p_matrix[i, j])
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=12, weight="bold")

    fig.colorbar(im, ax=ax, label="-log10(p)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "evaluation_significance.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
