"""
Seed ensemble section content and figure for the report.

Output: docs/figures/seed_ensemble.png
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    with open(os.path.join(MODELS_DIR, "btc_walkforward_bilstm_ensemble_results.json")) as f:
        ensemble = json.load(f)

    folds = [m["fold"] for m in ensemble]
    ens_balacc = np.array([m["balanced_accuracy"] for m in ensemble])
    seed_mean = np.array([m["mean_seed_balacc"] for m in ensemble])
    seed_std = np.array([m["std_seed_balacc"] for m in ensemble])

    print("# 4.3.1. Seed Ensemble\n")
    print("Mỗi fold huấn luyện **10 BiLSTM** với seed khác nhau (0–9), sau đó lấy trung bình xác suất dự báo rồi ngưỡng 0.5.\n")

    print("## Bảng kết quả theo fold")
    print("| Fold | Ensemble BalAcc | Mean Seed BalAcc | Std Seed | Min Seed | Max Seed |")
    print("|---|---|---|---|---|---|")
    for m in ensemble:
        print(
            f"| {m['fold']} | {m['balanced_accuracy']:.4f} | "
            f"{m['mean_seed_balacc']:.4f} | {m['std_seed_balacc']:.4f} | "
            f"{m['min_seed_balacc']:.4f} | {m['max_seed_balacc']:.4f} |"
        )

    print("\n## Thống kê tổng hợp")
    print(f"- Ensemble mean BalAcc: {ens_balacc.mean():.4f} ± {ens_balacc.std():.4f}")
    print(f"- Mean of single-seed averages: {seed_mean.mean():.4f}")
    print(f"- Average std across seeds: {seed_std.mean():.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width / 2, ens_balacc, width, label="Ensemble", color="#2ca02c", alpha=0.8, edgecolor="black")
    ax.bar(x + width / 2, seed_mean, width, yerr=seed_std, capsize=4, label="Avg single seed", color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Baseline 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("BiLSTM Seed Ensemble vs Average Single Seed (10 seeds/fold)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0.35, 0.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "seed_ensemble.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
