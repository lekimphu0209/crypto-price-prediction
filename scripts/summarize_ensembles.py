"""
Summarize all ensemble/bagging result JSON files in models/.
"""
import json
import glob
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _get_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("full_ensemble", "results", "folds"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def summarize(path):
    with open(path) as f:
        data = json.load(f)
    records = _get_records(data)
    if not records:
        return None
    balacc = np.array([m["balanced_accuracy"] for m in records])
    n = len(balacc)
    out = {
        "file": os.path.basename(path),
        "n_folds": n,
        "balacc_mean": balacc.mean(),
        "balacc_std": balacc.std(ddof=0),
        "balacc_min": balacc.min(),
        "balacc_max": balacc.max(),
    }
    if "mean_seed_balacc" in records[0]:
        seed_means = np.array([m["mean_seed_balacc"] for m in records])
        seed_stds = np.array([m["std_seed_balacc"] for m in records])
        out["seed_mean_avg"] = seed_means.mean()
        out["seed_std_avg"] = seed_stds.mean()
    return out


def label_from_file(fname):
    # e.g. btc_walkforward_bilstm_ensemble_results.json -> BiLSTM ensemble
    parts = fname.replace(".json", "").split("_")
    model = None
    for p in parts:
        if p in {"bilstm", "lstm", "rnn"}:
            model = p.upper() if p != "bilstm" else "BiLSTM"
    variant = " ".join([p for p in parts if p in {"ensemble", "filtered", "weighted", "bagging"}])
    return model, variant


def main():
    files = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*ensemble*.json"))
        + glob.glob(os.path.join(MODELS_DIR, "*bagging*.json"))
    )
    print("| Model | Phương pháp | n | BalAcc Mean | BalAcc Std | Min | Max | Avg Seed Mean | Avg Seed Std |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in files:
        s = summarize(f)
        if s is None:
            continue
        model, variant = label_from_file(s["file"])
        seed_mean = f"{s.get('seed_mean_avg', ''):.4f}" if "seed_mean_avg" in s else ""
        seed_std = f"{s.get('seed_std_avg', ''):.4f}" if "seed_std_avg" in s else ""
        print(
            f"| {model} | {variant} | {s['n_folds']} | "
            f"{s['balacc_mean']:.4f} | {s['balacc_std']:.4f} | "
            f"{s['balacc_min']:.4f} | {s['balacc_max']:.4f} | "
            f"{seed_mean} | {seed_std} |"
        )


if __name__ == "__main__":
    main()
