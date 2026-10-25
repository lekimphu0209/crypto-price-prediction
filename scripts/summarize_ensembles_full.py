"""
Full metrics summary for all ensemble/bagging result JSON files in models/.

Reports mean (across folds) of Accuracy, Balanced Accuracy, Precision, Recall, F1,
plus number of folds and Balanced Accuracy std/min/max.
"""
import json
import glob
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


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


def label_from_file(fname):
    parts = fname.replace(".json", "").split("_")
    model = None
    for p in parts:
        if p in {"bilstm", "lstm", "rnn", "logistic"}:
            model = {"bilstm": "BiLSTM", "lstm": "LSTM", "rnn": "RNN", "logistic": "Logistic"}[p]
    variant_parts = [p for p in parts if p in {"ensemble", "filtered", "weighted", "bagging"}]
    variant = " ".join(variant_parts).title() if variant_parts else "Ensemble"
    return model or "?", variant


def summarize(path):
    with open(path) as f:
        data = json.load(f)
    records = _get_records(data)
    if not records:
        return None
    out = {"file": os.path.basename(path), "n_folds": len(records)}
    for m in METRICS:
        vals = np.array([r[m] for r in records if m in r])
        out[m] = vals.mean() if len(vals) else float("nan")
    balacc = np.array([r["balanced_accuracy"] for r in records])
    out["balacc_std"] = balacc.std(ddof=0)
    out["balacc_min"] = balacc.min()
    out["balacc_max"] = balacc.max()
    return out


def main():
    files = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*ensemble*.json"))
        + glob.glob(os.path.join(MODELS_DIR, "*bagging*.json"))
    )
    print("| Model | Phương pháp | n folds | Accuracy | Balanced Acc | Precision | Recall | F1 | BalAcc Std | Min | Max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for fpath in files:
        s = summarize(fpath)
        if s is None:
            continue
        model, variant = label_from_file(s["file"])
        rows.append((model, variant, s))
        print(
            f"| {model} | {variant} | {s['n_folds']} | "
            f"{s['accuracy']:.4f} | {s['balanced_accuracy']:.4f} | "
            f"{s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | "
            f"{s['balacc_std']:.4f} | {s['balacc_min']:.4f} | {s['balacc_max']:.4f} |"
        )


if __name__ == "__main__":
    main()
