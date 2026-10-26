"""
Walk-forward (expanding-window) evaluation for any registered model.

Unlike train_one.py (single chronological 80/20 split), this evaluates a
model over N consecutive test folds. Each fold:
  - refits the RobustScaler on training rows ONLY (no scaler leakage),
  - tunes the decision threshold on a trailing validation slice of train,
  - refits a FRESH forecaster on the full training window,
  - evaluates on the next unseen fold.

This matches the 'repeated hold-out / expanding window' evaluation protocol
recommended for time series (slide L6: repeated hold-out; model selection
via a separate validation set).

Usage (from repo root):
    python backend/scripts/walk_forward_eval.py --model LightGBM
    python backend/scripts/walk_forward_eval.py -m LightGBM -s BTCUSDT \
        --folds 5 --feature-set E --exclude rsi_14,bb_middle
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.persistence.file_registry import FileModelRegistry
from src.models.registry import create_forecaster, registered_models
from src.training.composition import build_extractor, MODELS_DIR
from src.training.datasets.market_data_loader import MarketDataLoader
from src.training.datasets.window_dataset import WindowDataset
from src.training.targets.trend_target import TrendTargetBuilder


def _tune_threshold(forecaster, X_fit, y_fit, X_val, y_val):
    """Scan thresholds on the validation slice; return best threshold."""
    forecaster.fit(X_fit, y_fit)
    val_proba = np.asarray(forecaster.predict(X_val))
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.3, 0.7, 41):
        score = balanced_accuracy_score(y_val, (val_proba >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    cfg = load_config("default")
    parser = argparse.ArgumentParser(description="Expanding-window evaluation")
    parser.add_argument("--symbol", "-s", default="BTCUSDT")
    parser.add_argument("--model", "-m", required=True,
                        help=f"Registered: {registered_models()}")
    parser.add_argument("--folds", "-k", type=int, default=5)
    parser.add_argument("--initial-ratio", type=float, default=0.5,
                        help="Fraction of samples in the first training window")
    parser.add_argument("--seq-len", type=int, default=int(cfg.get("seq_len", 20)))
    parser.add_argument("--horizon", type=int, default=int(cfg.get("horizon", 7)))
    parser.add_argument("--feature-set", default=None,
                        help="Override extractor feature_set (A|B|C|E)")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated feature names to exclude")
    args = parser.parse_args()

    seq_len, horizon = args.seq_len, args.horizon

    # 1. Load data + extract features ONCE (labels are scaler-independent)
    print(f"Loading {args.symbol} ...")
    loader = MarketDataLoader(BinanceProvider(), interval="1d",
                              limit=int(cfg.get("train_days", 1000)))
    df = loader.load(args.symbol)

    extractor_cfg = dict(cfg.get("extractor", {}))
    params = dict(extractor_cfg.get("params", {}))
    if args.feature_set:
        params["feature_set"] = args.feature_set
    if args.exclude:
        params["exclude_features"] = [f.strip() for f in args.exclude.split(",") if f.strip()]
    extractor = build_extractor({"name": extractor_cfg.get("name", "technical"),
                                 "params": params})

    feat_df = extractor.extract(df)
    feature_names = extractor.feature_names()
    values = feat_df[feature_names].values
    close = feat_df["close"].values
    print(f"rows={len(df)}  features={len(feature_names)}  feature_set={params.get('feature_set', 'A')}")

    # Labels don't depend on scaling -> compute once on unscaled windows
    window = WindowDataset(seq_len=seq_len, target_horizon=horizon)
    _, base_close, future_close = window.build(values, close)
    target = TrendTargetBuilder()
    y_all = target.build(np.asarray(base_close), np.asarray(future_close))
    n_win = len(y_all)

    initial_train = int(n_win * args.initial_ratio)
    fold_size = max(10, (n_win - initial_train) // args.folds)
    print(f"windows={n_win}  initial_train={initial_train}  fold_size={fold_size}")

    folds_metrics, importances, thresholds = [], [], []

    for k in range(args.folds):
        split = initial_train + k * fold_size
        if split >= n_win:
            break
        test_end = min(split + fold_size, n_win)

        # Scaler fits only on feature rows visible to TRAINING windows.
        # Train windows are [0 .. split-1]; the last ends at row split+seq_len-2.
        cut = split + seq_len - 1
        scaler = RobustScaler().fit(values[:cut])
        X, base_close2, future_close2 = window.build(scaler.transform(values), close)
        # sanity: same window count/labels
        assert len(X) == n_win

        X_train, X_test = X[:split], X[split:test_end]
        y_train, y_test = y_all[:split], y_all[split:test_end]

        # Trailing validation slice for threshold tuning (like the pipeline)
        forecaster = create_forecaster(args.model)
        val_size = max(10, int(len(X_train) * 0.15))
        if len(X_train) - val_size >= 20:
            best_t = _tune_threshold(
                forecaster,
                X_train[:-val_size], y_train[:-val_size],
                X_train[-val_size:], y_train[-val_size:],
            )
        else:
            best_t = 0.5
        thresholds.append(best_t)

        # Fresh forecaster on full train -> evaluate unseen fold
        forecaster = create_forecaster(args.model)
        forecaster.fit(X_train, y_train)
        proba = np.asarray(forecaster.predict(X_test))
        m = _metrics(y_test, (proba >= best_t).astype(int))
        m["fold"] = k + 1
        m["threshold"] = best_t
        m["train_samples"] = len(X_train)
        m["test_samples"] = len(X_test)
        folds_metrics.append(m)
        print(f"  fold {k+1}: train={len(X_train)} test={len(X_test)} "
              f"thr={best_t:.2f} bal_acc={m['balanced_accuracy']:.4f} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

        # Aggregate feature importances when the model exposes them
        if hasattr(forecaster, "feature_importances"):
            flat = list(forecaster.feature_importances().values())
            if len(flat) == seq_len * len(feature_names):
                importances.append(
                    np.asarray(flat).reshape(seq_len, len(feature_names)).sum(axis=0)
                )

    if not folds_metrics:
        print("No folds evaluated.")
        sys.exit(1)

    # Summary
    print("\n=== Walk-forward summary ===")
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary = {"model": args.model, "symbol": args.symbol, "folds": len(folds_metrics),
               "feature_set": params.get("feature_set", "A"),
               "excluded": params.get("exclude_features", []),
               "seq_len": seq_len, "horizon": horizon,
               "mean_threshold": float(np.mean(thresholds))}
    for key in keys:
        vals = [f[key] for f in folds_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        print(f"  {key:<20} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Per-feature importance ranking (only for tree models)
    if importances:
        imp = np.mean(importances, axis=0)
        ranking = sorted(zip(feature_names, imp), key=lambda t: -t[1])
        summary["feature_importance"] = {n: float(v) for n, v in ranking}
        print("\n=== Feature importance (mean over folds, summed over window) ===")
        for name, v in ranking:
            print(f"  {name:<24} {v:,.0f}")

    out = os.path.join(
        "backend", "data",
        f"{args.symbol}_{args.model}_walkforward_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "folds": folds_metrics}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
