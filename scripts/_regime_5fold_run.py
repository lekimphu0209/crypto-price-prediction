import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.infrastructure.forecasters.keras_forecaster import SUPPORTED as KERAS_MODELS
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster
from src.market_regime.regime_feature_extractor import RegimeFeatureExtractor
from src.training.market_data_loader import MarketDataLoader
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def build_model(model_name: str, epochs: int):
    if model_name in KERAS_MODELS:
        return KerasForecaster(model_name, epochs=epochs)
    if model_name == "Logistic Regression":
        return LinearForecaster()
    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="5-fold training with regime features.")
    parser.add_argument("--model", default="BiLSTM", help="Model name.")
    parser.add_argument("--epochs", type=int, default=15, help="Keras training epochs.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol.")
    args = parser.parse_args()

    print(f"=== 5-FOLD {args.model.upper()} + REGIME | {args.symbol} ===")

    symbol = args.symbol
    epochs = args.epochs
    test_size = 100
    fold_starts = [400, 500, 600, 700, 800]

    df = MarketDataLoader(BinanceProvider(), "1d", 1000).load(symbol)
    extractor = RegimeFeatureExtractor(base_feature_set="E")
    feat = extractor.extract(df)
    values = feat[extractor.feature_names()].values
    close = feat["close"].values
    window = WindowDataset(seq_len=20, target_horizon=7)
    builder = TrendTargetBuilder()
    rows = []

    for i, test_start in enumerate(fold_starts, 1):
        test_end = test_start + test_size
        train_v, test_v = values[:test_start], values[test_start:test_end]
        train_c, test_c = close[:test_start], close[test_start:test_end]
        scaler = RobustScaler().fit(train_v)
        X_tr, b_tr, f_tr = window.build(scaler.transform(train_v), train_c)
        X_te, b_te, f_te = window.build(scaler.transform(test_v), test_c)
        y_tr = builder.build(b_tr, f_tr)
        y_te = builder.build(b_te, f_te)
        if len(X_te) == 0:
            continue
        t0 = time.time()
        model = build_model(args.model, epochs)
        model.fit(X_tr, y_tr)
        y_pred = (model.predict(X_te) >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y_te, y_pred))
        bal = float(balanced_accuracy_score(y_te, y_pred))
        prec = float(precision_score(y_te, y_pred, zero_division=0))
        rec = float(recall_score(y_te, y_pred, zero_division=0))
        f1v = float(f1_score(y_te, y_pred, zero_division=0))
        elapsed = time.time() - t0
        rows.append(
            {
                "fold": i,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "precision": prec,
                "recall": rec,
                "f1": f1v,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Fold {i}: train={len(X_tr)} test={len(X_te)} "
            f"acc={acc:.4f} bal={bal:.4f} f1={f1v:.4f} time={elapsed:.1f}s"
        )

    out = pd.DataFrame(rows)
    model_slug = slug(args.model)
    out.to_csv(f"data/_{model_slug}_regime_5fold_latest.csv", index=False)
    summary = {
        "symbol": symbol,
        "model": args.model,
        "epochs": epochs,
        "folds": len(out),
        "mean": out[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
        ].mean().to_dict(),
        "std": out[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
        ].std().to_dict(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(f"data/_{model_slug}_regime_5fold_summary_latest.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("MEAN:")
    print(
        out[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
        ].mean().to_string()
    )
    print("STD:")
    print(
        out[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
        ].std().to_string()
    )


if __name__ == "__main__":
    main()
