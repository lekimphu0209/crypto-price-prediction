import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.market_regime.regime_feature_extractor import RegimeFeatureExtractor
from src.training.market_data_loader import MarketDataLoader
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset

print("=== 5-FOLD BiLSTM + REGIME ===")

symbol = "BTCUSDT"
epochs = 15
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
    model = KerasForecaster("BiLSTM", epochs=epochs)
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
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out.to_csv(f"data/_bilstm_regime_5fold_latest.csv", index=False)

summary = {
    "symbol": symbol,
    "epochs": epochs,
    "folds": len(out),
    "mean": out[
        ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
    ].mean().to_dict(),
    "std": out[
        ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "elapsed_sec"]
    ].std().to_dict(),
    "timestamp": ts,
}
with open("data/_bilstm_regime_5fold_summary_latest.json", "w") as f:
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