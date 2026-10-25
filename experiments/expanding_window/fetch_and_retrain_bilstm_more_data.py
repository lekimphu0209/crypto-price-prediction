"""
Fetch more BTCUSDT daily data from Binance and retrain BiLSTM with expanding window.

- Loads existing cached data (data/BTCUSDT_raw_20260705_065322.csv).
- Fetches 1000 older daily candles from Binance.
- Merges, sorts, drops duplicates -> data/btc_1d_extended.csv.
- Retrains BiLSTM with expanding window on the extended dataset.
"""
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy import stats

from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.utils.determinism import set_full_determinism
from src.infrastructure.data_providers.binance_provider import BinanceProvider

EXTENDED_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "btc_1d_extended.csv")
BASE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "BTCUSDT_raw_20260705_065322.csv")
FEATURE_SET = "B"
SEQ_LEN = 20
HORIZON = 7
INITIAL_TRAIN_SIZE = 800
TEST_SIZE = 100
N_FOLDS = 3


def load_base_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_extended_df():
    if os.path.exists(EXTENDED_CSV):
        print(f"Using existing extended data: {EXTENDED_CSV}")
        return load_base_csv(EXTENDED_CSV)

    print("Loading base data...")
    base_df = load_base_csv(BASE_CSV)
    print(f"  Base rows: {len(base_df)}, from {base_df['timestamp'].min()} to {base_df['timestamp'].max()}")

    existing_min = base_df["timestamp"].min()
    if existing_min.tzinfo is not None:
        existing_min = existing_min.tz_convert("UTC").tz_localize(None)
    start_date = existing_min - pd.Timedelta(days=1000)
    print(f"\nFetching {1000} older daily candles from Binance since {start_date}...")

    provider = BinanceProvider()
    ohlcv_list = provider.fetch_ohlcv("BTCUSDT", "1d", limit=1000, start_date=start_date)
    older_df = pd.DataFrame([{
        "timestamp": o.timestamp,
        "open": o.open,
        "high": o.high,
        "low": o.low,
        "close": o.close,
        "volume": o.volume,
    } for o in ohlcv_list])
    older_df["timestamp"] = pd.to_datetime(older_df["timestamp"])
    older_df = older_df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Fetched rows: {len(older_df)}, from {older_df['timestamp'].min()} to {older_df['timestamp'].max()}")

    combined = pd.concat([older_df, base_df], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    combined.to_csv(EXTENDED_CSV, index=False)
    print(f"\nSaved extended dataset: {EXTENDED_CSV} ({len(combined)} rows)")
    return combined


def main():
    print("=" * 80)
    print("FETCH MORE DATA + RETRAIN BiLSTM (Expanding Window)")
    print("=" * 80)

    df = build_extended_df()
    print(f"Extended data: {len(df)} rows, from {df['timestamp'].min()} to {df['timestamp'].max()}")

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    feat_df = extractor.extract(df)
    feature_names = extractor.feature_names()
    print(f"Features shape: {feat_df.shape}")

    values = feat_df[feature_names].values
    close = feat_df["close"].values

    X_raw, base_close, future_close = window.build(values, close)
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X_raw = X_raw[mask]
    y = y[mask]
    print(f"Windows shape: {X_raw.shape}, Target shape: {y.shape}")

    results = []

    for fold in range(N_FOLDS):
        train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if test_end >= len(X_raw):
            print(f"\nFold {fold + 1}: not enough data, stop")
            break

        print(f"\nFold {fold + 1}: train[0:{train_end}], test[{test_start}:{test_end}]")

        X_train_raw = X_raw[:train_end]
        y_train = y[:train_end]
        X_test_raw = X_raw[test_start:test_end]
        y_test = y[test_start:test_end]

        flat_train = X_train_raw.reshape(len(X_train_raw), -1)
        flat_test = X_test_raw.reshape(len(X_test_raw), -1)
        scaler = RobustScaler()
        scaler.fit(flat_train)
        X_train = scaler.transform(flat_train).reshape(X_train_raw.shape)
        X_test = scaler.transform(flat_test).reshape(X_test_raw.shape)

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Test: UP={(y_test == 1).sum()}, DOWN={(y_test == 0).sum()}")

        set_full_determinism(42 + fold)
        forecaster = KerasForecaster(
            name="BiLSTM",
            epochs=15,
            batch_size=16,
            units=32,
            dropout=0.1,
            early_stopping_patience=5,
            reduce_lr_patience=0,
        )

        start = time.time()
        forecaster.fit(X_train, y_train)
        proba = np.asarray(forecaster.predict(X_test))
        y_pred = (proba >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        metrics = {
            "fold": fold + 1,
            "train_end": int(train_end),
            "test_start": int(test_start),
            "test_end": int(test_end),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        results.append(metrics)
        elapsed = time.time() - start
        print(f"  -> Acc={metrics['accuracy']:.4f} BalAcc={metrics['balanced_accuracy']:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} ({elapsed:.1f}s)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "results_bilstm_more_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if results:
        balacc_vals = np.array([m["balanced_accuracy"] for m in results])
        print("\n" + "=" * 80)
        print("T-TEST: BiLSTM (more data) vs Baseline (0.5)")
        print("=" * 80)
        print(f"  n (folds): {len(balacc_vals)}")
        print(f"  Mean: {balacc_vals.mean():.4f}")
        print(f"  Std: {balacc_vals.std():.4f}")
        t_stat, p_value = stats.ttest_1samp(balacc_vals, 0.5)
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (alpha=0.05): {'YES' if p_value < 0.05 else 'NO'}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"BiLSTM (more data): Acc={np.mean([m['accuracy'] for m in results]):.4f} "
              f"BalAcc={balacc_vals.mean():.4f} ± {balacc_vals.std():.4f}")


if __name__ == "__main__":
    main()
