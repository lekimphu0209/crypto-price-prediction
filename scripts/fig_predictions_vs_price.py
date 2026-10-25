"""
Overlay model predictions (UP/DOWN) on actual BTC close price for the 5 walk-forward folds.

Each subplot corresponds to one model. For speed, one seed (seed 0) is used per fold.
Green triangles = predicted UP, red triangles = predicted DOWN.

Output: docs/figures/predictions_vs_btc_price.png
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster
from src.utils.determinism import set_full_determinism

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")

N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50
EPOCHS = 40


def build_forecaster(name):
    if name == "Logistic":
        return LinearForecaster(seed=0, C=1.0, class_weight="balanced", max_iter=1000)
    return KerasForecaster(
        name=name,
        epochs=EPOCHS,
        batch_size=16,
        units=32,
        dropout=0.1,
        rnn_dropout=0.0,
        recurrent_dropout=0.0,
        l2_reg=0.0,
        learning_rate=0.001,
        early_stopping_patience=0,
        validation_split=0.0,
        mode="classification",
    )


def main():
    print("=" * 80)
    print("GENERATING PREDICTIONS VS BTC PRICE CHART")
    print("=" * 80)

    extractor = TechnicalExtractor(feature_set=FEATURE_SET)
    target_builder = TrendTargetBuilder()
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)

    pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    print(f"Raw data shape: {df.shape}")

    feat_df = pipeline.extractor.extract(df)
    feature_names = pipeline.extractor.feature_names()
    print(f"Features shape: {feat_df.shape}")

    # Ensure datetime index for plotting
    if "timestamp" in feat_df.columns:
        plot_index = pd.to_datetime(feat_df["timestamp"])
    elif isinstance(feat_df.index, pd.DatetimeIndex):
        plot_index = feat_df.index
    else:
        plot_index = pd.to_datetime(df["timestamp"].iloc[:len(feat_df)])
    feat_df = feat_df.set_index(plot_index) if not isinstance(feat_df.index, pd.DatetimeIndex) else feat_df

    values = feat_df[feature_names].values
    close = feat_df["close"].values

    X_raw, base_close, future_close = window.build(values, close)
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X_raw = X_raw[mask]
    y = y[mask]
    original_idx = np.where(mask)[0]
    print(f"Windows shape: {X_raw.shape}, Target shape: {y.shape}")

    models = ["BiLSTM", "LSTM", "RNN", "Logistic"]
    all_predictions = {m: {"up": [], "down": []} for m in models}

    total_models = len(models) * N_FOLDS
    run_count = 0
    start_total = time.time()

    for model_name in models:
        print(f"\n--- {model_name} ---")
        for fold in range(N_FOLDS):
            train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
            test_start = train_end
            test_end = test_start + TEST_SIZE

            if test_end > len(X_raw):
                print(f"  Fold {fold + 1}: not enough data, skipping")
                break

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

            set_full_determinism(0)
            tf.keras.backend.clear_session()

            forecaster = build_forecaster(model_name)
            start = time.time()
            forecaster.fit(X_train, y_train)
            proba = np.asarray(forecaster.predict(X_test))
            pred = (proba >= 0.5).astype(int)
            elapsed = time.time() - start

            test_idx = original_idx[test_start:test_end]
            up_idx = test_idx[pred == 1]
            down_idx = test_idx[pred == 0]
            all_predictions[model_name]["up"].extend(up_idx.tolist())
            all_predictions[model_name]["down"].extend(down_idx.tolist())

            run_count += 1
            print(f"  [{run_count}/{total_models}] Fold {fold + 1}: UP={len(up_idx)}, DOWN={len(down_idx)} ({elapsed:.1f}s)")

    print(f"\nTotal prediction time: {time.time() - start_total:.1f}s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    close_prices = feat_df["close"].values
    plot_dates = feat_df.index

    for ax, model_name in zip(axes, models):
        ax.plot(plot_dates, close_prices, color="#1f77b4", linewidth=1.2, label="BTC close price", zorder=1)

        up_idx = np.array(all_predictions[model_name]["up"])
        down_idx = np.array(all_predictions[model_name]["down"])

        if len(up_idx) > 0:
            ax.scatter(plot_dates[up_idx], close_prices[up_idx], color="green", marker="^",
                       s=35, alpha=0.9, edgecolor="black", linewidth=0.5, label="Predict UP", zorder=2)
        if len(down_idx) > 0:
            ax.scatter(plot_dates[down_idx], close_prices[down_idx], color="red", marker="v",
                       s=35, alpha=0.9, edgecolor="black", linewidth=0.5, label="Predict DOWN", zorder=2)

        ax.set_title(f"{model_name} Seed-0 Predictions vs BTC Price", fontsize=12, weight="bold")
        ax.set_ylabel("Price (USDT)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.text(0.5, 0.02, "Predictions are from one seed per fold (seed=0) over 5 walk-forward test windows. "
                         "Green ▲ = UP, red ▼ = DOWN.",
             ha="center", fontsize=10, color="#555555")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "predictions_vs_btc_price.png")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
