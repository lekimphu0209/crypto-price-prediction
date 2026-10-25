"""
TrainingPipeline - orchestrates the full training flow.

It depends only on ABSTRACTIONS (data loader, feature extractor, target
builder, model registry) and a forecaster instance. It knows the *sequence*
of steps, not their concrete implementations.

Flow:
    load OHLCV -> extract features -> split -> scale (fit on train only)
    -> window -> build up/down target -> fit classifier -> evaluate
    (accuracy + confusion matrix) -> save via registry
"""
from typing import Dict
import os
from datetime import datetime

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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.domain.interfaces.target_builder import ITargetBuilder
from src.domain.interfaces.forecaster import IForecaster
from src.domain.interfaces.model_registry import IModelRegistry
from src.training.datasets.market_data_loader import MarketDataLoader
from src.training.datasets.window_dataset import WindowDataset


class TrainingPipeline:
    """Coordinates training of a single forecaster for a symbol."""

    def __init__(
        self,
        loader: MarketDataLoader,
        extractor: IFeatureExtractor,
        target_builder: ITargetBuilder,
        registry: IModelRegistry,
        seq_len: int = 20,
        train_split: float = 0.8,
        data_dir: str = "data",
        target_horizon: int = 7,
        threshold_metric: str = "balanced_accuracy",
        mode: str = "classification",
    ):
        self.loader = loader
        self.extractor = extractor
        self.target_builder = target_builder
        self.registry = registry
        self.window = WindowDataset(seq_len=seq_len, target_horizon=target_horizon)
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.train_split = train_split
        self.data_dir = data_dir
        self.threshold_metric = threshold_metric
        self.mode = mode
        os.makedirs(data_dir, exist_ok=True)

    def train(self, symbol: str, forecaster: IForecaster, model_name: str = "model", save_csv: bool = False) -> Dict:
        # 1. Load raw data
        df = self.loader.load(symbol)
        if len(df) < self.seq_len + 50:
            return {"ok": False, "error": f"Not enough data: {len(df)}"}

        # Create timestamp for CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw data CSV (optional, only once per symbol)
        if save_csv:
            raw_csv_path = os.path.join(self.data_dir, f"{symbol}_raw_{timestamp}.csv")
            df.to_csv(raw_csv_path, index=True)
            print(f"  Saved raw data to: {raw_csv_path}")

        # 2. Feature engineering
        feat_df = self.extractor.extract(df)
        feature_names = self.extractor.feature_names()

        # Save features CSV (optional, only once per symbol)
        if save_csv:
            features_csv_path = os.path.join(self.data_dir, f"{symbol}_features_{timestamp}.csv")
            feat_df.to_csv(features_csv_path, index=True)
            print(f"  Saved features to: {features_csv_path}")

        # 3. Scale features (fit scaler on the training rows only -> no leakage)
        values = feat_df[feature_names].values
        close = feat_df["close"].values
        row_split = int(len(values) * self.train_split)
        scaler = RobustScaler()
        scaler.fit(values[:row_split])
        feature_matrix = scaler.transform(values)

        # 4. Build windows + up/down target
        X, base_close, future_close = self.window.build(feature_matrix, close)
        if len(X) < 20:
            return {"ok": False, "error": "Not enough sequences"}
        y = self.target_builder.build(base_close, future_close)

        # 5. Train/test split (time-ordered)
        split = int(len(X) * self.train_split)
        X_train, X_test = X[:split], X[split:]
        y_train, y_true = y[:split], y[split:]

        if self.mode == "regression":
            # 6. Regression: train on full training set, no threshold tuning
            forecaster.fit(X_train, y_train)
            y_pred_reg = np.asarray(forecaster.predict(X_test))
            mse = float(mean_squared_error(y_true, y_pred_reg))
            mae = float(mean_absolute_error(y_true, y_pred_reg))
            r2 = float(r2_score(y_true, y_pred_reg))
            y_pred = self.target_builder.reconstruct(y_pred_reg)
            y_true_dir = self.target_builder.reconstruct(y_true)
            tn, fp, fn, tp = confusion_matrix(y_true_dir, y_pred, labels=[0, 1]).ravel()
            metrics = {
                "accuracy": float(accuracy_score(y_true_dir, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_dir, y_pred)),
                "precision": float(precision_score(y_true_dir, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_dir, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_dir, y_pred, zero_division=0)),
                "threshold": 0.0,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        else:
            # 6. Classification: tune decision threshold on a validation slice (end of train)
            val_size = max(10, int(len(X_train) * 0.15))
            if len(X_train) - val_size >= 20:
                X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
                y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
                # Pass validation data to forecaster for early stopping
                forecaster.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
                val_proba = np.asarray(forecaster.predict(X_val))
                best_t, best_score = 0.5, -1.0
                score_fn = accuracy_score if self.threshold_metric == "accuracy" else balanced_accuracy_score
                for t in np.linspace(0.3, 0.7, 41):
                    pred = (val_proba >= t).astype(int)
                    score = score_fn(y_val, pred)
                    if score > best_score:
                        best_score, best_t = score, float(t)
            else:
                best_t = 0.5
                forecaster.fit(X_train, y_train)

            # 7. Refit on the full training set, evaluate on test with tuned threshold
            forecaster.fit(X_train, y_train)
            proba = np.asarray(forecaster.predict(X_test))
            y_pred = (proba >= best_t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "threshold": float(best_t),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

        # Save training results CSV
        results_csv_path = os.path.join(self.data_dir, f"{symbol}_{model_name}_results_{timestamp}.csv")
        results_df = pd.DataFrame([{
            "symbol": symbol,
            "model_name": model_name,
            "timestamp": timestamp,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"],
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "seq_len": self.seq_len,
            "horizon": self.target_horizon,
            "mode": self.mode,
        }])
        results_df.to_csv(results_csv_path, index=False)
        print(f"  Saved training results to: {results_csv_path}")

        # 8. Persist
        extractor_config = (
            self.extractor.to_config() if hasattr(self.extractor, "to_config") else None
        )
        self.registry.save(
            symbol=symbol,
            forecaster=forecaster,
            scaler=scaler,
            feature_names=feature_names,
            seq_len=self.seq_len,
            metrics=metrics,
            horizon=self.target_horizon,
            threshold=metrics["threshold"],
            mode=self.mode,
            extractor_config=extractor_config,
        )
        return {"ok": True, "metrics": metrics}
