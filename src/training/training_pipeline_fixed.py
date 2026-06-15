"""
TrainingPipelineFixed - orchestrates training with NO data leakage.

FIX: Split train/test BEFORE scaling to avoid data leakage.
Original: scale entire data -> split (leakage)
Fixed: split -> fit scaler on train only -> transform train/test separately

Flow:
    load OHLCV -> extract features -> SPLIT -> scale (train only) -> window -> build target
    -> fit forecaster -> evaluate -> save via registry
"""
from typing import Dict
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.domain.interfaces.target_builder import ITargetBuilder
from src.domain.interfaces.forecaster import IForecaster
from src.domain.interfaces.model_registry import IModelRegistry
from src.training.market_data_loader import MarketDataLoader
from src.training.window_dataset import WindowDataset


class TrainingPipelineFixed:
    """Coordinates training of a single forecaster for a symbol - NO DATA LEAKAGE."""

    def __init__(
        self,
        loader: MarketDataLoader,
        extractor: IFeatureExtractor,
        target_builder: ITargetBuilder,
        registry: IModelRegistry,
        seq_len: int = 20,
        train_split: float = 0.8,
        data_dir: str = "data",
    ):
        self.loader = loader
        self.extractor = extractor
        self.target_builder = target_builder
        self.registry = registry
        self.window = WindowDataset(seq_len=seq_len)
        self.seq_len = seq_len
        self.train_split = train_split
        self.data_dir = data_dir
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
            raw_csv_path = os.path.join(self.data_dir, f"{symbol}_raw_fixed_{timestamp}.csv")
            df.to_csv(raw_csv_path, index=True)
            print(f"  Saved raw data to: {raw_csv_path}")

        # 2. Feature engineering
        feat_df = self.extractor.extract(df)
        feature_names = self.extractor.feature_names()

        # Save features CSV (optional, only once per symbol)
        if save_csv:
            features_csv_path = os.path.join(self.data_dir, f"{symbol}_features_fixed_{timestamp}.csv")
            feat_df.to_csv(features_csv_path, index=True)
            print(f"  Saved features to: {features_csv_path}")

        # 3. SPLIT TRAIN/TEST FIRST (FIX: no leakage)
        split = int(len(feat_df) * self.train_split)
        train_df = feat_df[:split].copy()
        test_df = feat_df[split:].copy()

        # 4. Scale features - FIT ONLY ON TRAIN (FIX: no leakage)
        scaler = RobustScaler()
        scaler.fit(train_df[feature_names].values)
        
        train_features = scaler.transform(train_df[feature_names].values)
        test_features = scaler.transform(test_df[feature_names].values)

        # 5. Build windows separately for train/test
        train_close = train_df["close"].values
        test_close = test_df["close"].values

        X_train, base_train, next_train = self.window.build(train_features, train_close)
        if len(X_train) < 20:
            return {"ok": False, "error": "Not enough train sequences"}
        y_train = self.target_builder.build(base_train, next_train)

        X_test, base_test, next_test = self.window.build(test_features, test_close)
        if len(X_test) < 5:
            return {"ok": False, "error": "Not enough test sequences (lost seq_len due to split)"}

        # 6. Fit
        forecaster.fit(X_train, y_train)

        # 7. Evaluate on reconstructed prices
        ret_pred = forecaster.predict(X_test)
        price_pred = np.array([
            self.target_builder.reconstruct(b, r) for b, r in zip(base_test, ret_pred)
        ])
        metrics = {
            "mae": float(mean_absolute_error(next_test, price_pred)),
            "r2": float(r2_score(next_test, price_pred)),
        }

        # Save training results CSV
        results_csv_path = os.path.join(self.data_dir, f"{symbol}_{model_name}_results_fixed_{timestamp}.csv")
        results_df = pd.DataFrame([{
            "symbol": symbol,
            "model_name": model_name,
            "timestamp": timestamp,
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "seq_len": self.seq_len,
            "pipeline": "fixed_no_leakage",
        }])
        results_df.to_csv(results_csv_path, index=False)
        print(f"  Saved training results to: {results_csv_path}")

        # 8. Persist
        self.registry.save(
            symbol=symbol,
            forecaster=forecaster,
            scaler=scaler,
            feature_names=feature_names,
            seq_len=self.seq_len,
            metrics=metrics,
        )
        return {"ok": True, "metrics": metrics}
