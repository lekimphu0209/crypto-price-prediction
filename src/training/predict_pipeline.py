"""
PredictPipeline - orchestrates the prediction flow.

Loads a trained model bundle from the registry, rebuilds the latest feature
window, predicts the next-day return, reconstructs the price, and produces a
multi-day forecast via a damped trend extension.

Returns a dict whose shape matches what the dashboard/API already expect, so
no downstream code needs to change.
"""
import time
from typing import Dict

import numpy as np
import pandas as pd

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.domain.interfaces.target_builder import ITargetBuilder
from src.domain.interfaces.model_registry import IModelRegistry
from src.training.market_data_loader import MarketDataLoader


class PredictPipeline:
    """Coordinates prediction using a trained forecaster."""

    def __init__(
        self,
        loader: MarketDataLoader,
        extractor: IFeatureExtractor,
        target_builder: ITargetBuilder,
        registry: IModelRegistry,
        cache_ttl: int = 300,
    ):
        self.loader = loader
        self.extractor = extractor
        self.target_builder = target_builder
        self.registry = registry
        self.cache_ttl = cache_ttl
        self._data_cache: dict = {}

    def _load_data(self, symbol: str) -> pd.DataFrame:
        now = time.time()
        cached = self._data_cache.get(symbol)
        if cached and now - cached[1] < self.cache_ttl:
            return cached[0].copy()
        df = self.loader.load(symbol)
        self._data_cache[symbol] = (df.copy(), now)
        return df

    def predict(self, symbol: str, model_name: str, display_days: int, horizon: int = 7) -> Dict:
        # 1. Load model bundle
        try:
            bundle = self.registry.load(symbol, model_name)
        except FileNotFoundError as e:
            return {"ok": False, "message": str(e)}

        forecaster = bundle["forecaster"]
        scaler = bundle["scaler"]
        feature_names = bundle["feature_names"]
        seq_len = bundle["seq_len"]
        metrics = bundle.get("metrics", {})

        # 2. Load data
        try:
            df = self._load_data(symbol)
        except Exception as e:
            return {"ok": False, "message": f"Failed to fetch data: {e}"}

        if len(df) < seq_len + 1:
            return {"ok": False, "message": f"Not enough data: {len(df)}"}

        # 3. Features -> scale -> latest window
        feat_df = self.extractor.extract(df)
        feature_matrix = scaler.transform(feat_df[feature_names].values)
        window = feature_matrix[-seq_len:][np.newaxis, :, :]

        # 4. Predict next-day return -> reconstruct price
        pred_return = float(forecaster.predict(window)[0])
        close = feat_df["close"].values
        current_price = float(close[-1])
        predicted_next = self.target_builder.reconstruct(current_price, pred_return)

        # 5. Multi-day damped trend extension
        preds = [predicted_next]
        daily_change = predicted_next - current_price
        for i in range(1, horizon):
            preds.append(current_price + daily_change * (0.85 ** i) * (i + 1))
        forecast_values = np.array(preds)

        # 6. Assemble response (matches existing API format)
        dates = pd.to_datetime(feat_df["timestamp"]) if "timestamp" in feat_df else pd.to_datetime(df["timestamp"])
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        change_pct = (predicted_next - current_price) / current_price * 100
        train_r2 = float(metrics.get("r2", 0.0))
        train_mae = float(metrics.get("mae", 0.0))
        confidence = max(0.0, min(1.0, train_r2)) * 100

        n = min(display_days, len(close))
        return {
            "ok": True,
            "is_real": True,
            "source": "loaded_clean",
            "model_name": model_name,
            "hist_dates": dates.iloc[-n:].tolist(),
            "hist_values": close[-n:].tolist(),
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast_values.tolist(),
            "current_price": current_price,
            "predicted_tomorrow": predicted_next,
            "change_pct": change_pct,
            "confidence": confidence,
            "train_r2": train_r2,
            "train_mae": train_mae,
        }
