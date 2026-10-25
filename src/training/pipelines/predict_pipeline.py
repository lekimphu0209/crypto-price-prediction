"""
PredictPipeline - orchestrates the prediction flow.

Loads a trained model bundle from the registry (model + scaler + meta),
rebuilds the *same* feature extractor the model was trained with (from
meta.json's extractor_config), builds the latest window and predicts.

The models are up/down TREND CLASSIFIERS over `horizon` days, so the output
is a direction (UP/DOWN) plus the model's probability - not a price point.
Regression-mode models additionally expose `predicted_return`.

Response shape matches what the dashboard/API consume; adding a new model or
extractor requires no changes here because everything is rebuilt from meta.
"""
import time
from typing import Dict

import numpy as np
import pandas as pd

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.domain.interfaces.model_registry import IModelRegistry
from src.features.extractors.registry import build_extractor
from src.training.datasets.market_data_loader import MarketDataLoader


class PredictPipeline:
    """Coordinates prediction using a trained forecaster."""

    def __init__(
        self,
        loader: MarketDataLoader,
        extractor: IFeatureExtractor = None,
        registry: IModelRegistry = None,
        cache_ttl: int = 300,
    ):
        self.loader = loader
        # Default extractor only used for legacy bundles without extractor_config
        self.default_extractor = extractor
        self.registry = registry
        self.cache_ttl = cache_ttl
        self._data_cache: dict = {}
        self._extractor_cache: dict = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_data(self, symbol: str) -> pd.DataFrame:
        now = time.time()
        cached = self._data_cache.get(symbol)
        if cached and now - cached[1] < self.cache_ttl:
            return cached[0].copy()
        df = self.loader.load(symbol)
        self._data_cache[symbol] = (df.copy(), now)
        return df

    def _extractor_for(self, bundle: dict) -> IFeatureExtractor:
        """Rebuild the extractor the model was trained with, cached by config."""
        config = bundle.get("extractor_config")
        if not config:
            if self.default_extractor is None:
                raise ValueError("Model bundle has no extractor_config and no default extractor set")
            return self.default_extractor
        key = f"{config.get('name')}:{config.get('params')}"
        if key not in self._extractor_cache:
            self._extractor_cache[key] = build_extractor(config)
        return self._extractor_cache[key]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict(self, symbol: str, model_name: str, display_days: int, horizon: int = None) -> Dict:
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
        mode = bundle.get("mode", "classification")
        model_horizon = bundle.get("horizon", 7)
        threshold = bundle.get("threshold", 0.5)

        # 2. Load data
        try:
            df = self._load_data(symbol)
        except Exception as e:
            return {"ok": False, "message": f"Failed to fetch data: {e}"}

        if len(df) < seq_len + 1:
            return {"ok": False, "message": f"Not enough data: {len(df)}"}

        # 3. Features (extractor reconstructed from the saved config) -> scale -> window
        try:
            extractor = self._extractor_for(bundle)
        except Exception as e:
            return {"ok": False, "message": f"Failed to build extractor: {e}"}

        try:
            feat_df = extractor.extract(df)
            feature_matrix = scaler.transform(feat_df[feature_names].values)
        except Exception as e:
            return {"ok": False, "message": f"Feature extraction failed: {e}"}

        if len(feature_matrix) < seq_len:
            return {"ok": False, "message": f"Not enough feature rows: {len(feature_matrix)}"}

        window = feature_matrix[-seq_len:][np.newaxis, :, :]

        # 4. Predict -> direction + probability
        raw = float(forecaster.predict(window)[0])
        close = feat_df["close"].values
        current_price = float(close[-1])

        if mode == "regression":
            # raw is a predicted log-return over the horizon
            direction = "UP" if raw >= 0 else "DOWN"
            direction_label = "TANG" if direction == "UP" else "GIAM"
            probability = None
            predicted_return = raw
            change_pct = float((np.exp(raw) - 1.0) * 100.0)
        else:
            # raw is a probability of UP; apply the tuned decision threshold
            probability = raw
            direction = "UP" if raw >= threshold else "DOWN"
            direction_label = "TANG" if direction == "UP" else "GIAM"
            predicted_return = None
            # Signed "strength" proxy: distance of proba from the threshold
            change_pct = float((raw - threshold) * 100.0)

        # 5. Assemble response
        dates = (
            pd.to_datetime(feat_df["timestamp"])
            if "timestamp" in feat_df
            else pd.to_datetime(df["timestamp"])
        )

        bal_acc = float(metrics.get("balanced_accuracy", 0.0))
        confidence = max(0.0, min(1.0, bal_acc)) * 100.0

        n = min(display_days, len(close))
        return {
            "ok": True,
            "is_real": True,
            "model_name": bundle.get("model_name", model_name),
            "mode": mode,
            "symbol": symbol,
            # direction result
            "direction": direction,
            "direction_label": direction_label,
            "probability": probability,
            "predicted_return": predicted_return,
            "threshold": threshold,
            "horizon": model_horizon,
            "current_price": current_price,
            "change_pct": change_pct,
            # chart data (history only - classifiers do not emit a price path)
            "hist_dates": dates.iloc[-n:].tolist(),
            "hist_values": close[-n:].tolist(),
            # training metrics for transparency
            "confidence": confidence,
            "train_metrics": metrics,
            "train_r2": float(metrics.get("r2", 0.0)),
            "train_mae": float(metrics.get("mae", 0.0)),
        }
