"""
FileModelRegistry - stores trained models on the local file system.

Layout (one folder per model, clean and self-describing):

    models/
      BTCUSDT/
        linear_regression/
          model.pkl
          scaler.pkl
          meta.json
        lstm/
          model.keras
          scaler.pkl
          meta.json
      ETHUSDT/
        ...

Implements IModelRegistry so storage can later be swapped for a DB/cloud
backend without changing the training or prediction pipelines.
"""
import os
import json
from typing import Any, Dict, List

import joblib

from src.domain.interfaces.model_registry import IModelRegistry
from src.domain.interfaces.forecaster import IForecaster
from src.infrastructure.forecasters.factory import create_forecaster

SCALER_FILE = "scaler.pkl"
META_FILE = "meta.json"


def _slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


class FileModelRegistry(IModelRegistry):
    """File-system implementation of the model registry."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def _model_dir(self, symbol: str, model_name: str) -> str:
        return os.path.join(self.models_dir, symbol, _slug(model_name))

    def save(
        self,
        symbol: str,
        forecaster: IForecaster,
        scaler: Any,
        feature_names: List[str],
        seq_len: int,
        metrics: Dict[str, float],
    ) -> None:
        directory = self._model_dir(symbol, forecaster.name)
        os.makedirs(directory, exist_ok=True)

        forecaster.save(directory)
        joblib.dump(scaler, os.path.join(directory, SCALER_FILE))

        meta = {
            "symbol": symbol,
            "model_name": forecaster.name,
            "feature_names": feature_names,
            "seq_len": seq_len,
            "n_features": len(feature_names),
            "metrics": metrics,
        }
        with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, symbol: str, model_name: str) -> Dict[str, Any]:
        directory = self._model_dir(symbol, model_name)
        meta_path = os.path.join(directory, META_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model not found: {symbol}/{_slug(model_name)}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        forecaster = create_forecaster(meta["model_name"])
        forecaster.load(directory)
        scaler = joblib.load(os.path.join(directory, SCALER_FILE))

        return {
            "forecaster": forecaster,
            "scaler": scaler,
            "feature_names": meta["feature_names"],
            "seq_len": meta["seq_len"],
            "metrics": meta.get("metrics", {}),
        }

    def exists(self, symbol: str, model_name: str) -> bool:
        return os.path.exists(os.path.join(self._model_dir(symbol, model_name), META_FILE))
