"""
LightGBMForecaster - gradient-boosted trees over flattened windows.

Flattens each (seq_len, n_features) window into a single feature vector and
fits a LightGBM classifier (or regressor in regression mode).

Why this model: peer-reviewed studies rank gradient boosting as the strongest
family for crypto *direction* classification - LightGBM achieved Sharpe 2.64
and beat buy-and-hold on BTC Buy/Sell/Hold signals (ISD 2025), XGBoost hit
0.930 trend accuracy (JDSIS 2023). It is also fast (<1s fit), deterministic,
and exposes feature_importances_ for explainability.
"""
import os

import numpy as np
import joblib
import lightgbm as lgb

from src.domain.interfaces.forecaster import IForecaster


class LightGBMForecaster(IForecaster):
    """LightGBM classifier/regressor for up/down trend (or return) prediction."""

    MODEL_FILE = "model.pkl"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        mode: str = "classification",
        seed: int = 42,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        self._mode = mode
        self._seed = seed
        self._model = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def _build_model(self) -> None:
        common = dict(self._params, random_state=self._seed, verbosity=-1)
        if self._mode == "regression":
            self._model = lgb.LGBMRegressor(**common)
        else:
            self._model = lgb.LGBMClassifier(**common)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        # LightGBM needs no validation split for early stopping by default;
        # pipeline-provided X_val/y_val are accepted and ignored.
        self._build_model()
        self._model.fit(self._flatten(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "regression":
            return self._model.predict(self._flatten(X))
        return self._model.predict_proba(self._flatten(X))[:, 1]

    def feature_importances(self, feature_names=None) -> dict:
        """Return {feature: importance} - useful for the dashboard (real, not demo)."""
        if self._model is None:
            return {}
        names = feature_names or [f"f{i}" for i in range(len(self._model.feature_importances_))]
        return dict(zip(names, self._model.feature_importances_.tolist()))

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, self.MODEL_FILE))

    def load(self, directory: str) -> None:
        self._model = joblib.load(os.path.join(directory, self.MODEL_FILE))

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)
