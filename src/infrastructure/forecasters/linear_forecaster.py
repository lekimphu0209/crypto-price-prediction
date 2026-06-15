"""
LinearForecaster - Linear Regression over flattened windows.

Flattens each (seq_len, n_features) window into a single feature vector and
fits an ordinary least-squares model. Implements IForecaster.
"""
import os

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

from src.domain.interfaces.forecaster import IForecaster


class LinearForecaster(IForecaster):
    """Linear Regression forecaster."""

    MODEL_FILE = "model.pkl"

    def __init__(self):
        self._model = LinearRegression()

    @property
    def name(self) -> str:
        return "Linear Regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._flatten(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._flatten(X)).flatten()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, self.MODEL_FILE))

    def load(self, directory: str) -> None:
        self._model = joblib.load(os.path.join(directory, self.MODEL_FILE))

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)
