"""
LinearForecaster - Logistic Regression over flattened windows.

Flattens each (seq_len, n_features) window into a single feature vector and
fits a logistic-regression classifier. `predict` returns the probability of
the positive class (UP / TANG). Implements IForecaster.

When `seed` is provided, the model uses SGDClassifier with log_loss (still a
linear logistic model) so that different seeds produce different models. When
seed is None, the standard deterministic LogisticRegression is used.
"""
import os

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.domain.interfaces.forecaster import IForecaster


class LinearForecaster(IForecaster):
    """Logistic Regression classifier (up/down trend)."""

    MODEL_FILE = "model.pkl"

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict = "balanced",
        max_iter: int = 1000,
        seed: int | None = None,
    ):
        self._C = C
        self._class_weight = class_weight
        self._max_iter = max_iter
        self._seed = seed
        self._model = None

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def _build_model(self, n_samples: int) -> None:
        if self._seed is None:
            self._model = LogisticRegression(
                C=self._C,
                max_iter=self._max_iter,
                class_weight=self._class_weight,
            )
        else:
            # SGD with log_loss is still a logistic regression, but stochastic
            # across seeds so that n=50 (seed x fold) is real.
            alpha = 1.0 / (self._C * n_samples)
            self._model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=self._max_iter,
                class_weight=self._class_weight,
                random_state=self._seed,
                shuffle=True,
                tol=1e-3,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        # Logistic regression does not use early stopping/validation data,
        # but the pipeline may pass X_val/y_val. We accept and ignore them.
        self._build_model(n_samples=X.shape[0])
        self._model.fit(self._flatten(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._flatten(X))[:, 1]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, self.MODEL_FILE))

    def load(self, directory: str) -> None:
        self._model = joblib.load(os.path.join(directory, self.MODEL_FILE))

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)
