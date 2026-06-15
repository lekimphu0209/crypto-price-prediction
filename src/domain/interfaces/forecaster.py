"""
IForecaster Interface - Abstraction for a forecasting model.

Responsibility: learn a mapping from prepared input windows to a target,
and predict on new windows. It does NOT do feature engineering, scaling,
or windowing (those are separate responsibilities).
Replaceable: LinearRegression, RNN, LSTM, BiLSTM, Transformer, GRU, etc.
"""
from abc import ABC, abstractmethod
import numpy as np


class IForecaster(ABC):
    """Contract for a model that trains/predicts on prepared 3D windows."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: input windows, shape (samples, seq_len, n_features) — already scaled
            y: target values, shape (samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on input windows.

        Args:
            X: input windows, shape (samples, seq_len, n_features)

        Returns:
            1D array of predictions, shape (samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist the model into the given directory."""
        raise NotImplementedError

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load the model from the given directory."""
        raise NotImplementedError
