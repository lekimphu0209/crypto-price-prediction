"""
WindowDataset - builds sliding windows over a feature matrix.

Single responsibility: convert a 2D feature matrix (n_days, n_features) into
3D windows (samples, seq_len, n_features) plus the aligned base/next close
prices used by the target builder.
"""
from typing import Tuple

import numpy as np


class WindowDataset:
    """Create time-ordered sliding windows for sequence models."""

    def __init__(self, seq_len: int = 20, target_horizon: int = 7):
        self.seq_len = seq_len
        self.target_horizon = target_horizon

    def build(
        self,
        feature_matrix: np.ndarray,
        close: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build windows.

        Args:
            feature_matrix: scaled features, shape (n_days, n_features)
            close: raw close prices, shape (n_days,)

        Returns:
            X: shape (samples, seq_len, n_features)
            base_close: close of the last day in each window, shape (samples,)
            future_close: close `target_horizon` days after the window, shape (samples,)
        """
        X, base_close, future_close = [], [], []
        h = self.target_horizon
        for i in range(len(feature_matrix) - self.seq_len - h + 1):
            X.append(feature_matrix[i:i + self.seq_len])
            base_close.append(close[i + self.seq_len - 1])
            future_close.append(close[i + self.seq_len - 1 + h])
        return np.array(X), np.array(base_close), np.array(future_close)

    def last_window(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return the most recent window, shape (1, seq_len, n_features)."""
        window = feature_matrix[-self.seq_len:]
        return window[np.newaxis, :, :]
