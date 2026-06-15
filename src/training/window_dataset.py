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

    def __init__(self, seq_len: int = 20):
        self.seq_len = seq_len

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
            next_close: next-day close (forecast target source), shape (samples,)
        """
        X, base_close, next_close = [], [], []
        for i in range(len(feature_matrix) - self.seq_len):
            X.append(feature_matrix[i:i + self.seq_len])
            base_close.append(close[i + self.seq_len - 1])
            next_close.append(close[i + self.seq_len])
        return np.array(X), np.array(base_close), np.array(next_close)

    def last_window(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return the most recent window, shape (1, seq_len, n_features)."""
        window = feature_matrix[-self.seq_len:]
        return window[np.newaxis, :, :]
