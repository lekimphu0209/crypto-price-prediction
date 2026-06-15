"""
ReturnTargetBuilder - predicts the next-day log return.

Why log return instead of simple return:
    Log returns are symmetric and more stable for financial time series.
    They are additive (log returns sum over time) and better suited for
    models. The price is reconstructed using exponential.

target  = log(next_close / base_close)
price   = base_close * exp(prediction)
"""
import numpy as np

from src.domain.interfaces.target_builder import ITargetBuilder


class ReturnTargetBuilder(ITargetBuilder):
    """Next-day log return target."""

    def build(self, base_close: np.ndarray, next_close: np.ndarray) -> np.ndarray:
        return np.log(next_close / base_close)

    def reconstruct(self, base_close: float, prediction: float) -> float:
        return float(base_close) * np.exp(float(prediction))
