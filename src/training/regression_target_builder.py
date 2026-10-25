"""
RegressionTargetBuilder - continuous target for return forecasting.

Builds the log return between the base close and the future close:
    target = log(future_close / base_close)

`reconstruct` converts a predicted return back to a direction label:
    1 if return >= 0 else 0

Implements ITargetBuilder.
"""
import numpy as np

from src.domain.interfaces.target_builder import ITargetBuilder


class RegressionTargetBuilder(ITargetBuilder):
    """Continuous log-return target builder."""

    def build(self, base_close: np.ndarray, future_close: np.ndarray) -> np.ndarray:
        return np.log(future_close / base_close)

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y) >= 0).astype(int)
