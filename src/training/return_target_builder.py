"""
TrendTargetBuilder - builds an up/down trend label over a horizon.

The model predicts whether the close price `target_horizon` days ahead is
higher (1 = UP / TANG) or not (0 = DOWN / GIAM) than the last close of the
input window. `reconstruct` turns a model probability into a 0/1 label.

When `threshold` is provided, only movements larger than the threshold are
labelled; returns within [-threshold, +threshold] are marked as NaN and
excluded from training.

target = 1  if (future - base) / base > threshold
         0  if (future - base) / base < -threshold
         NaN otherwise
label  = 1 if probability >= 0.5 else 0
"""
import numpy as np

from src.domain.interfaces.target_builder import ITargetBuilder


class TrendTargetBuilder(ITargetBuilder):
    """Binary up/down trend target over a fixed horizon."""

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: Minimum absolute return (as a fraction) to assign a
                label. Returns within [-threshold, +threshold] are set to NaN
                and ignored. Default 0.0 keeps the original binary behaviour.
        """
        self.threshold = threshold

    def build(self, base_close: np.ndarray, future_close: np.ndarray) -> np.ndarray:
        base_close = np.asarray(base_close)
        future_close = np.asarray(future_close)
        return_pct = (future_close - base_close) / base_close
        target = np.full_like(return_pct, np.nan, dtype=float)
        target[return_pct > self.threshold] = 1
        target[return_pct < -self.threshold] = 0
        return target

    def reconstruct(self, base_close: float, prediction: float) -> int:
        return int(float(prediction) >= 0.5)


# Backwards-compatible alias (composition is updated in a later step).
ReturnTargetBuilder = TrendTargetBuilder
