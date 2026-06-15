"""
ITargetBuilder Interface - Abstraction for the prediction target.

Responsibility: define what the model predicts and how to convert that
prediction back into an absolute price.
Replaceable: next-day return, log-return, up/down classification, etc.
"""
from abc import ABC, abstractmethod
import numpy as np


class ITargetBuilder(ABC):
    """Contract for building training targets and reconstructing prices."""

    @abstractmethod
    def build(self, base_close: np.ndarray, next_close: np.ndarray) -> np.ndarray:
        """
        Build the training target.

        Args:
            base_close: close price of the last day in each input window
            next_close: actual next-day close price (the value being forecast)

        Returns:
            Target array (e.g. next-day return) aligned with the inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, base_close: float, prediction: float) -> float:
        """
        Convert a model prediction back into an absolute price.

        Args:
            base_close: current/last known close price
            prediction: raw model output (e.g. a return)

        Returns:
            Predicted absolute price.
        """
        raise NotImplementedError
