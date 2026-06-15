"""
IModelRegistry Interface - Abstraction for model persistence.

Responsibility: save and load a trained forecaster together with its
metadata (feature scaler, feature names, metrics, config).
Replaceable: file system, MongoDB, S3, etc.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.domain.interfaces.forecaster import IForecaster


class IModelRegistry(ABC):
    """Contract for storing and retrieving trained models."""

    @abstractmethod
    def save(
        self,
        symbol: str,
        forecaster: IForecaster,
        scaler: Any,
        feature_names: List[str],
        seq_len: int,
        metrics: Dict[str, float],
    ) -> None:
        """Persist a trained forecaster and all artifacts needed to predict."""
        raise NotImplementedError

    @abstractmethod
    def load(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """
        Load a trained model bundle.

        Returns:
            Dict with keys: forecaster, scaler, feature_names, seq_len, metrics.

        Raises:
            FileNotFoundError if the model does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, symbol: str, model_name: str) -> bool:
        """Check whether a trained model exists."""
        raise NotImplementedError
