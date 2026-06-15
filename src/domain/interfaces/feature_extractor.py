"""
IFeatureExtractor Interface - Abstraction for feature engineering.

Responsibility: turn raw OHLCV data into a feature matrix.
Replaceable: technical features, macro features, sentiment features, etc.
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class IFeatureExtractor(ABC):
    """Contract for any component that builds features from OHLCV data."""

    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from raw OHLCV data.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame containing at least the columns in feature_names() plus 'close',
            with NaN rows dropped.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the ordered list of feature column names produced by extract()."""
        raise NotImplementedError
