"""
IDataRepository Interface - Abstraction for data storage
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from pandas import DataFrame
from src.domain.entities.ohlcv import OHLCV
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class IDataRepository(ABC):
    """Interface for data repository"""
    
    @abstractmethod
    def save_ohlcv(self, data: List[OHLCV], symbol: Symbol, interval: Interval) -> None:
        """Save OHLCV data"""
        pass
    
    @abstractmethod
    def load_ohlcv(
        self,
        symbol: Symbol,
        interval: Interval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OHLCV]:
        """Load OHLCV data"""
        pass
    
    @abstractmethod
    def get_latest_data(
        self,
        symbol: Symbol,
        interval: Interval,
        limit: int = 100
    ) -> List[OHLCV]:
        """Get latest OHLCV data"""
        pass
    
    @abstractmethod
    def load_as_dataframe(self, symbol: Symbol, interval: Interval) -> DataFrame:
        """Load data as pandas DataFrame"""
        pass
