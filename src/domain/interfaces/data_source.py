"""
IDataSource Interface - Abstraction for data sources
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from src.domain.entities.ohlcv import OHLCV
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class IDataSource(ABC):
    """Interface for data sources"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get data source name"""
        pass
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: Symbol,
        interval: Interval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV data from source"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available"""
        pass
