"""
CSV Data Repository Implementation
"""
import os
import pandas as pd
from typing import List, Optional
from datetime import datetime

from src.domain.interfaces.data_repository import IDataRepository
from src.domain.entities.ohlcv import OHLCV
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class CSVDataRepository(IDataRepository):
    """CSV file-based data repository"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_file_path(self, symbol: Symbol, interval: Interval) -> str:
        """Get file path for symbol and interval"""
        symbol_lower = symbol.base.lower()
        return os.path.join(self.data_dir, f"{symbol_lower}_{interval.value}.csv")
    
    def save_ohlcv(self, data: List[OHLCV], symbol: Symbol, interval: Interval) -> None:
        """
        Save OHLCV data to CSV
        
        Args:
            data: List of OHLCV entities
            symbol: Symbol
            interval: Interval
        """
        if not data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': o.timestamp,
            'open': o.open,
            'high': o.high,
            'low': o.low,
            'close': o.close,
            'volume': o.volume,
            'symbol': o.symbol
        } for o in data])
        
        # Save to CSV
        file_path = self._get_file_path(symbol, interval)
        df.to_csv(file_path, index=False)
    
    def load_ohlcv(
        self,
        symbol: Symbol,
        interval: Interval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Load OHLCV data from CSV
        
        Args:
            symbol: Symbol
            interval: Interval
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of OHLCV entities
        """
        file_path = self._get_file_path(symbol, interval)
        
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Convert to entities
        ohlcv_list = [
            OHLCV(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                symbol=row['symbol']
            )
            for _, row in df.iterrows()
        ]
        
        return ohlcv_list
    
    def get_latest_data(
        self,
        symbol: Symbol,
        interval: Interval,
        limit: int = 100
    ) -> List[OHLCV]:
        """
        Get latest OHLCV data
        
        Args:
            symbol: Symbol
            interval: Interval
            limit: Number of records to return
        
        Returns:
            List of OHLCV entities
        """
        file_path = self._get_file_path(symbol, interval)
        
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp and get latest
        df = df.sort_values('timestamp').tail(limit)
        
        # Convert to entities
        ohlcv_list = [
            OHLCV(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                symbol=row['symbol']
            )
            for _, row in df.iterrows()
        ]
        
        return ohlcv_list
    
    def load_as_dataframe(self, symbol: Symbol, interval: Interval) -> pd.DataFrame:
        """
        Load data as pandas DataFrame
        
        Args:
            symbol: Symbol
            interval: Interval
        
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self._get_file_path(symbol, interval)
        
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
