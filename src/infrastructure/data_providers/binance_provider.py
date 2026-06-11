"""
Binance Data Provider Implementation
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import ccxt

from src.domain.interfaces.data_source import IDataSource
from src.domain.entities.ohlcv import OHLCV
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class BinanceProvider(IDataSource):
    """Binance data provider implementation"""
    
    def __init__(self):
        """Initialize Binance client"""
        self.client = ccxt.binance()
        self._name = "Binance"
    
    @property
    def name(self) -> str:
        """Get data source name"""
        return self._name
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data from Binance
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Time interval (e.g., "1h", "1d")
            limit: Number of candles to fetch
            start_date: Start date
            end_date: End date
        
        Returns:
            List of OHLCV entities
        """
        try:
            # Map interval to ccxt format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d',
                '1w': '1w'
            }
            
            ccxt_interval = interval_map.get(interval, '1h')
            
            # Fetch data
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)
            
            ohlcv_data = self.client.fetch_ohlcv(
                symbol,
                timeframe=ccxt_interval,
                limit=limit,
                since=since
            )
            
            # Convert to OHLCV entities
            ohlcv_list = []
            for candle in ohlcv_data:
                timestamp = datetime.fromtimestamp(candle[0] / 1000)
                
                # Filter by end_date if provided
                if end_date and timestamp > end_date:
                    continue
                
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    symbol=symbol
                )
                ohlcv_list.append(ohlcv)
            
            return ohlcv_list
            
        except Exception as e:
            raise RuntimeError(f"Error fetching data from Binance: {e}")
    
    def is_available(self) -> bool:
        """Check if Binance is available"""
        try:
            self.client.fetch_markets()
            return True
        except:
            return False
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        try:
            markets = self.client.fetch_markets()
            return [market['symbol'] for market in markets if market['active']]
        except:
            return []
