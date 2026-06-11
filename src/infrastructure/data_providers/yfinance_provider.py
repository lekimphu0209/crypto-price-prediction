"""
Yahoo Finance Data Provider Implementation
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
import yfinance as yf

from src.domain.interfaces.data_source import IDataSource
from src.domain.entities.ohlcv import OHLCV


class YahooFinanceProvider(IDataSource):
    """Yahoo Finance data provider implementation"""
    
    def __init__(self):
        """Initialize Yahoo Finance client"""
        self._name = "YahooFinance"
    
    @property
    def name(self) -> str:
        """Get data source name"""
        return self._name
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD" for Bitcoin)
            interval: Time interval (e.g., "1d", "1h")
            limit: Number of candles to fetch
            start_date: Start date
            end_date: End date
        
        Returns:
            List of OHLCV entities
        """
        try:
            # Map interval to yfinance format
            interval_map = {
                '1d': '1d',
                '1h': '1h',
                '1wk': '1wk',
                '1mo': '1mo'
            }
            
            yf_interval = interval_map.get(interval, '1d')
            
            # Calculate date range
            if start_date is None:
                start_date = datetime.now() - timedelta(days=limit)
            if end_date is None:
                end_date = datetime.now()
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
            
            if df.empty:
                return []
            
            # Convert to OHLCV entities
            ohlcv_list = []
            for idx, row in df.iterrows():
                ohlcv = OHLCV(
                    timestamp=idx,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    symbol=symbol
                )
                ohlcv_list.append(ohlcv)
            
            # Limit results
            return ohlcv_list[-limit:]
            
        except Exception as e:
            raise RuntimeError(f"Error fetching data from Yahoo Finance: {e}")
    
    def fetch_macro_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch macro data (e.g., Gold, DXY)
        
        Args:
            symbol: Symbol (e.g., "GLD" for Gold, "UUP" for DXY)
            period: Time period (e.g., "1y", "2y")
        
        Returns:
            DataFrame with macro data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            return df
        except Exception as e:
            raise RuntimeError(f"Error fetching macro data: {e}")
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            ticker = yf.Ticker("BTC-USD")
            ticker.history(period="1d")
            return True
        except:
            return False
