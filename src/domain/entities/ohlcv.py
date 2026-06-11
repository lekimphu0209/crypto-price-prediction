"""
OHLCV Entity - Open, High, Low, Close, Volume data
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OHLCV:
    """Entity representing OHLCV candlestick data"""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    
    def __post_init__(self):
        """Validate OHLCV data"""
        if self.high < self.low:
            raise ValueError("High cannot be less than Low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("High cannot be less than Open or Close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low cannot be greater than Open or Close")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def range_(self) -> float:
        """Price range (high - low)"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Candle body (close - open)"""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """Is candle bullish (close > open)"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Is candle bearish (close < open)"""
        return self.close < self.open
    
    @property
    def upper_wick(self) -> float:
        """Upper wick (high - max(open, close))"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick (min(open, close) - low)"""
        return min(self.open, self.close) - self.low
