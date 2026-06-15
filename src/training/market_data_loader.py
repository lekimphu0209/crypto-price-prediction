"""
MarketDataLoader - adapts a data source into a pandas DataFrame.

Single responsibility: fetch OHLCV from any IDataSource-like provider and
return a clean DataFrame. Depends on the abstraction, not on Binance.
"""
import pandas as pd


class MarketDataLoader:
    """Load OHLCV data as a DataFrame from a data source."""

    def __init__(self, data_source, interval: str = "1d", limit: int = 1000):
        self.data_source = data_source
        self.interval = interval
        self.limit = limit

    def load(self, symbol: str) -> pd.DataFrame:
        ohlcv = self.data_source.fetch_ohlcv(symbol, self.interval, limit=self.limit)
        df = pd.DataFrame([{
            "timestamp": o.timestamp,
            "open": o.open,
            "high": o.high,
            "low": o.low,
            "close": o.close,
            "volume": o.volume,
        } for o in ohlcv])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
