"""
Data Collection Use Case
"""
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.infrastructure.data_providers.yfinance_provider import YahooFinanceProvider
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval


class CollectDataUseCase:
    """Use case for collecting data from various sources"""
    
    def __init__(
        self,
        binance_provider: BinanceProvider,
        yfinance_provider: YahooFinanceProvider,
        data_repository: CSVDataRepository
    ):
        """
        Args:
            binance_provider: Binance data provider
            yfinance_provider: Yahoo Finance data provider
            data_repository: Data repository
        """
        self.binance_provider = binance_provider
        self.yfinance_provider = yfinance_provider
        self.data_repository = data_repository
    
    def execute(
        self,
        symbol: str,
        interval: str,
        source: str = "binance",
        limit: int = 1000,
        days_back: int = 365
    ) -> List:
        """
        Execute data collection
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            source: Data source ('binance' or 'yahoo')
            limit: Number of candles
            days_back: Number of days to look back
        
        Returns:
            List of collected data
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        if source == "binance":
            data = self.binance_provider.fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                limit=limit,
                start_date=start_date
            )
        elif source == "yahoo":
            # Convert symbol for Yahoo Finance
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            data = self.yfinance_provider.fetch_ohlcv(
                symbol=yahoo_symbol,
                interval=interval,
                limit=limit,
                start_date=start_date
            )
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Save to repository
        if data:
            symbol_vo = Symbol(symbol)
            interval_vo = Interval(interval)
            self.data_repository.save_ohlcv(data, symbol_vo, interval_vo)
        
        return data
    
    def collect_macro_data(
        self,
        symbols: List[str],
        period: str = "1y"
    ) -> dict:
        """
        Collect macro data (Gold, DXY, etc.)
        
        Args:
            symbols: List of macro symbols (e.g., ['GLD', 'UUP'])
            period: Time period
        
        Returns:
            Dictionary of DataFrames
        """
        macro_data = {}
        
        for symbol in symbols:
            try:
                df = self.yfinance_provider.fetch_macro_data(symbol, period)
                macro_data[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return macro_data
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """
        Convert Binance symbol to Yahoo Finance symbol
        
        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
        
        Returns:
            Yahoo Finance symbol (e.g., "BTC-USD")
        """
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}-USD"
        elif "BUSD" in symbol:
            base = symbol.replace("BUSD", "")
            return f"{base}-USD"
        return symbol
