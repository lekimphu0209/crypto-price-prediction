"""
Binance Data Provider Implementation
"""
import time

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
        """Initialize Binance client (spot markets only, rate-limited)."""
        self.client = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._name = "Binance"

    def _fetch_ohlcv_page(self, symbol, timeframe, since, until, limit=1000, retries=4):
        """
        Fetch one page of raw Binance klines.

        Calls publicGetKlines directly instead of client.fetch_ohlcv so that
        no load_markets() / exchangeInfo request is needed (that endpoint is
        heavy and intermittently times out). Raw kline rows keep the same
        [ms, open, high, low, close, volume, ...] layout our caller reads.
        """
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": since,
            "endTime": until,
            "limit": limit,
        }
        last_err = None
        for attempt in range(retries):
            try:
                return self.client.publicGetKlines(params)
            except ccxt.NetworkError as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise last_err
    
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

            # Milliseconds per candle, used to walk `since` backwards through
            # history when `limit` exceeds one API page (~1000 candles).
            interval_ms = {
                '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
                '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
                '1w': 604_800_000,
            }[ccxt_interval]

            # Oldest candle we want: `limit` candles back from now (or from
            # start_date if the caller gave one).
            until = int(end_date.timestamp() * 1000) if end_date else int(
                datetime.utcnow().timestamp() * 1000
            )
            first_target = (
                int(start_date.timestamp() * 1000)
                if start_date else until - limit * interval_ms
            )

            # Paginate: Binance returns up to ~1000 candles per request, so
            # loop from first_target forward until we reach `until`.
            all_rows = []
            page_since = first_target
            seen_first = None
            while True:
                page = self._fetch_ohlcv_page(
                    symbol, ccxt_interval, since=page_since, until=until, limit=1000
                )
                if not page:
                    break
                # Skip duplicate of previous page's last candle
                rows = page[1:] if seen_first == page[0][0] else page
                all_rows.extend(rows)
                last_ts = page[-1][0]
                seen_first = last_ts
                # Stop when: exhausted history, reached `until`, or hit limit
                if len(page) < 1000 or last_ts >= until or len(all_rows) >= limit:
                    break
                page_since = last_ts + interval_ms

            if len(all_rows) > limit:
                all_rows = all_rows[-limit:]

            # Convert to OHLCV entities
            ohlcv_list = []
            for candle in all_rows:
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
