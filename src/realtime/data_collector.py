"""
Realtime Data Collector Module
Thu thập dữ liệu theo thời gian thực từ nhiều nguồn
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
import yfinance as yf
from collections import deque
import threading
import time


class RealtimeDataCollector:
    """Class để thu thập dữ liệu theo thời gian thực"""
    
    def __init__(self, symbol: str = "BTCUSDT", buffer_size: int = 1000):
        """
        Args:
            symbol: Symbol trading pair (e.g., BTCUSDT, ETHUSDT)
            buffer_size: Số lượng dữ liệu lưu trong buffer
        """
        self.symbol = symbol
        self.buffer_size = buffer_size
        
        # Buffer để lưu dữ liệu realtime
        self.price_buffer = deque(maxlen=buffer_size)
        self.ohlcv_buffer = deque(maxlen=buffer_size)
        
        # Callback functions
        self.on_new_data: Optional[Callable] = None
        
        # WebSocket connection
        self.ws = None
        self.is_running = False
        
        # Macro data cache
        self.macro_data = {
            'gold_close': None,
            'dxy_close': None,
            'last_update': None
        }
    
    def set_callback(self, callback: Callable):
        """Set callback function khi có dữ liệu mới"""
        self.on_new_data = callback
    
    async def connect_binance_websocket(self):
        """Kết nối đến Binance WebSocket để lấy giá realtime"""
        symbol_lower = self.symbol.lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@trade"
        
        try:
            self.ws = await websockets.connect(ws_url)
            print(f"Đã kết nối đến Binance WebSocket: {self.symbol}")
            
            while self.is_running:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)
                    
                    # Parse dữ liệu
                    price_data = self._parse_binance_message(data)
                    
                    if price_data:
                        self.price_buffer.append(price_data)
                        
                        # Callback nếu có
                        if self.on_new_data:
                            self.on_new_data(price_data)
                            
                except Exception as e:
                    print(f"Lỗi khi nhận dữ liệu: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"Lỗi kết nối WebSocket: {e}")
    
    def _parse_binance_message(self, data: dict) -> Optional[dict]:
        """Parse message từ Binance WebSocket"""
        try:
            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': float(data['p']),
                'quantity': float(data['q']),
                'time': int(data['T']),
                'is_buyer_maker': data['m']
            }
        except Exception as e:
            print(f"Lỗi parse message: {e}")
            return None
    
    async def fetch_macro_data(self):
        """Fetch dữ liệu macro (Gold, DXY) periodically"""
        while self.is_running:
            try:
                # Fetch Gold (GLD ETF)
                gold = yf.download("GLD", period="1d", interval="1d", progress=False)
                if not gold.empty:
                    self.macro_data['gold_close'] = gold['Close'].iloc[-1]
                
                # Fetch DXY (UUP ETF)
                dxy = yf.download("UUP", period="1d", interval="1d", progress=False)
                if not dxy.empty:
                    self.macro_data['dxy_close'] = dxy['Close'].iloc[-1]
                
                self.macro_data['last_update'] = datetime.now()
                print(f"Đã cập nhật macro data: Gold={self.macro_data['gold_close']}, DXY={self.macro_data['dxy_close']}")
                
            except Exception as e:
                print(f"Lỗi fetch macro data: {e}")
            
            # Cập nhật mỗi 1 giờ
            await asyncio.sleep(3600)
    
    def get_latest_price(self) -> Optional[float]:
        """Lấy giá mới nhất"""
        if self.price_buffer:
            return self.price_buffer[-1]['price']
        return None
    
    def get_price_history(self, n: int = 100) -> pd.DataFrame:
        """Lấy lịch sử giá n điểm gần nhất"""
        if len(self.price_buffer) < n:
            n = len(self.price_buffer)
        
        data = list(self.price_buffer)[-n:]
        return pd.DataFrame(data)
    
    def get_macro_data(self) -> Dict:
        """Lấy dữ liệu macro"""
        return self.macro_data.copy()
    
    async def start(self):
        """Bắt đầu thu thập dữ liệu"""
        self.is_running = True
        
        # Start WebSocket connection
        ws_task = asyncio.create_task(self.connect_binance_websocket())
        
        # Start macro data fetcher
        macro_task = asyncio.create_task(self.fetch_macro_data())
        
        await asyncio.gather(ws_task, macro_task)
    
    def stop(self):
        """Dừng thu thập dữ liệu"""
        self.is_running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
        print("Đã dừng Realtime Data Collector")


class OHLCVCollector:
    """Collector để thu thập dữ liệu OHLCV theo interval"""
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m", buffer_size: int = 1000):
        """
        Args:
            symbol: Symbol trading pair
            interval: Khoảng thời gian (1m, 5m, 15m, 1h)
            buffer_size: Số lượng candles lưu trong buffer
        """
        self.symbol = symbol
        self.interval = interval
        self.buffer_size = buffer_size
        self.ohlcv_buffer = deque(maxlen=buffer_size)
        self.is_running = False
        self.on_new_candle: Optional[Callable] = None
    
    def set_callback(self, callback: Callable):
        """Set callback khi có candle mới"""
        self.on_new_candle = callback
    
    async def connect_kline_websocket(self):
        """Kết nối đến Binance Kline WebSocket"""
        symbol_lower = self.symbol.lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@kline_{self.interval}"
        
        try:
            async with websockets.connect(ws_url) as ws:
                print(f"Đã kết nối đến Kline WebSocket: {self.symbol} {self.interval}")
                
                while self.is_running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        # Parse kline data
                        kline = data['k']
                        candle_data = {
                            'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                            'open': float(kline['o']),
                            'high': float(kline['h']),
                            'low': float(kline['l']),
                            'close': float(kline['c']),
                            'volume': float(kline['v']),
                            'is_closed': kline['x']  # True nếu candle đã đóng
                        }
                        
                        # Chỉ lưu khi candle đóng
                        if candle_data['is_closed']:
                            self.ohlcv_buffer.append(candle_data)
                            
                            if self.on_new_candle:
                                self.on_new_candle(candle_data)
                                
                    except Exception as e:
                        print(f"Lỗi khi nhận kline data: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"Lỗi kết nối Kline WebSocket: {e}")
    
    def get_latest_candles(self, n: int = 100) -> pd.DataFrame:
        """Lấy n candles gần nhất"""
        if len(self.ohlcv_buffer) < n:
            n = len(self.ohlcv_buffer)
        
        data = list(self.ohlcv_buffer)[-n:]
        return pd.DataFrame(data)
    
    async def start(self):
        """Bắt đầu thu thập OHLCV"""
        self.is_running = True
        await self.connect_kline_websocket()
    
    def stop(self):
        """Dừng thu thập"""
        self.is_running = False


# Synchronous wrapper cho dễ sử dụng
class SyncRealtimeCollector:
    """Synchronous wrapper cho realtime collector"""
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        self.symbol = symbol
        self.interval = interval
        self.price_collector = RealtimeDataCollector(symbol)
        self.ohlcv_collector = OHLCVCollector(symbol, interval)
        self.loop = None
        self.thread = None
    
    def start(self):
        """Bắt đầu thu thập dữ liệu trong thread riêng"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_collectors())
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        print(f"Đã bắt đầu realtime collector cho {self.symbol}")
    
    async def _start_collectors(self):
        """Start cả price và OHLCV collectors"""
        await asyncio.gather(
            self.price_collector.start(),
            self.ohlcv_collector.start()
        )
    
    def stop(self):
        """Dừng thu thập"""
        self.price_collector.stop()
        self.ohlcv_collector.stop()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    def get_latest_price(self) -> Optional[float]:
        return self.price_collector.get_latest_price()
    
    def get_latest_candles(self, n: int = 100) -> pd.DataFrame:
        return self.ohlcv_collector.get_latest_candles(n)


if __name__ == "__main__":
    # Test realtime collector
    async def test_callback(data):
        print(f"New price: {data['price']} at {data['timestamp']}")
    
    collector = RealtimeDataCollector("BTCUSDT")
    collector.set_callback(test_callback)
    
    try:
        await collector.start()
    except KeyboardInterrupt:
        collector.stop()
