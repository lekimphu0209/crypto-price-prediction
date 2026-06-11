"""
Data Collector Module
Thu thập dữ liệu giá crypto và macro economic data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Tuple, Optional


class DataCollector:
    """Class để thu thập dữ liệu từ các nguồn khác nhau"""
    
    def __init__(self):
        self.binance_base_url = "https://api.binance.com/api/v3"
    
    def fetch_binance_ohlcv(self, symbol: str = "BTCUSDT", interval: str = "1d", 
                           days: int = 365) -> pd.DataFrame:
        """
        Thu thập dữ liệu OHLCV từ Binance
        
        Args:
            symbol: Symbol trading pair (e.g., BTCUSDT, ETHUSDT)
            interval: Khoảng thời gian (1m, 5m, 15m, 1h, 4h, 1d)
            days: Số ngày dữ liệu cần thu thập
        
        Returns:
            DataFrame với các cột: timestamp, open, high, low, close, volume
        """
        try:
            # Sử dụng yfinance thay vì Binance API trực tiếp (đơn giản hơn)
            # Chuyển symbol sang format yfinance
            yf_symbol = symbol.replace("USDT", "-USD")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"Đang thu thập dữ liệu {symbol} từ {start_date.date()} đến {end_date.date()}...")
            
            df = yf.download(yf_symbol, start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"Cảnh báo: Không thể thu thập dữ liệu cho {symbol}")
                return pd.DataFrame()
            
            # Đổi tên cột để nhất quán
            df.columns = [col.lower() for col in df.columns]
            df.reset_index(inplace=True)
            df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'}, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Lỗi khi thu thập dữ liệu từ Binance: {e}")
            return pd.DataFrame()
    
    def fetch_macro_data(self, days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Thu thập dữ liệu Gold (SPDR Gold Shares) và DXY (US Dollar Index)
        
        Returns:
            Tuple (gold_df, dxy_df)
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"Đang thu thập dữ liệu Gold và DXY từ {start_date.date()} đến {end_date.date()}...")
            
            # Gold - sử dụng GLD ETF
            gold_df = yf.download("GLD", start=start_date, end=end_date, interval="1d")
            gold_df.columns = [col.lower() for col in gold_df.columns]
            gold_df.reset_index(inplace=True)
            gold_df.rename(columns={'date': 'timestamp'}, inplace=True)
            
            # DXY - sử dụng UUP ETF (proxy cho DXY)
            dxy_df = yf.download("UUP", start=start_date, end=end_date, interval="1d")
            dxy_df.columns = [col.lower() for col in dxy_df.columns]
            dxy_df.reset_index(inplace=True)
            dxy_df.rename(columns={'date': 'timestamp'}, inplace=True)
            
            return gold_df, dxy_df
            
        except Exception as e:
            print(f"Lỗi khi thu thập dữ liệu macro: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def merge_data(self, crypto_df: pd.DataFrame, gold_df: pd.DataFrame, 
                   dxy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gộp tất cả dữ liệu lại với nhau dựa trên timestamp
        
        Returns:
            DataFrame đã merge
        """
        try:
            # Đảm bảo timestamp là datetime
            crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
            gold_df['timestamp'] = pd.to_datetime(gold_df['timestamp'])
            dxy_df['timestamp'] = pd.to_datetime(dxy_df['timestamp'])
            
            # Merge crypto với gold
            merged = pd.merge(crypto_df, gold_df[['timestamp', 'close']], 
                            on='timestamp', how='left', suffixes=('', '_gold'))
            merged.rename(columns={'close_gold': 'gold_close'}, inplace=True)
            
            # Merge với dxy
            merged = pd.merge(merged, dxy_df[['timestamp', 'close']], 
                            on='timestamp', how='left', suffixes=('', '_dxy'))
            merged.rename(columns={'close_dxy': 'dxy_close'}, inplace=True)
            
            # Forward fill cho các giá trị thiếu
            merged['gold_close'].fillna(method='ffill', inplace=True)
            merged['dxy_close'].fillna(method='ffill', inplace=True)
            
            return merged
            
        except Exception as e:
            print(f"Lỗi khi merge dữ liệu: {e}")
            return crypto_df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Lưu dữ liệu ra file CSV"""
        try:
            df.to_csv(f"data/{filename}", index=False)
            print(f"Đã lưu dữ liệu vào data/{filename}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu: {e}")


if __name__ == "__main__":
    # Test data collection
    collector = DataCollector()
    
    # Thu thập dữ liệu BTC
    btc_df = collector.fetch_binance_ohlcv("BTCUSDT", "1d", 365)
    print(f"BTC Data shape: {btc_df.shape}")
    print(btc_df.head())
    
    # Thu thập dữ liệu ETH
    eth_df = collector.fetch_binance_ohlcv("ETHUSDT", "1d", 365)
    print(f"ETH Data shape: {eth_df.shape}")
    
    # Thu thập dữ liệu macro
    gold_df, dxy_df = collector.fetch_macro_data(365)
    print(f"Gold Data shape: {gold_df.shape}")
    print(f"DXY Data shape: {dxy_df.shape}")
    
    # Merge dữ liệu
    merged_btc = collector.merge_data(btc_df, gold_df, dxy_df)
    print(f"Merged BTC Data shape: {merged_btc.shape}")
    
    # Lưu dữ liệu
    collector.save_data(merged_btc, "btc_complete_data.csv")
    
    merged_eth = collector.merge_data(eth_df, gold_df, dxy_df)
    collector.save_data(merged_eth, "eth_complete_data.csv")
