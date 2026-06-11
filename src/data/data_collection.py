"""
Module thu thập dữ liệu cho dự đoán giá Bitcoin

Module này xử lý việc lấy dữ liệu giá lịch sử Bitcoin từ Yahoo Finance
sử dụng thư viện yfinance.

Author: Dự án Dự đoán Giá Bitcoin
Date: 2026
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Cấu hình logging - Ghi log để theo dõi quá trình chạy chương trình
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BitcoinDataCollector:
    """
    Lớp thu thập dữ liệu giá lịch sử Bitcoin từ Yahoo Finance.
    
    Lớp này cung cấp các phương thức để:
    - Lấy dữ liệu OHLCV lịch sử cho Bitcoin (BTC-USD)
    - Lưu dữ liệu vào file CSV
    - Tải dữ liệu từ file CSV
    
    OHLCV = Open (giá mở), High (giá cao), Low (giá thấp), Close (giá đóng), Volume (khối lượng)
    """
    
    def __init__(self, ticker="BTC-USD"):
        """
        Khởi tạo BitcoinDataCollector.
        
        Parameters:
        -----------
        ticker : str
            Mã chứng khoán của Bitcoin trên Yahoo Finance (mặc định: "BTC-USD")
        """
        self.ticker = ticker  # Mã chứng khoán Bitcoin
        self.data = None      # Dữ liệu sẽ được lưu ở đây
        
    def fetch_data(self, period="5y", interval="1d"):
        """
        Lấy dữ liệu giá lịch sử Bitcoin từ Yahoo Finance.
        
        Parameters:
        -----------
        period : str
            Khoảng thời gian để lấy dữ liệu.
            Options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
            Mặc định: "5y" (5 năm dữ liệu)
        
        interval : str
            Khoảng thời gian của dữ liệu (timeframe).
            Options: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
            Mặc định: "1d" (dữ liệu hàng ngày)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu OHLCV (Open, High, Low, Close, Volume)
        """
        logger.info(f"Fetching {self.ticker} data for period: {period}, interval: {interval}")
        
        try:
            # Download data from Yahoo Finance
            self.data = yf.download(self.ticker, period=period, interval=interval)
            
            # Reset index to make Date a column
            self.data = self.data.reset_index()
            
            # Check if Adj Close exists, if so remove it
            if 'Adj Close' in self.data.columns:
                self.data = self.data.drop(columns=['Adj Close'])
            
            # Rename columns to be more descriptive
            self.data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            logger.info(f"Successfully fetched {len(self.data)} data points")
            logger.info(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_data_by_date_range(self, start_date, end_date, interval="1d"):
        """
        Fetch Bitcoin data for a specific date range.
        
        Parameters:
        -----------
        start_date : str
            Start date in format "YYYY-MM-DD"
        end_date : str
            End date in format "YYYY-MM-DD"
        interval : str
            Data interval (default: "1d")
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing OHLCV data
        """
        logger.info(f"Fetching {self.ticker} data from {start_date} to {end_date}")
        
        try:
            self.data = yf.download(self.ticker, start=start_date, end=end_date, interval=interval)
            self.data = self.data.reset_index()
            
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
                
            if 'Adj Close' in self.data.columns:
                self.data = self.data.drop(columns=['Adj Close'])
            
            logger.info(f"Successfully fetched {len(self.data)} data points")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def save_to_csv(self, filepath):
        """
        Save the fetched data to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path where the CSV file will be saved
        """
        if self.data is None:
            raise ValueError("No data to save. Fetch data first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def load_from_csv(self, filepath):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file to load
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the loaded data
        """
        self.data = pd.read_csv(filepath)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        logger.info(f"Data loaded from {filepath}")
        return self.data
    
    def get_data_summary(self):
        """
        Get a summary of the fetched data.
        
        Returns:
        --------
        dict
            Dictionary containing data summary statistics
        """
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
        
        summary = {
            'total_records': len(self.data),
            'date_range': (self.data['Date'].min(), self.data['Date'].max()),
            'price_range': (self.data['Close'].min(), self.data['Close'].max()),
            'avg_volume': self.data['Volume'].mean(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return summary


def main():
    """
    Main function to demonstrate data collection.
    """
    # Initialize collector
    collector = BitcoinDataCollector(ticker="BTC-USD")
    
    # Fetch 5 years of daily data
    data = collector.fetch_data(period="5y", interval="1d")
    
    # Display first few rows
    print("\nFirst 5 rows of data:")
    print(data.head())
    
    # Display data summary
    print("\nData Summary:")
    summary = collector.get_data_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save data
    collector.save_to_csv("data/raw/bitcoin_data.csv")
    
    return data


if __name__ == "__main__":
    main()
