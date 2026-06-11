"""
Feature Engineering Module for Bitcoin Price Prediction

This module handles the creation of technical indicators and features
from OHLCV data for Bitcoin price prediction.

Technical Indicators Included:
- RSI (Relative Strength Index)
- MA20 (20-day Moving Average)
- MA50 (50-day Moving Average)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    A class to calculate technical indicators for Bitcoin price prediction.
    
    Technical indicators are mathematical calculations based on historical
    price and volume data that help traders and analysts predict future price movements.
    """
    
    def __init__(self, data):
        """
        Initialize the TechnicalIndicators class.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing OHLCV data with columns: Date, Open, High, Low, Close, Volume
        """
        self.data = data.copy()
        
    def add_moving_averages(self, periods=[20, 50]):
        """
        Add Moving Average (MA) indicators.
        
        Moving Average smooths out price data by creating a constantly updated average price.
        It helps identify the trend direction and potential support/resistance levels.
        
        Formula: MA = (Sum of prices over n periods) / n
        
        Parameters:
        -----------
        periods : list
            List of periods for moving averages (default: [20, 50])
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added MA columns
        """
        for period in periods:
            col_name = f'MA{period}'
            self.data[col_name] = self.data['Close'].rolling(window=period).mean()
            logger.info(f"Added {col_name} indicator")
        
        return self.data
    
    def add_rsi(self, period=14):
        """
        Add Relative Strength Index (RSI) indicator.
        
        RSI is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between 0 and 100. Traditionally:
        - RSI > 70: Overbought (price may drop)
        - RSI < 30: Oversold (price may rise)
        
        Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over n periods
        
        Parameters:
        -----------
        period : int
            Period for RSI calculation (default: 14)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added RSI column
        """
        # Calculate price changes
        delta = self.data['Close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate Relative Strength (RS)
        rs = gain / loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        self.data['RSI'] = rsi
        logger.info(f"Added RSI{period} indicator")
        
        return self.data
    
    def add_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Add Moving Average Convergence Divergence (MACD) indicator.
        
        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.
        
        Components:
        - MACD Line: EMA(fast) - EMA(slow)
        - Signal Line: EMA of MACD Line
        - Histogram: MACD Line - Signal Line
        
        Interpretation:
        - MACD crosses above Signal: Bullish signal (buy)
        - MACD crosses below Signal: Bearish signal (sell)
        
        Parameters:
        -----------
        fast_period : int
            Period for fast EMA (default: 12)
        slow_period : int
            Period for slow EMA (default: 26)
        signal_period : int
            Period for signal line EMA (default: 9)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added MACD columns
        """
        # Calculate EMAs
        ema_fast = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram
        
        logger.info(f"Added MACD indicator (fast={fast_period}, slow={slow_period}, signal={signal_period})")
        
        return self.data
    
    def add_atr(self, period=14):
        """
        Add Average True Range (ATR) indicator.
        
        ATR measures market volatility by decomposing the entire range of an asset price
        for that period. It does not indicate price direction, only volatility.
        
        True Range (TR) is the greatest of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        
        ATR is the moving average of TR over n periods.
        
        Interpretation:
        - Higher ATR: Higher volatility
        - Lower ATR: Lower volatility
        
        Parameters:
        -----------
        period : int
            Period for ATR calculation (default: 14)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added ATR column
        """
        # Calculate True Range
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (simple moving average of True Range)
        atr = true_range.rolling(window=period).mean()
        
        self.data['ATR'] = atr
        logger.info(f"Added ATR{period} indicator")
        
        return self.data
    
    def add_price_changes(self, periods=[1, 5, 10]):
        """
        Add price change features (percentage change over different periods).
        
        Price changes help capture momentum and trend strength.
        
        Formula: Price Change % = ((Price_t - Price_{t-n}) / Price_{t-n}) * 100
        
        Parameters:
        -----------
        periods : list
            List of periods for price change calculation (default: [1, 5, 10])
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added price change columns
        """
        for period in periods:
            col_name = f'Price_Change_{period}d'
            self.data[col_name] = self.data['Close'].pct_change(periods=period) * 100
            logger.info(f"Added {col_name} feature")
        
        return self.data
    
    def add_volume_features(self):
        """
        Add volume-related features.
        
        Volume is an important indicator as it shows the strength of price movements.
        High volume with price increase = strong bullish signal
        High volume with price decrease = strong bearish signal
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added volume features
        """
        # Volume change percentage
        self.data['Volume_Change'] = self.data['Volume'].pct_change() * 100
        
        # Volume moving average
        self.data['Volume_MA20'] = self.data['Volume'].rolling(window=20).mean()
        
        # Volume relative to MA (volume ratio)
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA20']
        
        logger.info("Added volume features")
        
        return self.data
    
    def add_all_indicators(self):
        """
        Add all technical indicators and features to the dataset.
        
        This method adds:
        - Moving Averages (MA20, MA50)
        - RSI (14)
        - MACD (12, 26, 9)
        - ATR (14)
        - Price Changes (1d, 5d, 10d)
        - Volume Features
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all technical indicators added
        """
        logger.info("Adding all technical indicators...")
        
        self.add_moving_averages(periods=[20, 50])
        self.add_rsi(period=14)
        self.add_macd(fast_period=12, slow_period=26, signal_period=9)
        self.add_atr(period=14)
        self.add_price_changes(periods=[1, 5, 10])
        self.add_volume_features()
        
        logger.info("All technical indicators added successfully")
        
        return self.data
    
    def get_feature_columns(self):
        """
        Get the list of feature columns (excluding original OHLCV).
        
        Returns:
        --------
        list
            List of feature column names
        """
        base_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in self.data.columns if col not in base_columns]
        return feature_columns


def main():
    """
    Main function to demonstrate feature engineering.
    """
    # Load sample data (assuming data_collection.py has been run)
    try:
        data = pd.read_csv("data/raw/bitcoin_data.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Initialize technical indicators
        ti = TechnicalIndicators(data)
        
        # Add all indicators
        data_with_features = ti.add_all_indicators()
        
        # Display feature columns
        print("\nFeature columns:")
        print(ti.get_feature_columns())
        
        # Display sample data
        print("\nSample data with features (last 5 rows):")
        print(data_with_features.tail())
        
        # Save data with features
        data_with_features.to_csv("data/processed/bitcoin_data_with_features.csv", index=False)
        print("\nData with features saved to data/bitcoin_data_with_features.csv")
        
    except FileNotFoundError:
        print("Error: bitcoin_data.csv not found. Run data_collection.py first.")


if __name__ == "__main__":
    main()
