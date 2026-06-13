"""
Feature Engineering Module for Financial Time Series
Based on Chapter 5 of Pythonic Quant
"""
import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """Feature engineering for financial time series"""
    
    def __init__(self):
        self.features = []
    
    def add_moving_average(self, df: pd.DataFrame, price_col: str, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Add moving averages"""
        for window in windows:
            df[f'MA_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df[price_col] / df[f'MA_{window}']
        return df
    
    def add_rsi(self, df: pd.DataFrame, price_col: str, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    def add_macd(self, df: pd.DataFrame, price_col: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        exp_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        exp_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp_fast - exp_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, price_col: str, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df['BB_middle'] = df[price_col].rolling(window=window).mean()
        df['BB_upper'] = df['BB_middle'] + (df[price_col].rolling(window=window).std() * num_std)
        df['BB_lower'] = df['BB_middle'] - (df[price_col].rolling(window=window).std() * num_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df[price_col] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df
    
    def add_volatility(self, df: pd.DataFrame, price_col: str, window: int = 20) -> pd.DataFrame:
        """Add volatility indicators"""
        df['returns'] = df[price_col].pct_change()
        df['volatility'] = df['returns'].rolling(window=window).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=window*2).mean()
        return df
    
    def add_lag_features(self, df: pd.DataFrame, price_col: str, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lag features"""
        for lag in lags:
            df[f'lag_{lag}'] = df[price_col].shift(lag)
            df[f'return_lag_{lag}'] = df[price_col].pct_change(lag)
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_change'] = df['volume'].pct_change()
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features"""
        if sentiment_data is not None and len(sentiment_data) > 0:
            df = df.join(sentiment_data, how='left')
            df['sentiment_ma'] = df['sentiment'].rolling(window=24).mean()
        return df
    
    def add_all_features(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Add all features at once"""
        df = self.add_moving_average(df, price_col)
        df = self.add_rsi(df, price_col)
        df = self.add_macd(df, price_col)
        df = self.add_bollinger_bands(df, price_col)
        df = self.add_volatility(df, price_col)
        df = self.add_lag_features(df, price_col)
        df = self.add_volume_features(df)
        df = self.add_time_features(df)
        return df
