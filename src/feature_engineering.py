"""
Feature Engineering Module
Tạo các technical indicators và features cho mô hình
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Class để tạo features từ dữ liệu thô"""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các technical indicators cơ bản
        
        Args:
            df: DataFrame với các cột open, high, low, close, volume
        
        Returns:
            DataFrame với thêm các cột indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_14'] = df['close'].rolling(window=14).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        df['atr_14'] = self.calculate_atr(df, 14)
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_7'] = df['close'].pct_change(7)
        df['price_change_30'] = df['close'].pct_change(30)
        
        # Volume indicators
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_7']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Tính RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Tính Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm features từ dữ liệu macro (Gold, DXY)
        
        Args:
            df: DataFrame với các cột gold_close, dxy_close
        
        Returns:
            DataFrame với thêm macro features
        """
        df = df.copy()
        
        # Gold returns
        df['gold_return'] = df['gold_close'].pct_change()
        df['gold_return_7'] = df['gold_close'].pct_change(7)
        
        # DXY returns
        df['dxy_return'] = df['dxy_close'].pct_change()
        df['dxy_return_7'] = df['dxy_close'].pct_change(7)
        
        # Correlation features
        df['price_gold_ratio'] = df['close'] / df['gold_close']
        df['price_dxy_ratio'] = df['close'] / df['dxy_close']
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 2, 3, 7]) -> pd.DataFrame:
        """
        Thêm lag features (giá quá khứ)
        
        Args:
            df: DataFrame
            lag_periods: List of periods to lag
        
        Returns:
            DataFrame với lag features
        """
        df = df.copy()
        
        for lag in lag_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def prepare_target(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Tạo target variable cho dự đoán
        
        Args:
            df: DataFrame
            forecast_horizon: Số ngày dự đoán trước (1 = dự đoán ngày mai)
        
        Returns:
            DataFrame với cột target
        """
        df = df.copy()
        
        # Target: giá close của forecast_horizon ngày sau
        df['target'] = df['close'].shift(-forecast_horizon)
        
        # Target direction (1 = tăng, 0 = giảm)
        df['target_direction'] = (df['target'] > df['close']).astype(int)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu: xóa NaN, xử lý outliers
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        
        # Xóa các dòng có NaN ở target
        df = df.dropna(subset=['target'])
        
        # Forward fill cho các features
        df = df.ffill()
        
        # Backward fill cho các giá trị còn thiếu
        df = df.bfill()
        
        # Xóa các cột macro nếu toàn bộ là NaN
        macro_cols = ['gold_close', 'dxy_close', 'gold_return', 'gold_return_7', 
                      'dxy_return', 'dxy_return_7', 'price_gold_ratio', 'price_dxy_ratio']
        for col in macro_cols:
            if col in df.columns and df[col].isna().all():
                df = df.drop(columns=[col])
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Lấy danh sách các cột feature (loại bỏ target và các cột không cần thiết)
        
        Args:
            df: DataFrame
            exclude_cols: List of columns to exclude
        
        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'target', 'target_direction']
        
        # Chỉ lấy các cột numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ các cột trong exclude_cols
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'gold_close': np.random.randn(100).cumsum() + 180,
        'dxy_close': np.random.randn(100).cumsum() + 100
    }
    df = pd.DataFrame(data)
    
    engineer = FeatureEngineer()
    
    # Add technical indicators
    df = engineer.add_technical_indicators(df)
    print(f"Sau khi thêm technical indicators: {df.shape}")
    
    # Add macro features
    df = engineer.add_macro_features(df)
    print(f"Sau khi thêm macro features: {df.shape}")
    
    # Add lag features
    df = engineer.add_lag_features(df)
    print(f"Sau khi thêm lag features: {df.shape}")
    
    # Prepare target
    df = engineer.prepare_target(df)
    print(f"Sau khi thêm target: {df.shape}")
    
    # Clean data
    df = engineer.clean_data(df)
    print(f"Sau khi làm sạch: {df.shape}")
    
    print("\nCác cột feature:")
    feature_cols = engineer.get_feature_columns(df)
    print(feature_cols)
