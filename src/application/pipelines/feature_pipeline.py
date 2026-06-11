"""
Feature Pipeline - Pipeline for feature engineering
"""
from abc import ABC, abstractmethod
from typing import List, Any, Optional
import pandas as pd
import numpy as np


class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get step name"""
        pass


class MissingValueHandler(PipelineStep):
    """Handle missing values"""
    
    def __init__(self, strategy: str = "ffill"):
        """
        Args:
            strategy: Strategy for handling missing values ('ffill', 'bfill', 'drop', 'mean')
        """
        self.strategy = strategy
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        df = data.copy()
        
        if self.strategy == "ffill":
            df = df.ffill()
        elif self.strategy == "bfill":
            df = df.bfill()
        elif self.strategy == "drop":
            df = df.dropna()
        elif self.strategy == "mean":
            df = df.fillna(df.mean())
        
        return df
    
    def get_name(self) -> str:
        return f"MissingValueHandler({self.strategy})"


class OutlierHandler(PipelineStep):
    """Handle outliers using IQR method"""
    
    def __init__(self, columns: List[str] = None, multiplier: float = 1.5):
        """
        Args:
            columns: Columns to check for outliers
            multiplier: IQR multiplier
        """
        self.columns = columns
        self.multiplier = multiplier
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers"""
        df = data.copy()
        
        if self.columns is None:
            # Use numeric columns only
            self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.multiplier * IQR
            upper_bound = Q3 + self.multiplier * IQR
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def get_name(self) -> str:
        return f"OutlierHandler(multiplier={self.multiplier})"


class Normalizer(PipelineStep):
    """Normalize features"""
    
    def __init__(self, method: str = "standard"):
        """
        Args:
            method: Normalization method ('standard', 'minmax')
        """
        self.method = method
        self.scaler = None
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data"""
        df = data.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == "standard":
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        elif self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def get_name(self) -> str:
        return f"Normalizer({self.method})"


class TechnicalFeatureGenerator(PipelineStep):
    """Generate technical indicator features"""
    
    def __init__(self):
        pass
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators"""
        df = data.copy()
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_14'] = df['close'].rolling(window=14).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        return df
    
    def get_name(self) -> str:
        return "TechnicalFeatureGenerator"


class FeaturePipeline:
    """Pipeline for feature engineering"""
    
    def __init__(self, steps: List[PipelineStep]):
        """
        Args:
            steps: List of pipeline steps
        """
        self.steps = steps
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the pipeline
        
        Args:
            data: Input DataFrame
        
        Returns:
            Processed DataFrame
        """
        result = data.copy()
        
        for step in self.steps:
            print(f"Executing: {step.get_name()}")
            result = step.process(result)
        
        return result
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline"""
        self.steps.append(step)
    
    def get_step_names(self) -> List[str]:
        """Get list of step names"""
        return [step.get_name() for step in self.steps]


def create_default_pipeline(macro_data: Optional[dict] = None) -> FeaturePipeline:
    """Create default feature pipeline
    
    Args:
        macro_data: Dictionary with 'gold' and 'dxy' DataFrames
    
    Returns:
        FeaturePipeline instance
    """
    steps = [
        MissingValueHandler(strategy="ffill"),
        OutlierHandler(multiplier=1.5),
        TechnicalFeatureGenerator(),
        Normalizer(method="standard")
    ]
    
    # Add macro feature generator if data available
    if macro_data:
        from .macro_feature_generator import MacroFeatureGenerator
        macro_generator = MacroFeatureGenerator(
            gold_data=macro_data.get('gold'),
            dxy_data=macro_data.get('dxy')
        )
        steps.insert(3, macro_generator)  # Insert before normalizer
    
    return FeaturePipeline(steps)
