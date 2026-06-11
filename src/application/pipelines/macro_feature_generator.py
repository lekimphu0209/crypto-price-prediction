"""
Macro Feature Generator - Add Gold and DXY features
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .feature_pipeline import PipelineStep


class MacroFeatureGenerator(PipelineStep):
    """Generate macro features (Gold, DXY)"""
    
    def __init__(self, gold_data: Optional[pd.DataFrame] = None, dxy_data: Optional[pd.DataFrame] = None):
        """
        Args:
            gold_data: DataFrame with Gold price data
            dxy_data: DataFrame with DXY data
        """
        self.gold_data = gold_data
        self.dxy_data = dxy_data
    
    def get_name(self) -> str:
        """Get step name"""
        return "MacroFeatureGenerator"
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add macro features to data
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with macro features
        """
        df = data.copy()
        
        # Add Gold features if available
        if self.gold_data is not None and not self.gold_data.empty:
            df = self._add_gold_features(df)
        
        # Add DXY features if available
        if self.dxy_data is not None and not self.dxy_data.empty:
            df = self._add_dxy_features(df)
        
        return df
    
    def _add_gold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Gold-related features"""
        # Align gold data with crypto data by date
        gold_aligned = self.gold_data.reindex(df.index, method='ffill')
        
        if 'Close' in gold_aligned.columns:
            df['gold_price'] = gold_aligned['Close']
            df['gold_return'] = df['gold_price'].pct_change()
            df['gold_ma7'] = df['gold_price'].rolling(7).mean()
            df['gold_price_ratio'] = df['close'] / df['gold_price']
        
        return df
    
    def _add_dxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DXY-related features"""
        # Align DXY data with crypto data by date
        dxy_aligned = self.dxy_data.reindex(df.index, method='ffill')
        
        if 'Close' in dxy_aligned.columns:
            df['dxy_price'] = dxy_aligned['Close']
            df['dxy_return'] = df['dxy_price'].pct_change()
            df['dxy_ma7'] = df['dxy_price'].rolling(7).mean()
            df['dxy_price_ratio'] = df['close'] / df['dxy_price']
        
        return df
