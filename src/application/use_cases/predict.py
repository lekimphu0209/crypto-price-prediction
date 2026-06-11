"""
Predict Use Case - Use case for making predictions
"""
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd

from src.domain.interfaces.model import IModel
from src.domain.interfaces.data_repository import IDataRepository
from src.domain.value_objects.symbol import Symbol
from src.domain.value_objects.interval import Interval
from src.application.dtos.prediction_dto import PredictionDTO


class PredictUseCase:
    """Use case for making predictions"""
    
    def __init__(
        self,
        models: List[IModel],
        data_repository: IDataRepository
    ):
        """
        Args:
            models: List of trained models
            data_repository: Data repository
        """
        self.models = models
        self.data_repository = data_repository
    
    def execute(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> PredictionDTO:
        """
        Execute prediction use case
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Number of data points to use
        
        Returns:
            PredictionDTO with prediction results
        """
        # Convert to value objects
        symbol_vo = Symbol(symbol)
        interval_vo = Interval(interval)
        
        # Get latest data
        ohlcv_data = self.data_repository.get_latest_data(symbol_vo, interval_vo, limit)
        
        if len(ohlcv_data) < 30:
            raise ValueError(f"Not enough data points (need 30, got {len(ohlcv_data)})")
        
        # Convert to DataFrame for feature engineering
        df = pd.DataFrame([{
            'timestamp': o.timestamp,
            'open': o.open,
            'high': o.high,
            'low': o.low,
            'close': o.close,
            'volume': o.volume
        } for o in ohlcv_data])
        
        # Prepare features (simplified - in real app, use FeatureEngineer)
        X = self._prepare_features(df)
        
        # Get current price
        current_price = ohlcv_data[-1].close
        
        # Predict with each model
        predictions_by_model = {}
        for model in self.models:
            if model.is_trained:
                try:
                    pred = model.predict(X)[-1]  # Get last prediction
                    predictions_by_model[model.name] = pred
                except Exception as e:
                    print(f"Error predicting with {model.name}: {e}")
        
        if not predictions_by_model:
            raise RuntimeError("No trained models available for prediction")
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(predictions_by_model.values()))
        
        # Calculate confidence (based on model agreement)
        confidence = self._calculate_confidence(predictions_by_model)
        
        return PredictionDTO(
            symbol=symbol,
            current_price=current_price,
            predicted_price=float(ensemble_pred),
            confidence=confidence,
            model_name="ensemble",
            timestamp=datetime.now(),
            predictions_by_model=predictions_by_model
        )
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Feature matrix X
        """
        # Simplified feature engineering
        # In real application, use comprehensive feature engineering
        
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Moving averages
        df['ma_7'] = df['close'].rolling(7).mean()
        df['ma_14'] = df['close'].rolling(14).mean()
        
        # Price relative to MA
        df['price_to_ma7'] = df['close'] / df['ma_7']
        df['price_to_ma14'] = df['close'] / df['ma_14']
        
        # Drop NaN
        df = df.dropna()
        
        # Select feature columns
        feature_cols = ['returns', 'volume_change', 'price_to_ma7', 'price_to_ma14']
        X = df[feature_cols].values
        
        return X
    
    def _calculate_confidence(self, predictions: dict) -> float:
        """
        Calculate confidence based on model agreement
        
        Args:
            predictions: Dict of model predictions
        
        Returns:
            Confidence score (0-1)
        """
        if len(predictions) < 2:
            return 0.5
        
        values = list(predictions.values())
        std = np.std(values)
        mean = np.mean(values)
        
        # Lower std = higher confidence
        # Normalize to 0-1 range
        cv = std / (abs(mean) + 1e-8)  # Coefficient of variation
        confidence = max(0, min(1, 1 - cv))
        
        return confidence
