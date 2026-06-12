"""
XGBoost Model Implementation
"""
import joblib
import numpy as np
from typing import Dict
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.infrastructure.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model implementation"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__()
        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        self._name = "XGBoost"
        self._params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train XGBoost model
        
        Args:
            X: Feature matrix
            y: Target values
        """
        self._model.fit(X, y)
        self._is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        self._check_trained()
        return self._model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True values
        
        Returns:
            Dictionary of metrics
        """
        self._check_trained()
        
        y_pred = self.predict(X)
        
        return {
            'mae': float(mean_absolute_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'r2': float(r2_score(y, y_pred))
        }
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: File path to save model
        """
        self._check_trained()
        
        joblib.dump({
            'model': self._model,
            'is_trained': self._is_trained,
            'params': self._params
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load model from file
        
        Args:
            path: File path to load model from
        """
        data = joblib.load(path)
        self._model = data['model']
        self._is_trained = data.get('is_trained', True)
        self._params = data.get('params', {})
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary of feature importance
        """
        self._check_trained()
        
        importance = self._model.feature_importances_
        return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
