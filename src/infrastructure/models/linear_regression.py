"""
Linear Regression Model Implementation
"""
import joblib
import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression as SKLearnLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.infrastructure.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model implementation"""
    
    def __init__(self):
        super().__init__()
        self._model = SKLearnLR()
        self._scaler = StandardScaler()
        self._name = "LinearRegression"
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train Linear Regression model
        
        Args:
            X: Feature matrix
            y: Target values
        """
        # Scale features
        X_scaled = self._scaler.fit_transform(X)
        
        # Train model
        self._model.fit(X_scaled, y)
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
        
        # Scale features
        X_scaled = self._scaler.transform(X)
        
        # Predict
        return self._model.predict(X_scaled)
    
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
            'scaler': self._scaler,
            'is_trained': self._is_trained
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load model from file
        
        Args:
            path: File path to load model from
        """
        data = joblib.load(path)
        self._model = data['model']
        self._scaler = data['scaler']
        self._is_trained = data.get('is_trained', True)
