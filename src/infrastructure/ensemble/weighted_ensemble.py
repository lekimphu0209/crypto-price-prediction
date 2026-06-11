"""
Weighted Ensemble Implementation
"""
import numpy as np
from typing import Dict, List
from src.domain.interfaces.model import IModel


class WeightedEnsemble:
    """Weighted average ensemble of multiple models"""
    
    def __init__(self, models: List[IModel], weights: List[float] = None):
        """
        Args:
            models: List of trained models
            weights: List of weights for each model (default: equal weights)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction
        
        Args:
            X: Feature matrix
        
        Returns:
            Weighted average predictions
        """
        predictions = []
        
        for model in self.models:
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise RuntimeError("No trained models available for ensemble")
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Make ensemble prediction with confidence score
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, confidence_score)
        """
        predictions = []
        
        for model in self.models:
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise RuntimeError("No trained models available for ensemble")
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        # Calculate confidence based on model agreement
        pred_array = np.array(predictions)
        std = np.std(pred_array, axis=0)
        mean = np.mean(pred_array, axis=0)
        
        # Normalize confidence (lower std = higher confidence)
        cv = std / (np.abs(mean) + 1e-8)
        confidence = 1.0 / (1.0 + cv)
        
        return ensemble_pred, confidence
