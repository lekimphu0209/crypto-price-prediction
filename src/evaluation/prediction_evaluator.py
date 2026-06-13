"""
Prediction Performance Evaluator

Evaluates model prediction accuracy using multiple metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PredictionMetrics:
    """Container for prediction performance metrics."""
    mae: float
    rmse: float
    mape: float
    r2_score: float
    direction_accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MAE': self.mae,
            'RMSE': self.rmse,
            'MAPE': self.mape,
            'R² Score': self.r2_score,
            'Direction Accuracy': self.direction_accuracy
        }


class PredictionEvaluator:
    """Evaluates prediction performance of models."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate direction accuracy.
        Important for trading - whether model predicts up/down correctly.
        """
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction) * 100
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 model_name: str = "Model") -> PredictionMetrics:
        """
        Evaluate prediction performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            PredictionMetrics object with all metrics
        """
        metrics = PredictionMetrics(
            mae=self.calculate_mae(y_true, y_pred),
            rmse=self.calculate_rmse(y_true, y_pred),
            mape=self.calculate_mape(y_true, y_pred),
            r2_score=self.calculate_r2_score(y_true, y_pred),
            direction_accuracy=self.calculate_direction_accuracy(y_true, y_pred)
        )
        
        self.metrics_history.append({
            'model': model_name,
            **metrics.to_dict()
        })
        
        return metrics
    
    def compare_models(self, results: Dict[str, PredictionMetrics]) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison table
        """
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                **metrics.to_dict()
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by Direction Accuracy (most important for trading)
        df = df.sort_values('Direction Accuracy', ascending=False)
        
        return df
    
    def get_best_model(self, results: Dict[str, PredictionMetrics], 
                      metric: str = 'direction_accuracy') -> str:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            results: Dictionary mapping model names to their metrics
            metric: Metric to use for comparison (default: direction_accuracy)
            
        Returns:
            Name of the best performing model
        """
        best_model = None
        best_value = -np.inf if metric != 'mae' and metric != 'rmse' and metric != 'mape' else np.inf
        
        for model_name, metrics in results.items():
            value = getattr(metrics, metric)
            
            if metric in ['mae', 'rmse', 'mape']:
                if value < best_value:
                    best_value = value
                    best_model = model_name
            else:
                if value > best_value:
                    best_value = value
                    best_model = model_name
        
        return best_model
    
    def print_report(self, metrics: PredictionMetrics, model_name: str = "Model"):
        """Print a formatted report of prediction metrics."""
        print(f"\n{'='*50}")
        print(f"Prediction Performance Report: {model_name}")
        print(f"{'='*50}")
        print(f"MAE:                {metrics.mae:.2f}")
        print(f"RMSE:               {metrics.rmse:.2f}")
        print(f"MAPE:               {metrics.mape:.2f}%")
        print(f"R² Score:           {metrics.r2_score:.4f}")
        print(f"Direction Accuracy: {metrics.direction_accuracy:.2f}% ⭐")
        print(f"{'='*50}\n")
