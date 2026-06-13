"""
Computational Performance Evaluator

Evaluates model computational efficiency and resource usage.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ComputationalMetrics:
    """Container for computational performance metrics."""
    training_time: float  # in seconds
    prediction_time: float  # in seconds
    num_parameters: int
    memory_usage_mb: float
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'Training Time (s)': self.training_time,
            'Prediction Time (s)': self.prediction_time,
            'Num Parameters': self.num_parameters,
            'Memory Usage (MB)': self.memory_usage_mb,
            'Model Size (MB)': self.model_size_mb
        }


class ComputationalEvaluator:
    """Evaluates computational performance of models."""
    
    def __init__(self):
        self.metrics_history = []
        self.process = psutil.Process()
    
    def measure_training_time(self, model, X_train: np.ndarray, 
                            y_train: np.ndarray) -> float:
        """
        Measure model training time.
        
        Args:
            model: Model object with fit() method
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training time in seconds
        """
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        return end_time - start_time
    
    def measure_prediction_time(self, model, X_test: np.ndarray,
                               iterations: int = 100) -> float:
        """
        Measure model prediction time (averaged over iterations).
        
        Args:
            model: Model object with predict() method
            X_test: Test features
            iterations: Number of iterations to average
            
        Returns:
            Average prediction time in seconds
        """
        # Warm-up
        model.predict(X_test[:1])
        
        # Measure
        times = []
        for _ in range(iterations):
            start_time = time.time()
            model.predict(X_test)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def count_parameters(self, model: Any) -> int:
        """
        Count number of trainable parameters in a model.
        
        Args:
            model: Model object
            
        Returns:
            Number of parameters
        """
        # For sklearn models
        if hasattr(model, 'coef_'):
            params = len(model.coef_.flatten()) + len(model.intercept_ if hasattr(model, 'intercept_') else [])
            if hasattr(model, 'n_estimators'):
                params *= model.n_estimators
            return params
        
        # For Keras/TensorFlow models (including RNN, LSTM, BiLSTM, Transformer)
        if hasattr(model, 'count_params'):
            return model.count_params()
        
        # For XGBoost (legacy - replaced with RNN)
        if hasattr(model, 'get_num_boosting_rounds'):
            return model.get_num_boosting_rounds() * 100  # Approximate
        
        return 0
    
    def measure_memory_usage(self, model: Any) -> float:
        """
        Measure memory usage of a model.
        
        Args:
            model: Model object
            
        Returns:
            Memory usage in MB
        """
        # Get current process memory
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    
    def estimate_model_size(self, model: Any) -> float:
        """
        Estimate model size in MB.
        
        Args:
            model: Model object
            
        Returns:
            Model size in MB
        """
        # For sklearn models
        if hasattr(model, '__getstate__'):
            import pickle
            size = len(pickle.dumps(model))
            return size / (1024 * 1024)
        
        # For Keras models
        if hasattr(model, 'save_weights'):
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False) as f:
                model.save_weights(f.name)
                size = os.path.getsize(f.name)
                os.unlink(f.name)
                return size / (1024 * 1024)
        
        return 0.0
    
    def evaluate(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, model_name: str = "Model") -> ComputationalMetrics:
        """
        Evaluate computational performance of a model.
        
        Args:
            model: Model object
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            model_name: Name of the model being evaluated
            
        Returns:
            ComputationalMetrics object
        """
        # Measure training time
        training_time = self.measure_training_time(model, X_train, y_train)
        
        # Measure prediction time
        prediction_time = self.measure_prediction_time(model, X_test)
        
        # Count parameters
        num_parameters = self.count_parameters(model)
        
        # Measure memory usage
        memory_usage = self.measure_memory_usage(model)
        
        # Estimate model size
        model_size = self.estimate_model_size(model)
        
        metrics = ComputationalMetrics(
            training_time=training_time,
            prediction_time=prediction_time,
            num_parameters=num_parameters,
            memory_usage_mb=memory_usage,
            model_size_mb=model_size
        )
        
        self.metrics_history.append({
            'model': model_name,
            **metrics.to_dict()
        })
        
        return metrics
    
    def compare_models(self, results: Dict[str, ComputationalMetrics]) -> pd.DataFrame:
        """
        Compare computational performance of multiple models.
        
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
        
        # Sort by Training Time (faster is better)
        df = df.sort_values('Training Time (s)', ascending=True)
        
        return df
    
    def print_report(self, metrics: ComputationalMetrics, model_name: str = "Model"):
        """Print a formatted report of computational metrics."""
        print(f"\n{'='*50}")
        print(f"Computational Performance Report: {model_name}")
        print(f"{'='*50}")
        print(f"Training Time:     {metrics.training_time:.2f}s")
        print(f"Prediction Time:   {metrics.prediction_time:.4f}s")
        print(f"Num Parameters:    {metrics.num_parameters:,}")
        print(f"Memory Usage:      {metrics.memory_usage_mb:.2f} MB")
        print(f"Model Size:        {metrics.model_size_mb:.2f} MB")
        print(f"{'='*50}\n")
