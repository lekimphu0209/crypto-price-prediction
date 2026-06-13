"""
Comprehensive Model Evaluator

Combines prediction, trading, and computational performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .prediction_evaluator import PredictionEvaluator, PredictionMetrics
from .trading_evaluator import TradingEvaluator, TradingMetrics
from .computational_evaluator import ComputationalEvaluator, ComputationalMetrics


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that combines all three evaluation types.
    Provides a complete assessment of model performance.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize comprehensive evaluator.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.prediction_evaluator = PredictionEvaluator()
        self.trading_evaluator = TradingEvaluator(risk_free_rate)
        self.computational_evaluator = ComputationalEvaluator()
        self.results = {}
    
    def evaluate_model(self, 
                      model: Any,
                      model_name: str,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      initial_capital: float = 10000,
                      equity_curve: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a model.
        
        Args:
            model: Model object to evaluate
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            initial_capital: Initial capital for trading evaluation
            equity_curve: Optional equity curve for trading evaluation
            
        Returns:
            Dictionary with all evaluation results
        """
        # 1. Prediction Performance
        y_pred = model.predict(X_test)
        prediction_metrics = self.prediction_evaluator.evaluate(
            y_test, y_pred, model_name
        )
        
        # 2. Computational Performance
        computational_metrics = self.computational_evaluator.evaluate(
            model, X_train, y_train, X_test, model_name
        )
        
        # 3. Trading Performance (if equity curve provided)
        trading_metrics = None
        if equity_curve is not None:
            trading_metrics = self.trading_evaluator.evaluate_from_equity_curve(
                equity_curve, initial_capital
            )
        
        # Store results
        self.results[model_name] = {
            'prediction': prediction_metrics,
            'computational': computational_metrics,
            'trading': trading_metrics
        }
        
        return self.results[model_name]
    
    def compare_all_models(self) -> pd.DataFrame:
        """
        Generate comparison table for all evaluated models.
        
        Returns:
            DataFrame with comprehensive comparison
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                # Prediction metrics
                'MAE': metrics['prediction'].mae,
                'RMSE': metrics['prediction'].rmse,
                'MAPE (%)': metrics['prediction'].mape,
                'R² Score': metrics['prediction'].r2_score,
                'Direction Acc (%)': metrics['prediction'].direction_accuracy,
                # Computational metrics
                'Train Time (s)': metrics['computational'].training_time,
                'Predict Time (s)': metrics['computational'].prediction_time,
                'Parameters': metrics['computational'].num_parameters,
            }
            
            # Add trading metrics if available
            if metrics['trading'] is not None:
                row['Return (%)'] = metrics['trading'].total_return
                row['Win Rate (%)'] = metrics['trading'].win_rate
                row['Sharpe Ratio'] = metrics['trading'].sharpe_ratio
                row['Max Drawdown (%)'] = metrics['trading'].max_drawdown
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by Direction Accuracy (most important for trading)
        df = df.sort_values('Direction Acc (%)', ascending=False)
        
        return df
    
    def generate_report(self, model_name: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name: Specific model to report on, or None for all models
            
        Returns:
            Formatted report string
        """
        if model_name is None:
            # Generate comparison report
            report = "\n" + "="*80 + "\n"
            report += "COMPREHENSIVE MODEL EVALUATION REPORT\n"
            report += "="*80 + "\n\n"
            
            report += "MODEL COMPARISON TABLE\n"
            report += "-"*80 + "\n"
            report += self.compare_all_models().to_string(index=False)
            report += "\n\n"
            
            report += "="*80 + "\n"
            return report
        else:
            # Generate individual model report
            if model_name not in self.results:
                return f"Model {model_name} not found in results."
            
            metrics = self.results[model_name]
            report = f"\n{'='*50}\n"
            report += f"COMPREHENSIVE REPORT: {model_name}\n"
            report += f"{'='*50}\n"
            
            # Prediction section
            report += "\n📊 PREDICTION PERFORMANCE\n"
            report += "-"*50 + "\n"
            self.prediction_evaluator.print_report(metrics['prediction'], model_name)
            
            # Computational section
            report += "\n💻 COMPUTATIONAL PERFORMANCE\n"
            report += "-"*50 + "\n"
            self.computational_evaluator.print_report(metrics['computational'], model_name)
            
            # Trading section
            if metrics['trading'] is not None:
                report += "\n💰 TRADING PERFORMANCE\n"
                report += "-"*50 + "\n"
                self.trading_evaluator.print_report(metrics['trading'], model_name)
            
            return report
    
    def export_to_csv(self, filename: str = "model_evaluation_results.csv"):
        """
        Export evaluation results to CSV file.
        
        Args:
            filename: Output filename
        """
        df = self.compare_all_models()
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def get_best_model_by_metric(self, metric: str) -> str:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric: Metric name (e.g., 'Direction Acc (%)', 'Return (%)', 'Train Time (s)')
            
        Returns:
            Name of the best performing model
        """
        df = self.compare_all_models()
        
        if metric in ['MAE', 'RMSE', 'MAPE (%)', 'Train Time (s)', 'Predict Time (s)', 'Max Drawdown (%)']:
            # Lower is better
            best_idx = df[metric].idxmin()
        else:
            # Higher is better
            best_idx = df[metric].idxmax()
        
        return df.loc[best_idx, 'Model']
