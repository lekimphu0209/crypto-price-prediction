"""
Evaluation Module for Bitcoin Price Prediction Models

This module provides functions to evaluate regression models using
various metrics and create comparison tables.

Metrics Explained:
1. MSE (Mean Squared Error): Average of squared differences
   Formula: MSE = (1/n) * Σ(y_i - ŷ_i)²
   - Penalizes large errors more heavily
   - Sensitive to outliers
   - Units: squared (e.g., USD²)

2. RMSE (Root Mean Squared Error): Square root of MSE
   Formula: RMSE = √MSE
   - Same unit as target (e.g., USD)
   - More interpretable than MSE
   - Still sensitive to outliers

3. MAE (Mean Absolute Error): Average of absolute differences
   Formula: MAE = (1/n) * Σ|y_i - ŷ_i|
   - Less sensitive to outliers
   - Easier to interpret
   - Same unit as target (e.g., USD)

4. MAPE (Mean Absolute Percentage Error): Average percentage error
   Formula: MAPE = (1/n) * Σ|y_i - ŷ_i| / y_i * 100
   - Scale-independent
   - Expressed as percentage
   - Not suitable when y_i = 0

5. R² (R-squared): Coefficient of determination
   Formula: R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
   - Measures proportion of variance explained
   - Range: 0 to 1 (higher is better)
   - Can be negative for poor models

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    A class to evaluate regression models for Bitcoin price prediction.
    
    This class provides methods to calculate various evaluation metrics
    and create comparison tables between different models.
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        self.metrics_history = {}
        
    def calculate_mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE).
        
        MSE measures the average squared difference between predicted
        and actual values. It penalizes large errors more heavily.
        
        Formula: MSE = (1/n) * Σ(y_i - ŷ_i)²
        
        Interpretation:
        - Lower MSE is better
        - MSE = 0 means perfect prediction
        - Sensitive to outliers
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
            
        Returns:
        --------
        float
            MSE value
        """
        mse = mean_squared_error(y_true, y_pred)
        return mse
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error (RMSE).
        
        RMSE is the square root of MSE, making it more interpretable
        as it's in the same units as the target variable.
        
        Formula: RMSE = √MSE = √[(1/n) * Σ(y_i - ŷ_i)²]
        
        Interpretation:
        - Lower RMSE is better
        - Same unit as target (e.g., USD)
        - Still sensitive to outliers
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
            
        Returns:
        --------
        float
            RMSE value
        """
        mse = self.calculate_mse(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE measures the average absolute difference between predicted
        and actual values. It's less sensitive to outliers than MSE/RMSE.
        
        Formula: MAE = (1/n) * Σ|y_i - ŷ_i|
        
        Interpretation:
        - Lower MAE is better
        - Same unit as target (e.g., USD)
        - Less sensitive to outliers
        - Easier to interpret for non-technical stakeholders
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
            
        Returns:
        --------
        float
            MAE value
        """
        mae = mean_absolute_error(y_true, y_pred)
        return mae
    
    def calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        MAPE measures the average percentage error between predicted
        and actual values. It's scale-independent.
        
        Formula: MAPE = (1/n) * Σ|y_i - ŷ_i| / y_i * 100
        
        Interpretation:
        - Lower MAPE is better
        - Expressed as percentage
        - Scale-independent
        - Not suitable when y_i = 0
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
            
        Returns:
        --------
        float
            MAPE value (percentage)
        """
        # Avoid division by zero
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        return mape
    
    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R-squared (R²) score.
        
        R² measures the proportion of variance in the dependent variable
        that is predictable from the independent variables.
        
        Formula: R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
        
        Interpretation:
        - Range: -∞ to 1
        - R² = 1: Perfect prediction
        - R² = 0: Model is as good as predicting the mean
        - R² < 0: Model is worse than predicting the mean
        - Higher R² is better
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
            
        Returns:
        --------
        float
            R² score
        """
        r2 = r2_score(y_true, y_pred)
        return r2
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate a model using all metrics.
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
        model_name : str
            Name of the model being evaluated
            
        Returns:
        --------
        dict
            Dictionary containing all metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = {
            'Model': model_name,
            'MSE': self.calculate_mse(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAE': self.calculate_mae(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred)
        }
        
        # Store metrics history
        self.metrics_history[model_name] = metrics
        
        # Log results
        logger.info(f"  MSE:  {metrics['MSE']:.6f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.6f}")
        logger.info(f"  MAE:  {metrics['MAE']:.6f}")
        logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
        logger.info(f"  R²:   {metrics['R2']:.6f}")
        
        return metrics
    
    def create_comparison_table(self, metrics_dict=None):
        """
        Create a comparison table for multiple models.
        
        Parameters:
        -----------
        metrics_dict : dict or None
            Dictionary of model metrics. If None, use stored history.
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with all models and metrics
        """
        if metrics_dict is None:
            metrics_dict = self.metrics_history
        
        if not metrics_dict:
            logger.warning("No metrics to compare")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Reorder columns
        columns_order = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
        df = df[columns_order]
        
        # Round to 6 decimal places
        df = df.round(6)
        
        logger.info("Model Comparison Table:")
        print(df.to_string(index=False))
        
        return df
    
    def save_comparison_table(self, df, filepath='results/model_comparison.csv'):
        """
        Save the comparison table to a CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Comparison table
        filepath : str
            Path to save the file (default: 'results/model_comparison.csv')
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Comparison table saved to {filepath}")
    
    def get_best_model(self, metric='RMSE'):
        """
        Get the best model based on a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison (default: 'RMSE')
            Options: 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'
            
        Returns:
        --------
        tuple
            (model_name, metric_value)
        """
        if not self.metrics_history:
            raise ValueError("No metrics history available")
        
        if metric not in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']:
            raise ValueError(f"Unknown metric: {metric}")
        
        # For R², higher is better
        if metric == 'R2':
            best_model = max(self.metrics_history.items(), 
                           key=lambda x: x[1][metric])
        else:
            # For other metrics, lower is better
            best_model = min(self.metrics_history.items(), 
                           key=lambda x: x[1][metric])
        
        logger.info(f"Best model by {metric}: {best_model[0]} with value {best_model[1][metric]:.6f}")
        
        return best_model


def main():
    """
    Main function to demonstrate model evaluation.
    """
    # Create sample data for demonstration
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred_1 = y_true + np.random.randn(100) * 0.1
    y_pred_2 = y_true + np.random.randn(100) * 0.2
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    print("=" * 60)
    print("Model 1 Evaluation")
    print("=" * 60)
    metrics_1 = evaluator.evaluate_model(y_true, y_pred_1, "Linear Regression")
    
    print("\n" + "=" * 60)
    print("Model 2 Evaluation")
    print("=" * 60)
    metrics_2 = evaluator.evaluate_model(y_true, y_pred_2, "RNN")
    
    # Create comparison table
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    comparison_df = evaluator.create_comparison_table()
    
    # Get best model
    best_model = evaluator.get_best_model(metric='RMSE')
    print(f"\nBest model by RMSE: {best_model[0]}")
    
    # Save comparison table
    evaluator.save_comparison_table(comparison_df)
    
    return evaluator


if __name__ == "__main__":
    main()
