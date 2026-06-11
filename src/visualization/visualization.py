"""
Visualization Module for Bitcoin Price Prediction

This module provides functions to create professional visualizations
for Bitcoin price prediction analysis.

Visualizations Included:
1. Bitcoin price history
2. Technical indicators (RSI, MA, MACD, ATR)
3. Train/test split visualization
4. Predicted vs Actual comparison
5. Loss curves for neural networks
6. Model comparison charts

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BitcoinVisualizer:
    """
    A class to create visualizations for Bitcoin price prediction analysis.
    
    This class provides methods to create various types of charts
    for exploratory data analysis and model evaluation.
    """
    
    def __init__(self, save_dir='results'):
        """
        Initialize the BitcoinVisualizer.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save visualizations (default: 'results')
        """
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    def plot_price_history(self, data, title='Bitcoin Price History'):
        """
        Plot Bitcoin price history over time.
        
        This visualization shows the trend of Bitcoin prices,
        helping identify patterns, trends, and volatility.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with 'Date' and 'Close' columns
        title : str
            Chart title (default: 'Bitcoin Price History')
        """
        plt.figure(figsize=(14, 6))
        plt.plot(data['Date'], data['Close'], linewidth=1.5, color='#F7931A', label='BTC Price')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/price_history.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Price history plot saved to {filepath}")
    
    def plot_technical_indicators(self, data, indicators=['MA20', 'MA50', 'RSI']):
        """
        Plot technical indicators along with price.
        
        This visualization shows how technical indicators
        relate to price movements.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with price and indicator columns
        indicators : list
            List of indicator columns to plot (default: ['MA20', 'MA50', 'RSI'])
        """
        # Create subplots
        fig, axes = plt.subplots(len(indicators) + 1, 1, figsize=(14, 4 * (len(indicators) + 1)))
        
        # Plot price
        axes[0].plot(data['Date'], data['Close'], linewidth=1.5, color='#F7931A', label='Close Price')
        
        # Plot moving averages on price chart
        if 'MA20' in indicators:
            axes[0].plot(data['Date'], data['MA20'], linewidth=1, color='blue', label='MA20', alpha=0.7)
        if 'MA50' in indicators:
            axes[0].plot(data['Date'], data['MA50'], linewidth=1, color='red', label='MA50', alpha=0.7)
        
        axes[0].set_title('Bitcoin Price with Moving Averages', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (USD)', fontsize=10)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot other indicators
        for i, indicator in enumerate(indicators):
            if indicator in ['MA20', 'MA50']:
                continue
            
            ax = axes[i + 1]
            ax.plot(data['Date'], data[indicator], linewidth=1, color='purple', label=indicator)
            
            # Add overbought/oversold lines for RSI
            if indicator == 'RSI':
                ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
                ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
                ax.set_ylabel('RSI', fontsize=10)
            else:
                ax.set_ylabel(indicator, fontsize=10)
            
            ax.set_title(f'{indicator} Indicator', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/technical_indicators.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Technical indicators plot saved to {filepath}")
    
    def plot_train_test_split(self, train_data, test_data, split_date):
        """
        Visualize train/test split.
        
        This visualization shows how the data is split into
        training and testing sets for model evaluation.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with 'Date' and 'Close' columns
        test_data : pd.DataFrame
            Test data with 'Date' and 'Close' columns
        split_date : str
            Date of the split point
        """
        plt.figure(figsize=(14, 6))
        
        plt.plot(train_data['Date'], train_data['Close'], 
                linewidth=1.5, color='blue', label='Training Data')
        plt.plot(test_data['Date'], test_data['Close'], 
                linewidth=1.5, color='red', label='Test Data')
        
        plt.axvline(x=pd.to_datetime(split_date), color='black', 
                   linestyle='--', linewidth=2, label='Split Point')
        
        plt.title('Train/Test Split Visualization', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        filepath = f'{self.save_dir}/train_test_split.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Train/test split plot saved to {filepath}")
    
    def plot_predictions(self, y_true, y_pred, model_name, dates=None):
        """
        Plot predicted vs actual values.
        
        This visualization compares model predictions with
        actual values to assess model performance.
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
        model_name : str
            Name of the model
        dates : pd.Series or None
            Dates for x-axis (optional)
        """
        plt.figure(figsize=(14, 6))
        
        if dates is not None:
            x = dates
            plt.plot(x, y_true, linewidth=1.5, color='blue', label='Actual', alpha=0.7)
            plt.plot(x, y_pred, linewidth=1.5, color='red', label='Predicted', alpha=0.7)
        else:
            x = range(len(y_true))
            plt.plot(x, y_true, linewidth=1.5, color='blue', label='Actual', alpha=0.7)
            plt.plot(x, y_pred, linewidth=1.5, color='red', label='Predicted', alpha=0.7)
        
        plt.title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (Normalized)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if dates is not None:
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/{model_name.lower()}_predictions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictions plot saved to {filepath}")
    
    def plot_loss_curve(self, history_path, model_name):
        """
        Plot training and validation loss curves.
        
        This visualization shows how the model's loss changes
        during training, helping identify overfitting or underfitting.
        
        Parameters:
        -----------
        history_path : str
            Path to the training history JSON file
        model_name : str
            Name of the model
        """
        # Load training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history['loss'], linewidth=2, color='blue', label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], linewidth=2, color='red', label='Validation Loss')
        axes[0].set_title(f'{model_name}: Loss Curve', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss (MSE)', fontsize=10)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        axes[1].plot(history['mae'], linewidth=2, color='blue', label='Training MAE')
        if 'val_mae' in history:
            axes[1].plot(history['val_mae'], linewidth=2, color='red', label='Validation MAE')
        axes[1].set_title(f'{model_name}: MAE Curve', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('MAE', fontsize=10)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/{model_name.lower()}_loss_curve.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss curve plot saved to {filepath}")
    
    def plot_model_comparison(self, comparison_df):
        """
        Create a bar chart comparing model performance.
        
        This visualization compares different models based on
        evaluation metrics.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison results
        """
        metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
        models = comparison_df['Model'].values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = comparison_df[metric].values
            bars = axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric, fontsize=10)
            axes[i].set_xlabel('Model', fontsize=10)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/model_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {filepath}")
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """
        Plot residuals (prediction errors).
        
        Residuals help identify patterns in prediction errors.
        Ideally, residuals should be randomly distributed around zero.
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        y_pred : np.array
            Predicted values
        model_name : str
            Name of the model
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f'{model_name}: Residual Plot', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted Values', fontsize=10)
        axes[0].set_ylabel('Residuals', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Residuals', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{self.save_dir}/{model_name.lower()}_residuals.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residuals plot saved to {filepath}")
    
    def plot_correlation_heatmap(self, data, features):
        """
        Plot correlation heatmap of features.
        
        This visualization shows correlations between features,
        helping identify multicollinearity.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with feature columns
        features : list
            List of feature names to include
        """
        correlation_matrix = data[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = f'{self.save_dir}/correlation_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation heatmap saved to {filepath}")


def main():
    """
    Main function to demonstrate visualization capabilities.
    """
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    prices = np.cumsum(np.random.randn(1000)) + 10000
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'MA20': pd.Series(prices).rolling(20).mean(),
        'MA50': pd.Series(prices).rolling(50).mean(),
        'RSI': 50 + np.random.randn(1000) * 10
    })
    
    # Initialize visualizer
    visualizer = BitcoinVisualizer()
    
    # Create visualizations
    visualizer.plot_price_history(data)
    visualizer.plot_technical_indicators(data)
    
    # Sample predictions
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    visualizer.plot_predictions(y_true, y_pred, "Sample Model")
    visualizer.plot_residuals(y_true, y_pred, "Sample Model")
    
    print("Visualization demonstration completed")


if __name__ == "__main__":
    main()
