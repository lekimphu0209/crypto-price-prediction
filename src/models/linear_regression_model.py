"""
Linear Regression Baseline Model for Bitcoin Price Prediction

This module implements a simple Linear Regression model as a baseline
for comparing against more complex models like RNN and LSTM.

Linear Regression Formula:
y = w*x + b

where:
- y is the predicted value
- w is the weight (coefficient)
- x is the input feature
- b is the bias (intercept)

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """
    A Linear Regression model for Bitcoin price prediction.
    
    Linear Regression is the simplest form of regression that attempts
    to model the relationship between two variables by fitting a linear
    equation to observed data.
    
    Why use Linear Regression as a baseline?
    - Simple and interpretable
    - Fast to train
    - Provides a benchmark for more complex models
    - If complex models don't outperform it, they may be overfitting
    """
    
    def __init__(self):
        """
        Initialize the Linear Regression model.
        """
        self.model = LinearRegression()
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """
        Train the Linear Regression model.
        
        Training finds the optimal weights (w) and bias (b) that minimize
        the sum of squared differences between predicted and actual values.
        
        Parameters:
        -----------
        X_train : np.array
            Training features of shape (n_samples, n_features)
        y_train : np.array
            Training target values of shape (n_samples,)
        """
        logger.info("Training Linear Regression model...")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        logger.info("Model trained successfully")
        logger.info(f"Model coefficients: {self.model.coef_}")
        logger.info(f"Model intercept: {self.model.intercept_:.4f}")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.array
            Input features of shape (n_samples, n_features)
            
        Returns:
        --------
        np.array
            Predicted values of shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.array
            Test features
        y_test : np.array
            Test target values
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # R² score (coefficient of determination)
        r2 = self.model.score(X_test, y_test)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        logger.info("Evaluation Results:")
        logger.info(f"  MSE:  {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  R²:   {r2:.6f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance based on model coefficients.
        
        In Linear Regression, the magnitude of coefficients indicates
        the importance of each feature. Larger absolute values mean
        the feature has a stronger effect on the prediction.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature names and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_,
            'Absolute_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='models/linear_regression_model.pkl'):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (default: 'models/linear_regression_model.pkl')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/linear_regression_model.pkl'):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from (default: 'models/linear_regression_model.pkl')
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate Linear Regression model training.
    """
    try:
        # Load preprocessed data
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        # For Linear Regression, we need 2D data (not sequences)
        # Flatten the sequence dimension by taking the last day of each sequence
        X_train_lr = X_train[:, -1, :]  # Take last day of each sequence
        X_test_lr = X_test[:, -1, :]
        
        print(f"Linear Regression Input Shapes:")
        print(f"  X_train: {X_train_lr.shape}")
        print(f"  X_test: {X_test_lr.shape}")
        
        # Initialize and train model
        lr_model = LinearRegressionModel()
        lr_model.train(X_train_lr, y_train)
        
        # Evaluate model
        metrics = lr_model.evaluate(X_test_lr, y_test)
        
        # Save model
        lr_model.save_model()
        
        # Save predictions
        y_pred = lr_model.predict(X_test_lr)
        np.save('results/lr_predictions.npy', y_pred)
        print("\nPredictions saved to results/lr_predictions.npy")
        
        return lr_model, metrics
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run data_preprocessing.py first to generate the preprocessed data.")


if __name__ == "__main__":
    main()
