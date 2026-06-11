"""
Data Preprocessing Module for Bitcoin Price Prediction

This module handles data preprocessing steps including:
- Handling missing values
- Sorting data by datetime
- Normalization using MinMaxScaler
- Creating sequences for RNN/LSTM models

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to preprocess Bitcoin price data for machine learning models.
    
    This class handles:
    - Missing value imputation
    - Data sorting and indexing
    - Feature scaling (normalization)
    - Sequence creation for time series models
    - Train/test splitting
    """
    
    def __init__(self, data):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing the Bitcoin price data with features
        """
        self.data = data.copy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
    def handle_missing_values(self, method='forward_fill'):
        """
        Handle missing values in the dataset.
        
        Missing values can occur due to:
        - Market holidays
        - Data collection issues
        - Technical indicator calculation lag
        
        Parameters:
        -----------
        method : str
            Method to handle missing values:
            - 'forward_fill': Use previous value (default)
            - 'backward_fill': Use next value
            - 'mean': Use mean of the column
            - 'drop': Drop rows with missing values
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using method: {method}")
        
        missing_before = self.data.isnull().sum().sum()
        logger.info(f"Missing values before handling: {missing_before}")
        
        if method == 'forward_fill':
            self.data = self.data.fillna(method='ffill')
            # If there are still NaN values at the beginning, use backward fill
            self.data = self.data.fillna(method='bfill')
        elif method == 'backward_fill':
            self.data = self.data.fillna(method='bfill')
            self.data = self.data.fillna(method='ffill')
        elif method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif method == 'drop':
            self.data = self.data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_after = self.data.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return self.data
    
    def sort_by_date(self, date_column='Date'):
        """
        Sort the dataset by date in ascending order.
        
        Time series data must be sorted chronologically for:
        - Correct sequence creation
        - Proper train/test split
        - Avoiding look-ahead bias
        
        Parameters:
        -----------
        date_column : str
            Name of the date column (default: 'Date')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame sorted by date
        """
        logger.info(f"Sorting data by {date_column}")
        
        # Ensure date column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Sort by date
        self.data = self.data.sort_values(by=date_column).reset_index(drop=True)
        
        logger.info(f"Data sorted. Date range: {self.data[date_column].min()} to {self.data[date_column].max()}")
        
        return self.data
    
    def normalize_data(self, columns_to_scale=None, save_scaler=True, scaler_path='models/scaler.pkl'):
        """
        Normalize data using MinMaxScaler.
        
        MinMaxScaler transforms features to a range between 0 and 1.
        This is important for:
        - Neural networks (they converge faster with normalized data)
        - Features with different scales (e.g., price vs volume)
        
        Formula:
        x' = (x - x_min) / (x_max - x_min)
        
        Parameters:
        -----------
        columns_to_scale : list or None
            List of columns to scale. If None, scale all numeric columns except Date
        save_scaler : bool
            Whether to save the scaler for later use (default: True)
        scaler_path : str
            Path to save the scaler (default: 'models/scaler.pkl')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized features
        """
        logger.info("Normalizing data using MinMaxScaler")
        
        # Identify columns to scale
        if columns_to_scale is None:
            # Scale all numeric columns except Date
            columns_to_scale = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Date' in columns_to_scale:
                columns_to_scale.remove('Date')
        
        self.feature_columns = columns_to_scale
        
        # Store original data for inverse transformation
        self.original_data = self.data[columns_to_scale].copy()
        
        # Fit and transform the data
        self.data[columns_to_scale] = self.scaler.fit_transform(self.data[columns_to_scale])
        
        logger.info(f"Normalized {len(columns_to_scale)} columns")
        logger.info(f"Columns scaled: {columns_to_scale}")
        
        # Save scaler for inverse transformation
        if save_scaler:
            import os
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
        
        return self.data
    
    def inverse_transform(self, scaled_values):
        """
        Inverse transform normalized values back to original scale.
        
        Parameters:
        -----------
        scaled_values : np.array
            Normalized values to transform back
            
        Returns:
        --------
        np.array
            Values in original scale
        """
        return self.scaler.inverse_transform(scaled_values)
    
    def create_sequences(self, feature_columns, target_column='Close', sequence_length=30):
        """
        Create sequences for time series prediction (RNN/LSTM).
        
        For time series prediction, we use a sliding window approach:
        - Input: Previous `sequence_length` days of features
        - Output: Next day's target value
        
        Example with sequence_length=30:
        - X[0] = features from day 0 to 29
        - y[0] = target on day 30
        
        This captures temporal dependencies in the data.
        
        Parameters:
        -----------
        feature_columns : list
            List of feature columns to use as input
        target_column : str
            Column to predict (default: 'Close')
        sequence_length : int
            Number of past days to use for prediction (default: 30)
            
        Returns:
        --------
        tuple
            (X, y) where:
            - X: Array of shape (samples, sequence_length, features)
            - y: Array of shape (samples,)
        """
        logger.info(f"Creating sequences with sequence_length={sequence_length}")
        
        # Extract features and target
        features = self.data[feature_columns].values
        target = self.data[target_column].values
        
        X, y = [], []
        
        # Create sequences
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"X shape: {X.shape} (samples, sequence_length, features)")
        logger.info(f"y shape: {y.shape} (samples,)")
        
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, shuffle=False):
        """
        Split data into training and testing sets.
        
        For time series data, we typically do NOT shuffle the data
        because temporal order is important. We use the most recent
        data for testing and older data for training.
        
        Parameters:
        -----------
        X : np.array
            Feature sequences
        y : np.array
            Target values
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        shuffle : bool
            Whether to shuffle data (default: False for time series)
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}, shuffle={shuffle}")
        
        # Calculate split point
        split_idx = int(len(X) * (1 - test_size))
        
        if shuffle:
            # Random split (not recommended for time series)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            # Sequential split (recommended for time series)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_for_regression(self, feature_columns, target_column='Close'):
        """
        Prepare data for regression models (Linear Regression, etc.).
        
        Regression models don't require sequences, so we use the
        current day's features to predict the next day's target.
        
        Parameters:
        -----------
        feature_columns : list
            List of feature columns to use as input
        target_column : str
            Column to predict (default: 'Close')
            
        Returns:
        --------
        tuple
            (X, y) where X and y are 2D arrays
        """
        logger.info("Preparing data for regression models")
        
        # Use current day's features to predict next day's target
        X = self.data[feature_columns].values[:-1]
        y = self.data[target_column].values[1:]
        
        logger.info(f"X shape: {X.shape} (samples, features)")
        logger.info(f"y shape: {y.shape} (samples,)")
        
        return X, y
    
    def get_preprocessing_summary(self):
        """
        Get a summary of the preprocessing steps.
        
        Returns:
        --------
        dict
            Dictionary containing preprocessing summary
        """
        summary = {
            'total_samples': len(self.data),
            'feature_columns': self.feature_columns,
            'missing_values': self.data.isnull().sum().sum(),
            'date_range': (self.data['Date'].min(), self.data['Date'].max()) if 'Date' in self.data.columns else None
        }
        
        return summary


def main():
    """
    Main function to demonstrate data preprocessing.
    """
    try:
        # Load data with features
        data = pd.read_csv("data/bitcoin_data_with_features.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        
        print("Original data shape:", data.shape)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data)
        
        # Handle missing values
        data = preprocessor.handle_missing_values(method='forward_fill')
        
        # Sort by date
        data = preprocessor.sort_by_date()
        
        # Normalize data (exclude Date column)
        feature_cols = [col for col in data.columns if col != 'Date']
        data = preprocessor.normalize_data(columns_to_scale=feature_cols)
        
        # Prepare for RNN (with sequences)
        feature_columns = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'ATR']
        X, y = preprocessor.create_sequences(feature_columns=feature_columns, sequence_length=30)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, test_size=0.2)
        
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Prepare for regression (without sequences)
        X_reg, y_reg = preprocessor.prepare_for_regression(
            feature_columns=feature_columns, 
            target_column='Close'
        )
        
        print(f"\nRegression X shape: {X_reg.shape}")
        print(f"Regression y shape: {y_reg.shape}")
        
        # Save preprocessed data
        np.save('data/X_train.npy', X_train)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_train.npy', y_train)
        np.save('data/y_test.npy', y_test)
        
        print("\nPreprocessed data saved to data/ directory")
        
    except FileNotFoundError:
        print("Error: bitcoin_data_with_features.csv not found. Run feature_engineering.py first.")


if __name__ == "__main__":
    main()
