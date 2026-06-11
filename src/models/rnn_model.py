"""
RNN (Recurrent Neural Network) Model for Bitcoin Price Prediction

This module implements a Simple RNN model for time series prediction.

RNN Hidden State Formula:
h_t = f(W * x_t + U * h_{t-1} + b)

where:
- h_t is the hidden state at time t
- x_t is the input at time t
- h_{t-1} is the hidden state at time t-1
- W is the input weight matrix
- U is the recurrent weight matrix
- b is the bias
- f is the activation function (usually tanh)

Why RNN for Time Series?
- RNNs have internal memory (hidden state) that captures temporal dependencies
- They can process sequences of arbitrary length
- The hidden state acts as a summary of past information
- Suitable for sequential data like stock prices

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RNNModel:
    """
    A Simple RNN model for Bitcoin price prediction.
    
    Architecture:
    - SimpleRNN(50): Recurrent layer with 50 units
    - Dropout(0.2): Dropout for regularization
    - Dense(1): Output layer for price prediction
    
    This architecture is simple yet effective for capturing
    temporal patterns in Bitcoin price data.
    """
    
    def __init__(self, sequence_length=30, n_features=6, rnn_units=50, dropout_rate=0.2):
        """
        Initialize the RNN model.
        
        Parameters:
        -----------
        sequence_length : int
            Number of past days to use as input (default: 30)
        n_features : int
            Number of input features (default: 6)
        rnn_units : int
            Number of RNN units (hidden size) (default: 50)
        dropout_rate : float
            Dropout rate for regularization (default: 0.2)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self):
        """
        Build the RNN model architecture.
        
        Layer Explanation:
        1. SimpleRNN(50): Processes the input sequence and maintains hidden state
           - 50 units means the hidden state has 50 dimensions
           - Each unit learns to capture different temporal patterns
           - Returns the final hidden state (not the full sequence)
        
        2. Dropout(0.2): Randomly drops 20% of neurons during training
           - Prevents overfitting
           - Forces the network to learn robust features
        
        3. Dense(1): Output layer with single neuron
           - Predicts the next day's price
           - Linear activation for regression
        """
        logger.info("Building RNN model...")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Number of features: {self.n_features}")
        logger.info(f"  RNN units: {self.rnn_units}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        
        self.model = Sequential([
            # SimpleRNN layer
            # - input_shape=(sequence_length, n_features): Shape of input sequences
            # - return_sequences=False: Return only the final hidden state
            SimpleRNN(
                units=self.rnn_units,
                activation='tanh',
                input_shape=(self.sequence_length, self.n_features),
                return_sequences=False
            ),
            
            # Dropout layer for regularization
            Dropout(self.dropout_rate),
            
            # Dense output layer
            # - 1 unit: Predict single value (next day's price)
            # - linear activation: For regression (no activation)
            Dense(1, activation='linear')
        ])
        
        # Compile the model
        # - Adam optimizer: Adaptive learning rate optimization
        # - MSE loss: Mean Squared Error for regression
        # - metrics: Track MAE during training
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model built successfully")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, patience=10):
        """
        Train the RNN model.
        
        Training Process:
        1. Forward pass: Compute predictions
        2. Calculate loss (MSE between predictions and actual values)
        3. Backward pass: Compute gradients
        4. Update weights using Adam optimizer
        5. Repeat for all epochs
        
        Parameters:
        -----------
        X_train : np.array
            Training sequences of shape (n_samples, sequence_length, n_features)
        y_train : np.array
            Training target values of shape (n_samples,)
        X_val : np.array or None
            Validation sequences (optional)
        y_val : np.array or None
            Validation target values (optional)
        epochs : int
            Number of training epochs (default: 50)
        batch_size : int
            Batch size for training (default: 32)
        patience : int
            Early stopping patience (default: 10)
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Training RNN model...")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Define callbacks
        callbacks = []
        
        # Early stopping: Stop training if validation loss doesn't improve
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Model checkpoint: Save best model
            checkpoint_path = 'models/rnn_best_model.h5'
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.array
            Input sequences of shape (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        np.array
            Predicted values of shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X, verbose=0)
        
        # Flatten predictions to 1D array
        predictions = predictions.flatten()
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.array
            Test sequences
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
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        logger.info("Evaluation Results:")
        logger.info(f"  MSE:  {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def save_model(self, filepath='models/rnn_model.h5'):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (default: 'models/rnn_model.h5')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/rnn_model.h5'):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from (default: 'models/rnn_model.h5')
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def save_training_history(self, filepath='results/rnn_training_history.json'):
        """
        Save training history to a JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the history (default: 'results/rnn_training_history.json')
        """
        if self.history is None:
            raise ValueError("No training history to save")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert history to dictionary
        history_dict = {key: [float(x) for x in value] 
                       for key, value in self.history.history.items()}
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")
    
    def get_training_history(self):
        """
        Get the training history.
        
        Returns:
        --------
        dict
            Dictionary containing training history
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        return self.history.history


def main():
    """
    Main function to demonstrate RNN model training.
    """
    try:
        # Load preprocessed data
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Data Shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Split training data into train and validation
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"\nAfter Validation Split:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        
        # Initialize RNN model
        rnn_model = RNNModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            rnn_units=50,
            dropout_rate=0.2
        )
        
        # Build and train model
        rnn_model.build_model()
        history = rnn_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=32,
            patience=10
        )
        
        # Evaluate model
        metrics = rnn_model.evaluate(X_test, y_test)
        
        # Save model and history
        rnn_model.save_model()
        rnn_model.save_training_history()
        
        # Save predictions
        y_pred = rnn_model.predict(X_test)
        np.save('results/rnn_predictions.npy', y_pred)
        print("\nPredictions saved to results/rnn_predictions.npy")
        
        return rnn_model, metrics
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run data_preprocessing.py first to generate the preprocessed data.")


if __name__ == "__main__":
    main()
