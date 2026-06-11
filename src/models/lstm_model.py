"""
LSTM (Long Short-Term Memory) Model for Bitcoin Price Prediction

This module implements an LSTM model for time series prediction.

LSTM Cell Formula:
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)    # Candidate cell state
C_t = f_t * C_{t-1} + i_t * C̃_t            # Cell state
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)                       # Hidden state

Why LSTM over Simple RNN?
- LSTM solves the vanishing gradient problem
- Can learn long-term dependencies (weeks/months of price patterns)
- Better for capturing complex temporal relationships
- More stable training for long sequences

Author: Bitcoin Price Prediction Project
Date: 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    An LSTM model for Bitcoin price prediction.
    
    Architecture:
    - LSTM(50): LSTM layer with 50 units
    - Dropout(0.2): Dropout for regularization
    - Dense(1): Output layer for price prediction
    
    LSTM vs Simple RNN:
    - LSTM has internal memory cell (C_t) in addition to hidden state (h_t)
    - Three gates control information flow: forget, input, output
    - Can remember information for long periods
    - More computationally expensive but more powerful
    """
    
    def __init__(self, sequence_length=30, n_features=6, lstm_units=50, dropout_rate=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Number of past days to use as input (default: 30)
        n_features : int
            Number of input features (default: 6)
        lstm_units : int
            Number of LSTM units (hidden size) (default: 50)
        dropout_rate : float
            Dropout rate for regularization (default: 0.2)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self):
        """
        Build the LSTM model architecture.
        
        Layer Explanation:
        1. LSTM(50): Processes the input sequence with memory cells
           - 50 units means the hidden state has 50 dimensions
           - Internal cell state maintains long-term memory
           - Gates control what to remember/forget
           - return_sequences=False: Return only the final hidden state
        
        2. Dropout(0.2): Randomly drops 20% of neurons during training
           - Prevents overfitting
           - Forces the network to learn robust features
        
        3. Dense(1): Output layer with single neuron
           - Predicts the next day's price
           - Linear activation for regression
        """
        logger.info("Building LSTM model...")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Number of features: {self.n_features}")
        logger.info(f"  LSTM units: {self.lstm_units}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        
        self.model = Sequential([
            # LSTM layer
            # - input_shape=(sequence_length, n_features): Shape of input sequences
            # - return_sequences=False: Return only the final hidden state
            LSTM(
                units=self.lstm_units,
                activation='tanh',
                recurrent_activation='sigmoid',
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
        Train the LSTM model.
        
        Training Process:
        1. Forward pass: Compute predictions through LSTM cells
        2. Calculate loss (MSE between predictions and actual values)
        3. Backward pass: Compute gradients through time (BPTT)
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
        
        logger.info("Training LSTM model...")
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
            checkpoint_path = 'models/lstm_best_model.h5'
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
    
    def save_model(self, filepath='models/lstm_model.h5'):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (default: 'models/lstm_model.h5')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/lstm_model.h5'):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from (default: 'models/lstm_model.h5')
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def save_training_history(self, filepath='results/lstm_training_history.json'):
        """
        Save training history to a JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the history (default: 'results/lstm_training_history.json')
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
    Main function to demonstrate LSTM model training.
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
        
        # Initialize LSTM model
        lstm_model = LSTMModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            lstm_units=50,
            dropout_rate=0.2
        )
        
        # Build and train model
        lstm_model.build_model()
        history = lstm_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=32,
            patience=10
        )
        
        # Evaluate model
        metrics = lstm_model.evaluate(X_test, y_test)
        
        # Save model and history
        lstm_model.save_model()
        lstm_model.save_training_history()
        
        # Save predictions
        y_pred = lstm_model.predict(X_test)
        np.save('results/lstm_predictions.npy', y_pred)
        print("\nPredictions saved to results/lstm_predictions.npy")
        
        return lstm_model, metrics
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run data_preprocessing.py first to generate the preprocessed data.")


if __name__ == "__main__":
    main()
