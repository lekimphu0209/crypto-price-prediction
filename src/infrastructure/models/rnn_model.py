"""
Simple RNN Model for Time Series Prediction

Basic Recurrent Neural Network for cryptocurrency price prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import logging

from .base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRNNModel(BaseModel):
    """
    Simple Recurrent Neural Network for time series prediction.
    
    Uses a basic RNN architecture with:
    - RNN layer for sequence processing
    - Dropout for regularization
    - Dense output layer
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int] = (60, 1),
                 rnn_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initialize Simple RNN Model.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            rnn_units: Number of RNN units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build the Simple RNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Simple RNN layer
            layers.SimpleRNN(
                units=self.rnn_units,
                activation='tanh',
                return_sequences=False,
                kernel_initializer='glorot_uniform'
            ),
            
            # Dropout for regularization
            layers.Dropout(self.dropout_rate),
            
            # Dense output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Simple RNN model built with {self.rnn_units} units")
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1,
            **kwargs) -> 'SimpleRNNModel':
        """
        Train the Simple RNN model.
        
        Args:
            X: Training features (3D array: samples, timesteps, features)
            y: Training labels
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
            **kwargs: Additional arguments for fit()
            
        Returns:
            Trained model
        """
        # Reshape input if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info("Training Simple RNN model...")
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        logger.info("Training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features (2D or 3D array)
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Reshape input if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Optional[keras.callbacks.History]:
        """
        Get the training history.
        
        Returns:
            Training history object
        """
        return self.history
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            print("Model not yet built")
        else:
            self.model.summary()
