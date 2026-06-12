"""
BiLSTM Model Implementation
"""
import joblib
import numpy as np
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.infrastructure.models.base_model import BaseModel


class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model implementation for time series prediction"""
    
    def __init__(
        self,
        sequence_length: int = 30,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        dense_units: int = 32
    ):
        """
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            dense_units: Number of dense layer units
        """
        super().__init__()
        self._sequence_length = sequence_length
        self._lstm_units = lstm_units
        self._dropout_rate = dropout_rate
        self._dense_units = dense_units
        self._name = "BiLSTM"
        self._params = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'dense_units': dense_units
        }
        self._scaler = None
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for BiLSTM training
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            Tuple of (X sequences, y sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self._sequence_length):
            X_seq.append(X[i:i + self._sequence_length])
            y_seq.append(y[i + self._sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build BiLSTM model architecture
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
        
        Returns:
            Compiled BiLSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(self._lstm_units, return_sequences=True), input_shape=input_shape),
            Dropout(self._dropout_rate),
            Bidirectional(LSTM(self._lstm_units // 2, return_sequences=False)),
            Dropout(self._dropout_rate),
            Dense(self._dense_units, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train BiLSTM model
        
        Args:
            X: Feature matrix
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
        """
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build model
        input_shape = (self._sequence_length, X_seq.shape[2])
        self._model = self._build_model(input_shape)
        
        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self._model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self._is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        self._check_trained()
        
        # Scale features
        X_scaled = self._scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Predict
        predictions = self._model.predict(X_seq, verbose=0)
        
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True values
        
        Returns:
            Dictionary of metrics
        """
        self._check_trained()
        
        y_pred = self.predict(X)
        
        # Align y with predictions
        y_aligned = y[self._sequence_length:self._sequence_length + len(y_pred)]
        
        return {
            'mae': float(mean_absolute_error(y_aligned, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_aligned, y_pred))),
            'r2': float(r2_score(y_aligned, y_pred))
        }
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: File path to save model
        """
        self._check_trained()
        
        self._model.save(path + '.h5')
        
        joblib.dump({
            'scaler': self._scaler,
            'is_trained': self._is_trained,
            'params': self._params
        }, path + '_scaler.pkl')
    
    def load(self, path: str) -> None:
        """
        Load model from file
        
        Args:
            path: File path to load model from
        """
        self._model = tf.keras.models.load_model(path + '.h5')
        
        scaler_data = joblib.load(path + '_scaler.pkl')
        self._scaler = scaler_data['scaler']
        self._is_trained = scaler_data.get('is_trained', True)
        self._params = scaler_data.get('params', {})
        self._sequence_length = self._params.get('sequence_length', 30)
