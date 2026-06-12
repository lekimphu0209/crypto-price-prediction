"""
Transformer Model Implementation
"""
import joblib
import numpy as np
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.infrastructure.models.base_model import BaseModel


class TransformerModel(BaseModel):
    """Transformer model implementation for time series prediction"""
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout_rate: Dropout rate
        """
        super().__init__()
        self._sequence_length = sequence_length
        self._n_features = n_features
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._dropout_rate = dropout_rate
        self._name = "Transformer"
        self._params = {
            'sequence_length': sequence_length,
            'n_features': n_features,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout_rate': dropout_rate
        }
        self._scaler = None
    
    def _positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding"""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def _transformer_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transformer encoder block"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self._n_heads,
            key_dim=self._d_model // self._n_heads
        )(inputs, inputs)
        attention_output = Dropout(self._dropout_rate)(attention_output)
        
        # Add & Normalize
        x = Add()([inputs, attention_output])
        x = LayerNormalization()(x)
        
        # Feed-forward
        ff_output = Dense(self._d_model * 2, activation='relu')(x)
        ff_output = Dense(self._d_model)(ff_output)
        ff_output = Dropout(self._dropout_rate)(ff_output)
        
        # Add & Normalize
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
        
        return x
    
    def _build_model(self) -> Model:
        """Build Transformer model"""
        # Input layer
        inputs = Input(shape=(self._sequence_length, self._n_features))
        
        # Project to d_model
        x = Dense(self._d_model)(inputs)
        
        # Add positional encoding
        pe = self._positional_encoding(self._sequence_length, self._d_model)
        x = x + pe
        
        # Transformer encoder layers
        for _ in range(self._n_layers):
            x = self._transformer_encoder(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(self._dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer training"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self._sequence_length):
            X_seq.append(X[i:i + self._sequence_length])
            y_seq.append(y[i + self._sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """Train Transformer model"""
        from sklearn.preprocessing import StandardScaler
        
        # Update n_features based on input
        self._n_features = X.shape[1]
        self._params['n_features'] = self._n_features
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build model
        self._model = self._build_model()
        
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
        """Make predictions"""
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
        """Evaluate model performance"""
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
        """Save model to file"""
        self._check_trained()
        
        self._model.save(path + '.h5')
        
        joblib.dump({
            'scaler': self._scaler,
            'is_trained': self._is_trained,
            'params': self._params
        }, path + '_scaler.pkl')
    
    def load(self, path: str) -> None:
        """Load model from file"""
        self._model = tf.keras.models.load_model(path + '.h5')
        
        scaler_data = joblib.load(path + '_scaler.pkl')
        self._scaler = scaler_data['scaler']
        self._is_trained = scaler_data.get('is_trained', True)
        self._params = scaler_data.get('params', {})
        self._sequence_length = self._params.get('sequence_length', 30)
        self._n_features = self._params.get('n_features', 10)
