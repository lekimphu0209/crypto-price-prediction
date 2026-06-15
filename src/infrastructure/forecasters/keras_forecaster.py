"""
KerasForecaster - sequence neural networks (RNN, LSTM, BiLSTM, Transformer).

A single class builds the requested architecture based on `name`. The model
operates on prepared 3D windows; it does NOT scale or window the data itself.
Implements IForecaster.
"""
import os

import numpy as np

from src.domain.interfaces.forecaster import IForecaster

SUPPORTED = {"RNN", "LSTM", "BiLSTM", "Transformer"}


class KerasForecaster(IForecaster):
    """Keras-based sequence forecaster."""

    MODEL_FILE = "model.keras"

    def __init__(self, name: str, epochs: int = 40, batch_size: int = 16):
        if name not in SUPPORTED:
            raise ValueError(f"Unsupported keras forecaster: {name}")
        self._name = name
        self._epochs = epochs
        self._batch_size = batch_size
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _build(self, seq_len: int, n_features: int):
        from tensorflow.keras import layers, models

        inp = layers.Input(shape=(seq_len, n_features))

        if self._name == "RNN":
            x = layers.SimpleRNN(32, activation="tanh")(inp)
        elif self._name == "LSTM":
            x = layers.LSTM(32)(inp)
        elif self._name == "BiLSTM":
            x = layers.Bidirectional(layers.LSTM(32))(inp)
        else:  # Transformer
            d_model = 32
            h = layers.Dense(d_model)(inp)
            attn = layers.MultiHeadAttention(num_heads=2, key_dim=d_model)(h, h)
            h = layers.LayerNormalization()(layers.Add()([h, attn]))
            ff = layers.Dense(64, activation="relu")(h)
            ff = layers.Dense(d_model)(ff)
            h = layers.LayerNormalization()(layers.Add()([h, ff]))
            x = layers.GlobalAveragePooling1D()(h)
            x = layers.Dense(16, activation="relu")(x)

        x = layers.Dropout(0.1)(x)
        out = layers.Dense(1)(x)

        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        seq_len, n_features = X.shape[1], X.shape[2]
        self._model = self._build(seq_len, n_features)
        self._model.fit(
            X, y,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).flatten()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._model.save(os.path.join(directory, self.MODEL_FILE))

    def load(self, directory: str) -> None:
        import tensorflow as tf
        self._model = tf.keras.models.load_model(os.path.join(directory, self.MODEL_FILE))
