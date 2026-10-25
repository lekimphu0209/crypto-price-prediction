"""
KerasForecaster - sequence neural networks (RNN, LSTM, BiLSTM, Transformer).

A single class builds the requested architecture based on `name`. The model
operates on prepared 3D windows; it does NOT scale or window the data itself.
Each model is a binary classifier (sigmoid output) predicting the probability
of an UP/TANG trend. Implements IForecaster.
"""
import os

import numpy as np
import tensorflow as tf

from src.domain.interfaces.forecaster import IForecaster

SUPPORTED = {"RNN", "LSTM", "BiLSTM", "Transformer"}


class KerasForecaster(IForecaster):
    """Keras-based sequence forecaster."""

    MODEL_FILE = "model.keras"

    def __init__(
        self,
        name: str,
        epochs: int = 40,
        batch_size: int = 16,
        units: int = 32,
        dropout: float = 0.1,
        rnn_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        l2_reg: float = 0.0,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 0,
        reduce_lr_factor: float = 0.5,
        checkpoint_best: bool = False,
        validation_split: float = 0.1,
        mode: str = "classification",
    ):
        if name not in SUPPORTED:
            raise ValueError(f"Unsupported keras forecaster: {name}")
        if mode not in {"classification", "regression"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self._name = name
        self._epochs = epochs
        self._batch_size = batch_size
        self._units = units
        self._dropout = dropout
        self._rnn_dropout = rnn_dropout
        self._recurrent_dropout = recurrent_dropout
        self._l2_reg = l2_reg
        self._learning_rate = learning_rate
        self._reduce_lr_patience = reduce_lr_patience
        self._reduce_lr_factor = reduce_lr_factor
        self._checkpoint_best = checkpoint_best
        self._early_stopping_patience = early_stopping_patience
        self._validation_split = validation_split
        self._mode = mode
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _build(self, seq_len: int, n_features: int):
        from tensorflow.keras import layers, models, regularizers

        kernel_reg = regularizers.l2(self._l2_reg) if self._l2_reg > 0 else None
        rnn_kwargs = {
            "kernel_regularizer": kernel_reg,
            "recurrent_regularizer": kernel_reg,
            "dropout": self._rnn_dropout,
            "recurrent_dropout": self._recurrent_dropout,
        }
        dense_kwargs = {"kernel_regularizer": kernel_reg} if kernel_reg else {}

        inp = layers.Input(shape=(seq_len, n_features))

        if self._name == "RNN":
            x = layers.SimpleRNN(self._units, activation="tanh", **rnn_kwargs)(inp)
        elif self._name == "LSTM":
            x = layers.LSTM(self._units, **rnn_kwargs)(inp)
        elif self._name == "BiLSTM":
            x = layers.Bidirectional(layers.LSTM(self._units, **rnn_kwargs))(inp)
        else:  # Transformer
            d_model = 32
            h = layers.Dense(d_model, **dense_kwargs)(inp)
            attn = layers.MultiHeadAttention(num_heads=2, key_dim=d_model)(h, h)
            h = layers.LayerNormalization()(layers.Add()([h, attn]))
            ff = layers.Dense(64, activation="relu", **dense_kwargs)(h)
            ff = layers.Dense(d_model, **dense_kwargs)(ff)
            h = layers.LayerNormalization()(layers.Add()([h, ff]))
            x = layers.GlobalAveragePooling1D()(h)
            x = layers.Dense(16, activation="relu", **dense_kwargs)(x)

        # rnn_dropout/recurrent_dropout already applied inside RNN/LSTM layers
        x = layers.Dropout(self._dropout)(x)
        if self._mode == "regression":
            out = layers.Dense(1, activation="linear", **dense_kwargs)(x)
            model = models.Model(inp, out)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
            model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        else:
            out = layers.Dense(1, activation="sigmoid", **dense_kwargs)(x)
            model = models.Model(inp, out)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        import tensorflow as tf

        seq_len, n_features = X.shape[1], X.shape[2]
        self._model = self._build(seq_len, n_features)
        class_weight = self._balanced_class_weight(y) if self._mode == "classification" else None
        callbacks = []
        if self._early_stopping_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self._early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        if self._reduce_lr_patience > 0:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self._reduce_lr_factor,
                    patience=self._reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=0,
                )
            )
        if self._checkpoint_best:
            import tempfile
            checkpoint_path = os.path.join(tempfile.gettempdir(), f"best_{self._name}.keras")
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                )
            )
            self._checkpoint_path = checkpoint_path
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        self._model.fit(
            X, y,
            epochs=self._epochs,
            batch_size=self._batch_size,
            class_weight=class_weight,
            validation_data=validation_data,
            validation_split=self._validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=0,
        )

    @staticmethod
    def _balanced_class_weight(y: np.ndarray) -> dict:
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

    @property
    def mode(self) -> str:
        return self._mode

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).flatten()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._model.save(os.path.join(directory, self.MODEL_FILE))

    def load(self, directory: str) -> None:
        import tensorflow as tf
        self._model = tf.keras.models.load_model(os.path.join(directory, self.MODEL_FILE))
