"""
Model Training Module
Implement các mô hình: Linear Regression và RNN
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressionModel:
    """Mô hình Linear Regression cơ bản"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train mô hình Linear Regression
        
        Args:
            X_train: Features training data
            y_train: Target training data
        """
        # Lưu tên features
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        print("Linear Regression model đã train xong")
        print(f"R² score trên training set: {self.model.score(X_train_scaled, y_train):.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán giá
        
        Args:
            X: Features data
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Đánh giá mô hình
        
        Args:
            X_test: Features test data
            y_test: Target test data
        
        Returns:
            Dictionary với các metrics
        """
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Direction accuracy
        direction_pred = (predictions > X_test['close'].values).astype(int)
        direction_true = (y_test.values > X_test['close'].values).astype(int)
        direction_acc = np.mean(direction_pred == direction_true)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Direction_Accuracy': direction_acc
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Lấy feature importance (coefficients)"""
        if not self.is_fitted:
            raise ValueError("Model chưa được train")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        
        importance['abs_coefficient'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def save_model(self, filepath: str):
        """Lưu model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Đã lưu model vào {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        print(f"Đã load model từ {filepath}")


class RNNModel:
    """Mô hình RNN đơn giản cho time series prediction"""
    
    def __init__(self, sequence_length: int = 30, units: int = 64):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo sequences cho RNN
        
        Args:
            data: Feature data
            target: Target data
        
        Returns:
            X (sequences), y (next values)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Xây dựng mô hình RNN
        
        Args:
            input_shape: Shape của input (sequence_length, n_features)
        """
        self.model = Sequential([
            SimpleRNN(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            SimpleRNN(self.units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("RNN model architecture:")
        self.model.summary()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train mô hình RNN
        
        Args:
            X_train: Features training data
            y_train: Target training data
            epochs: Số epochs
            batch_size: Batch size
            validation_split: Tỷ lệ validation
        """
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Tạo sequences
        X_seq, y_seq = self.create_sequences(X_train_scaled, y_train.values)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Sequence shape: {X_seq.shape}")
        
        # Build model nếu chưa có
        if self.model is None:
            self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.is_fitted = True
        
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán giá
        
        Args:
            X: Features data
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X)))
        
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Đánh giá mô hình
        
        Args:
            X_test: Features test data
            y_test: Target test data
        
        Returns:
            Dictionary với các metrics
        """
        predictions = self.predict(X_test)
        
        # Cần align predictions với y_test (do sequence length)
        y_test_aligned = y_test.iloc[self.sequence_length:].values
        X_test_aligned = X_test.iloc[self.sequence_length:]
        
        mae = mean_absolute_error(y_test_aligned, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_aligned, predictions))
        r2 = r2_score(y_test_aligned, predictions)
        
        # Direction accuracy
        direction_pred = (predictions > X_test_aligned['close'].values).astype(int)
        direction_true = (y_test_aligned > X_test_aligned['close'].values).astype(int)
        direction_acc = np.mean(direction_pred == direction_true)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Direction_Accuracy': direction_acc
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Lưu model"""
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Đã lưu model vào {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        self.model = tf.keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        print(f"Đã load model từ {filepath}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """Vẽ đồ thị so sánh predictions và actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_history(history, title: str = "Training History"):
    """Vẽ đồ thị training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test models with dummy data
    print("Testing Linear Regression and RNN models...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'ma_7': np.random.randn(n_samples).cumsum() + 100,
        'ma_30': np.random.randn(n_samples).cumsum() + 100,
        'rsi_14': np.random.uniform(30, 70, n_samples),
        'gold_close': np.random.randn(n_samples).cumsum() + 180,
        'dxy_close': np.random.randn(n_samples).cumsum() + 100
    }
    
    df = pd.DataFrame(data)
    df['target'] = df['close'].shift(-1)
    df = df.dropna()
    
    # Split data
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Test Linear Regression
    print("\n=== Testing Linear Regression ===")
    lr_model = LinearRegressionModel()
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    print("Linear Regression Metrics:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test RNN
    print("\n=== Testing RNN ===")
    rnn_model = RNNModel(sequence_length=30, units=64)
    history = rnn_model.train(X_train, y_train, epochs=5, batch_size=32)
    rnn_metrics = rnn_model.evaluate(X_test, y_test)
    print("RNN Metrics:")
    for metric, value in rnn_metrics.items():
        print(f"  {metric}: {value:.4f}")
