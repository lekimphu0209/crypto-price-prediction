"""
Models module - Các mô hình dự đoán
"""
from .linear_regression_model import LinearRegressionModel

# Chỉ import RNN/LSTM nếu TensorFlow đã cài đặt
try:
    from .rnn_model import RNNModel
    from .lstm_model import LSTMModel
    __all__ = ['LinearRegressionModel', 'RNNModel', 'LSTMModel']
except ImportError:
    __all__ = ['LinearRegressionModel']
