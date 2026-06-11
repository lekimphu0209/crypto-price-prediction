"""
Realtime Predictor Module
Dự đoán giá theo thời gian thực sử dụng các models đã train
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from feature_engineering import FeatureEngineer


class RealtimePredictor:
    """Class để dự đoán giá theo thời gian thực"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Args:
            symbol: Symbol trading pair
        """
        self.symbol = symbol
        self.feature_engineer = FeatureEngineer()
        
        # Models
        self.lr_model = None
        self.rnn_model = None
        self.lr_scaler = None
        self.rnn_scaler = None
        
        # Feature buffer
        self.feature_buffer = []
        self.max_buffer_size = 100
        
        # Prediction history
        self.prediction_history = []
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load các models đã train"""
        try:
            # Load Linear Regression model
            lr_path = f"models/linear_regression_{self.symbol.replace('USDT', '').lower()}.pkl"
            if os.path.exists(lr_path):
                self.lr_model = joblib.load(lr_path)
                self.lr_scaler = self.lr_model.scaler
                print(f"Đã load Linear Regression model từ {lr_path}")
            
            # Load RNN model
            rnn_path = f"models/rnn_{self.symbol.replace('USDT', '').lower()}.h5"
            scaler_path = f"models/rnn_{self.symbol.replace('USDT', '').lower()}_scaler.pkl"
            
            if os.path.exists(rnn_path):
                self.rnn_model = tf.keras.models.load_model(rnn_path)
                if os.path.exists(scaler_path):
                    self.rnn_scaler = joblib.load(scaler_path)
                print(f"Đã load RNN model từ {rnn_path}")
                
        except Exception as e:
            print(f"Lỗi khi load models: {e}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị features từ dữ liệu OHLCV
        
        Args:
            df: DataFrame với cột timestamp, open, high, low, close, volume
        
        Returns:
            DataFrame với features
        """
        df = df.copy()
        
        # Rename columns nếu cần
        if 'timestamp' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        # Thêm technical indicators
        df = self.feature_engineer.add_technical_indicators(df)
        
        # Thêm lag features
        df = self.feature_engineer.add_lag_features(df)
        
        # Chuẩn bị target (không cần cho prediction nhưng cần để clean)
        df = self.feature_engineer.prepare_target(df)
        
        # Clean data
        df = self.feature_engineer.clean_data(df)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Lấy danh sách feature columns"""
        return self.feature_engineer.get_feature_columns(df)
    
    def predict_linear_regression(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Dự đoán với Linear Regression
        
        Args:
            df: DataFrame với features
        
        Returns:
            Predictions array
        """
        if self.lr_model is None:
            print("Linear Regression model chưa được load")
            return None
        
        try:
            feature_cols = self.get_feature_columns(df)
            X = df[feature_cols].values
            
            # Scale
            X_scaled = self.lr_scaler.transform(X)
            
            # Predict
            predictions = self.lr_model.predict(X_scaled)
            
            return predictions
        except Exception as e:
            print(f"Lỗi khi predict với Linear Regression: {e}")
            return None
    
    def predict_rnn(self, df: pd.DataFrame, sequence_length: int = 30) -> Optional[np.ndarray]:
        """
        Dự đoán với RNN
        
        Args:
            df: DataFrame với features
            sequence_length: Độ dài sequence cho RNN
        
        Returns:
            Predictions array
        """
        if self.rnn_model is None or self.rnn_scaler is None:
            print("RNN model chưa được load")
            return None
        
        try:
            feature_cols = self.get_feature_columns(df)
            X = df[feature_cols].values
            
            # Scale
            X_scaled = self.rnn_scaler.transform(X)
            
            # Create sequences
            if len(X_scaled) < sequence_length:
                print(f"Không đủ dữ liệu để tạo sequence (cần {sequence_length}, có {len(X_scaled)})")
                return None
            
            sequences = []
            for i in range(len(X_scaled) - sequence_length + 1):
                sequences.append(X_scaled[i:i+sequence_length])
            
            if len(sequences) == 0:
                return None
            
            X_seq = np.array(sequences)
            
            # Predict
            predictions = self.rnn_model.predict(X_seq)
            
            # Inverse transform nếu cần
            if predictions.ndim == 2:
                predictions = predictions.flatten()
            
            return predictions
        except Exception as e:
            print(f"Lỗi khi predict với RNN: {e}")
            return None
    
    def predict_ensemble(self, df: pd.DataFrame, sequence_length: int = 30) -> Dict:
        """
        Ensemble predictions từ nhiều models
        
        Args:
            df: DataFrame với features
            sequence_length: Độ dài sequence cho RNN
        
        Returns:
            Dict với predictions từ các models và ensemble
        """
        results = {}
        
        # Linear Regression prediction
        lr_pred = self.predict_linear_regression(df)
        if lr_pred is not None:
            results['linear_regression'] = {
                'predictions': lr_pred,
                'last_prediction': lr_pred[-1],
                'mean': np.mean(lr_pred),
                'std': np.std(lr_pred)
            }
        
        # RNN prediction
        rnn_pred = self.predict_rnn(df, sequence_length)
        if rnn_pred is not None:
            results['rnn'] = {
                'predictions': rnn_pred,
                'last_prediction': rnn_pred[-1],
                'mean': np.mean(rnn_pred),
                'std': np.std(rnn_pred)
            }
        
        # Ensemble (weighted average)
        if 'linear_regression' in results and 'rnn' in results:
            # Linear Regression có R² tốt hơn, nên cho weight cao hơn
            lr_weight = 0.7
            rnn_weight = 0.3
            
            lr_last = results['linear_regression']['last_prediction']
            rnn_last = results['rnn']['last_prediction']
            
            ensemble_pred = lr_weight * lr_last + rnn_weight * rnn_last
            
            results['ensemble'] = {
                'prediction': ensemble_pred,
                'weights': {'linear_regression': lr_weight, 'rnn': rnn_weight}
            }
        elif 'linear_regression' in results:
            results['ensemble'] = {
                'prediction': results['linear_regression']['last_prediction'],
                'weights': {'linear_regression': 1.0, 'rnn': 0.0}
            }
        elif 'rnn' in results:
            results['ensemble'] = {
                'prediction': results['rnn']['last_prediction'],
                'weights': {'linear_regression': 0.0, 'rnn': 1.0}
            }
        
        return results
    
    def predict_realtime(self, candles: pd.DataFrame) -> Dict:
        """
        Dự đoán realtime từ dữ liệu candles mới nhất
        
        Args:
            candles: DataFrame với candles gần nhất
        
        Returns:
            Dict với predictions và metadata
        """
        # Prepare features
        df = self.prepare_features(candles)
        
        if len(df) < 30:
            return {
                'error': f'Không đủ dữ liệu (cần ít nhất 30 candles, có {len(df)})',
                'timestamp': datetime.now()
            }
        
        # Predict
        predictions = self.predict_ensemble(df)
        
        # Add metadata
        result = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'current_price': candles['close'].iloc[-1],
            'predictions': predictions,
            'data_points': len(df)
        }
        
        # Save to history
        self.prediction_history.append(result)
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
        
        return result
    
    def get_prediction_history(self, n: int = 100) -> List[Dict]:
        """Lấy n predictions gần nhất"""
        return self.prediction_history[-n:]
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Lấy prediction gần nhất"""
        if self.prediction_history:
            return self.prediction_history[-1]
        return None


class MultiSymbolPredictor:
    """Predictor cho nhiều symbols cùng lúc"""
    
    def __init__(self, symbols: List[str] = ["BTCUSDT", "ETHUSDT"]):
        """
        Args:
            symbols: List of symbols
        """
        self.predictors = {}
        for symbol in symbols:
            self.predictors[symbol] = RealtimePredictor(symbol)
    
    def predict(self, symbol: str, candles: pd.DataFrame) -> Dict:
        """Predict cho một symbol"""
        if symbol in self.predictors:
            return self.predictors[symbol].predict_realtime(candles)
        else:
            return {'error': f'Symbol {symbol} không được hỗ trợ'}
    
    def predict_all(self, candles_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Predict cho tất cả symbols
        
        Args:
            candles_dict: Dict {symbol: DataFrame}
        
        Returns:
            Dict {symbol: predictions}
        """
        results = {}
        for symbol, candles in candles_dict.items():
            results[symbol] = self.predict(symbol, candles)
        return results


if __name__ == "__main__":
    # Test predictor
    import yfinance as yf
    
    predictor = RealtimePredictor("BTCUSDT")
    
    # Load test data
    btc_data = yf.download("BTC-USD", period="2mo", interval="1d", progress=False)
    btc_data.reset_index(inplace=True)
    btc_data.rename(columns={'Date': 'timestamp'}, inplace=True)
    
    # Predict
    result = predictor.predict_realtime(btc_data)
    
    print("Prediction Result:")
    print(f"Current Price: {result['current_price']}")
    print(f"Ensemble Prediction: {result['predictions']['ensemble']['prediction']}")
    if 'linear_regression' in result['predictions']:
        print(f"LR Prediction: {result['predictions']['linear_regression']['last_prediction']}")
    if 'rnn' in result['predictions']:
        print(f"RNN Prediction: {result['predictions']['rnn']['last_prediction']}")
