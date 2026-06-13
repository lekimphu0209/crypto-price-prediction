"""
Abstract Database Interface
Allows switching between SQL and MongoDB without changing application code
Based on Dependency Inversion Principle from Clean Architecture
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime


class IDatabaseRepository(ABC):
    """Abstract interface for database operations"""
    
    @abstractmethod
    def save_prediction(self, symbol: str, model_name: str, predicted_price: float,
                       target_date: datetime, confidence: float = None,
                       features_used: Dict = None) -> Any:
        """Save a prediction"""
        pass
    
    @abstractmethod
    def get_predictions(self, symbol: str = None, model_name: str = None,
                       days: int = 30) -> List[Any]:
        """Get predictions"""
        pass
    
    @abstractmethod
    def update_prediction_actual_price(self, prediction_id: int, actual_price: float):
        """Update actual price for prediction"""
        pass
    
    @abstractmethod
    def save_model_performance(self, model_name: str, symbol: str,
                               metric_name: str, metric_value: float,
                               training_samples: int = None) -> Any:
        """Save model performance"""
        pass
    
    @abstractmethod
    def get_model_performance(self, model_name: str = None,
                             symbol: str = None) -> List[Any]:
        """Get model performance"""
        pass
    
    @abstractmethod
    def save_backtest_result(self, strategy_name: str, symbol: str,
                            start_date: datetime, end_date: datetime,
                            total_return: float, total_trades: int,
                            winning_trades: int, losing_trades: int,
                            win_rate: float, max_drawdown: float,
                            sharpe_ratio: float = None, profit_factor: float = None,
                            avg_win: float = None, avg_loss: float = None) -> Any:
        """Save backtest result"""
        pass
    
    @abstractmethod
    def get_backtest_results(self, symbol: str = None) -> List[Any]:
        """Get backtest results"""
        pass
    
    @abstractmethod
    def save_trading_signal(self, symbol: str, signal_type: str,
                           model_name: str, price_at_signal: float,
                           signal_strength: float = None, target_price: float = None,
                           stop_loss: float = None, expiry_date: datetime = None,
                           notes: str = None) -> Any:
        """Save trading signal"""
        pass
    
    @abstractmethod
    def get_active_signals(self, symbol: str = None) -> List[Any]:
        """Get active signals"""
        pass
    
    @abstractmethod
    def save_market_data(self, symbol: str, timestamp: datetime,
                        open_price: float, high_price: float,
                        low_price: float, close_price: float,
                        volume: float, source: str = 'binance') -> Any:
        """Save market data"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str, days: int = 30) -> List[Any]:
        """Get market data"""
        pass
    
    @abstractmethod
    def save_sentiment_data(self, symbol: str, source: str,
                           sentiment_score: float, sentiment_label: str,
                           content: str = None, url: str = None) -> Any:
        """Save sentiment data"""
        pass
    
    @abstractmethod
    def get_sentiment_data(self, symbol: str, days: int = 7) -> List[Any]:
        """Get sentiment data"""
        pass
    
    @abstractmethod
    def log_system_event(self, log_level: str, message: str,
                        module: str = None, details: str = None) -> Any:
        """Log system event"""
        pass
    
    @abstractmethod
    def save_model_version(self, model_name: str, version: str,
                          file_path: str, parameters: Dict = None,
                          performance_metrics: Dict = None) -> Any:
        """Save model version"""
        pass
    
    @abstractmethod
    def get_active_model_version(self, model_name: str) -> Optional[Any]:
        """Get active model version"""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection"""
        pass
