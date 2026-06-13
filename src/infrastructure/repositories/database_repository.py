"""
Database Repository for Data Persistence
Follows Repository Pattern from Clean Architecture
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from src.infrastructure.database import (
    Prediction,
    ModelPerformance,
    BacktestResult,
    TradingSignal,
    MarketData,
    SentimentData,
    SystemLog,
    ModelVersion,
    get_session
)


class DatabaseRepository:
    """Repository for database operations"""
    
    def __init__(self, session=None):
        self.session = session if session else get_session()
    
    def save_prediction(self, symbol: str, model_name: str, predicted_price: float,
                       target_date: datetime, confidence: float = None,
                       features_used: Dict = None) -> Prediction:
        """Save a prediction to database"""
        prediction = Prediction(
            symbol=symbol,
            model_name=model_name,
            predicted_price=predicted_price,
            target_date=target_date,
            confidence=confidence,
            features_used=json.dumps(features_used) if features_used else None
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction
    
    def get_predictions(self, symbol: str = None, model_name: str = None,
                       days: int = 30) -> List[Prediction]:
        """Get predictions from database"""
        query = self.session.query(Prediction)
        
        if symbol:
            query = query.filter(Prediction.symbol == symbol)
        if model_name:
            query = query.filter(Prediction.model_name == model_name)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Prediction.prediction_date >= cutoff_date)
        
        return query.order_by(Prediction.prediction_date.desc()).all()
    
    def update_prediction_actual_price(self, prediction_id: int, actual_price: float):
        """Update actual price for a prediction"""
        prediction = self.session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            prediction.actual_price = actual_price
            self.session.commit()
    
    def save_model_performance(self, model_name: str, symbol: str,
                               metric_name: str, metric_value: float,
                               training_samples: int = None) -> ModelPerformance:
        """Save model performance metrics"""
        performance = ModelPerformance(
            model_name=model_name,
            symbol=symbol,
            metric_name=metric_name,
            metric_value=metric_value,
            training_samples=training_samples
        )
        self.session.add(performance)
        self.session.commit()
        return performance
    
    def get_model_performance(self, model_name: str = None,
                             symbol: str = None) -> List[ModelPerformance]:
        """Get model performance metrics"""
        query = self.session.query(ModelPerformance)
        
        if model_name:
            query = query.filter(ModelPerformance.model_name == model_name)
        if symbol:
            query = query.filter(ModelPerformance.symbol == symbol)
        
        return query.order_by(ModelPerformance.evaluation_date.desc()).all()
    
    def save_backtest_result(self, strategy_name: str, symbol: str,
                            start_date: datetime, end_date: datetime,
                            total_return: float, total_trades: int,
                            winning_trades: int, losing_trades: int,
                            win_rate: float, max_drawdown: float,
                            sharpe_ratio: float = None, profit_factor: float = None,
                            avg_win: float = None, avg_loss: float = None) -> BacktestResult:
        """Save backtesting results"""
        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        self.session.add(result)
        self.session.commit()
        return result
    
    def get_backtest_results(self, symbol: str = None) -> List[BacktestResult]:
        """Get backtesting results"""
        query = self.session.query(BacktestResult)
        
        if symbol:
            query = query.filter(BacktestResult.symbol == symbol)
        
        return query.order_by(BacktestResult.created_at.desc()).all()
    
    def save_trading_signal(self, symbol: str, signal_type: str,
                           model_name: str, price_at_signal: float,
                           signal_strength: float = None, target_price: float = None,
                           stop_loss: float = None, expiry_date: datetime = None,
                           notes: str = None) -> TradingSignal:
        """Save trading signal"""
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            signal_strength=signal_strength,
            model_name=model_name,
            price_at_signal=price_at_signal,
            target_price=target_price,
            stop_loss=stop_loss,
            expiry_date=expiry_date,
            notes=notes
        )
        self.session.add(signal)
        self.session.commit()
        return signal
    
    def get_active_signals(self, symbol: str = None) -> List[TradingSignal]:
        """Get active trading signals"""
        query = self.session.query(TradingSignal).filter(TradingSignal.status == 'ACTIVE')
        
        if symbol:
            query = query.filter(TradingSignal.symbol == symbol)
        
        return query.order_by(TradingSignal.signal_date.desc()).all()
    
    def save_market_data(self, symbol: str, timestamp: datetime,
                        open_price: float, high_price: float,
                        low_price: float, close_price: float,
                        volume: float, source: str = 'binance') -> MarketData:
        """Save market data"""
        data = MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            source=source
        )
        self.session.add(data)
        self.session.commit()
        return data
    
    def get_market_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Get market data from database"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(MarketData).filter(
            MarketData.symbol == symbol,
            MarketData.timestamp >= cutoff_date
        ).order_by(MarketData.timestamp.desc()).all()
    
    def save_sentiment_data(self, symbol: str, source: str,
                           sentiment_score: float, sentiment_label: str,
                           content: str = None, url: str = None) -> SentimentData:
        """Save sentiment analysis data"""
        sentiment = SentimentData(
            symbol=symbol,
            source=source,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            content=content,
            url=url
        )
        self.session.add(sentiment)
        self.session.commit()
        return sentiment
    
    def get_sentiment_data(self, symbol: str, days: int = 7) -> List[SentimentData]:
        """Get sentiment data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(SentimentData).filter(
            SentimentData.symbol == symbol,
            SentimentData.data_date >= cutoff_date
        ).order_by(SentimentData.data_date.desc()).all()
    
    def log_system_event(self, log_level: str, message: str,
                        module: str = None, details: str = None) -> SystemLog:
        """Log system event"""
        log = SystemLog(
            log_level=log_level,
            module=module,
            message=message,
            details=details
        )
        self.session.add(log)
        self.session.commit()
        return log
    
    def save_model_version(self, model_name: str, version: str,
                          file_path: str, parameters: Dict = None,
                          performance_metrics: Dict = None) -> ModelVersion:
        """Save model version"""
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            file_path=file_path,
            parameters=json.dumps(parameters) if parameters else None,
            performance_metrics=json.dumps(performance_metrics) if performance_metrics else None,
            training_date=datetime.utcnow()
        )
        self.session.add(model_version)
        self.session.commit()
        return model_version
    
    def get_active_model_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get active model version"""
        return self.session.query(ModelVersion).filter(
            ModelVersion.model_name == model_name,
            ModelVersion.is_active == True
        ).first()
    
    def close(self):
        """Close database session"""
        self.session.close()
