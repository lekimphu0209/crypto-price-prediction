"""
MongoDB Database Repository Implementation
Uses PyMongo for MongoDB
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError


class MongoDatabaseRepository:
    """MongoDB implementation using PyMongo"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database_name: str = "crypto_prediction"):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for collections"""
        # Predictions
        self.db.predictions.create_index([("symbol", ASCENDING), ("prediction_date", DESCENDING)])
        self.db.predictions.create_index([("model_name", ASCENDING)])
        
        # Model Performance
        self.db.model_performance.create_index([("model_name", ASCENDING), ("symbol", ASCENDING)])
        
        # Backtest Results
        self.db.backtest_results.create_index([("symbol", ASCENDING), ("created_at", DESCENDING)])
        
        # Trading Signals
        self.db.trading_signals.create_index([("symbol", ASCENDING), ("status", ASCENDING)])
        
        # Market Data
        self.db.market_data.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
        
        # Sentiment Data
        self.db.sentiment_data.create_index([("symbol", ASCENDING), ("data_date", DESCENDING)])
        
        # System Logs
        self.db.system_logs.create_index([("log_date", DESCENDING)])
        
        # Model Versions
        self.db.model_versions.create_index([("model_name", ASCENDING), ("is_active", ASCENDING)])
    
    def save_prediction(self, symbol: str, model_name: str, predicted_price: float,
                       target_date: datetime, confidence: float = None,
                       features_used: Dict = None) -> str:
        doc = {
            "symbol": symbol,
            "model_name": model_name,
            "predicted_price": predicted_price,
            "actual_price": None,
            "prediction_date": datetime.utcnow(),
            "target_date": target_date,
            "confidence": confidence,
            "features_used": features_used,
            "created_at": datetime.utcnow()
        }
        result = self.db.predictions.insert_one(doc)
        return str(result.inserted_id)
    
    def get_predictions(self, symbol: str = None, model_name: str = None,
                       days: int = 30) -> List[Dict]:
        query = {}
        
        if symbol:
            query["symbol"] = symbol
        if model_name:
            query["model_name"] = model_name
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query["prediction_date"] = {"$gte": cutoff_date}
        
        return list(self.db.predictions.find(query).sort("prediction_date", DESCENDING))
    
    def update_prediction_actual_price(self, prediction_id: str, actual_price: float):
        self.db.predictions.update_one(
            {"_id": prediction_id},
            {"$set": {"actual_price": actual_price}}
        )
    
    def save_model_performance(self, model_name: str, symbol: str,
                               metric_name: str, metric_value: float,
                               training_samples: int = None) -> str:
        doc = {
            "model_name": model_name,
            "symbol": symbol,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "training_samples": training_samples,
            "evaluation_date": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        result = self.db.model_performance.insert_one(doc)
        return str(result.inserted_id)
    
    def get_model_performance(self, model_name: str = None,
                             symbol: str = None) -> List[Dict]:
        query = {}
        
        if model_name:
            query["model_name"] = model_name
        if symbol:
            query["symbol"] = symbol
        
        return list(self.db.model_performance.find(query).sort("evaluation_date", DESCENDING))
    
    def save_backtest_result(self, strategy_name: str, symbol: str,
                            start_date: datetime, end_date: datetime,
                            total_return: float, total_trades: int,
                            winning_trades: int, losing_trades: int,
                            win_rate: float, max_drawdown: float,
                            sharpe_ratio: float = None, profit_factor: float = None,
                            avg_win: float = None, avg_loss: float = None) -> str:
        doc = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "created_at": datetime.utcnow()
        }
        result = self.db.backtest_results.insert_one(doc)
        return str(result.inserted_id)
    
    def get_backtest_results(self, symbol: str = None) -> List[Dict]:
        query = {}
        
        if symbol:
            query["symbol"] = symbol
        
        return list(self.db.backtest_results.find(query).sort("created_at", DESCENDING))
    
    def save_trading_signal(self, symbol: str, signal_type: str,
                           model_name: str, price_at_signal: float,
                           signal_strength: float = None, target_price: float = None,
                           stop_loss: float = None, expiry_date: datetime = None,
                           notes: str = None) -> str:
        doc = {
            "symbol": symbol,
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "model_name": model_name,
            "price_at_signal": price_at_signal,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "signal_date": datetime.utcnow(),
            "expiry_date": expiry_date,
            "status": "ACTIVE",
            "notes": notes,
            "created_at": datetime.utcnow()
        }
        result = self.db.trading_signals.insert_one(doc)
        return str(result.inserted_id)
    
    def get_active_signals(self, symbol: str = None) -> List[Dict]:
        query = {"status": "ACTIVE"}
        
        if symbol:
            query["symbol"] = symbol
        
        return list(self.db.trading_signals.find(query).sort("signal_date", DESCENDING))
    
    def save_market_data(self, symbol: str, timestamp: datetime,
                        open_price: float, high_price: float,
                        low_price: float, close_price: float,
                        volume: float, source: str = 'binance') -> str:
        doc = {
            "symbol": symbol,
            "timestamp": timestamp,
            "open_price": open_price,
            "high_price": high_price,
            "low_price": low_price,
            "close_price": close_price,
            "volume": volume,
            "source": source,
            "created_at": datetime.utcnow()
        }
        result = self.db.market_data.insert_one(doc)
        return str(result.inserted_id)
    
    def get_market_data(self, symbol: str, days: int = 30) -> List[Dict]:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = {
            "symbol": symbol,
            "timestamp": {"$gte": cutoff_date}
        }
        return list(self.db.market_data.find(query).sort("timestamp", DESCENDING))
    
    def save_sentiment_data(self, symbol: str, source: str,
                           sentiment_score: float, sentiment_label: str,
                           content: str = None, url: str = None) -> str:
        doc = {
            "symbol": symbol,
            "source": source,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "content": content,
            "url": url,
            "data_date": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        result = self.db.sentiment_data.insert_one(doc)
        return str(result.inserted_id)
    
    def get_sentiment_data(self, symbol: str, days: int = 7) -> List[Dict]:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = {
            "symbol": symbol,
            "data_date": {"$gte": cutoff_date}
        }
        return list(self.db.sentiment_data.find(query).sort("data_date", DESCENDING))
    
    def log_system_event(self, log_level: str, message: str,
                        module: str = None, details: str = None) -> str:
        doc = {
            "log_level": log_level,
            "module": module,
            "message": message,
            "details": details,
            "log_date": datetime.utcnow()
        }
        result = self.db.system_logs.insert_one(doc)
        return str(result.inserted_id)
    
    def save_model_version(self, model_name: str, version: str,
                          file_path: str, parameters: Dict = None,
                          performance_metrics: Dict = None) -> str:
        doc = {
            "model_name": model_name,
            "version": version,
            "file_path": file_path,
            "parameters": parameters,
            "performance_metrics": performance_metrics,
            "training_date": datetime.utcnow(),
            "is_active": False,
            "created_at": datetime.utcnow()
        }
        result = self.db.model_versions.insert_one(doc)
        return str(result.inserted_id)
    
    def get_active_model_version(self, model_name: str) -> Optional[Dict]:
        return self.db.model_versions.find_one({
            "model_name": model_name,
            "is_active": True
        })
    
    def close(self):
        self.client.close()
