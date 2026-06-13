"""
Database Schema for Crypto Price Prediction Platform
Based on Chapter 2 of Pythonic Quant - Data storage for financial analysis
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class Prediction(Base):
    """Store model predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    model_name = Column(String(50), nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=True)
    features_used = Column(Text, nullable=True)  # JSON string of features
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPerformance(Base):
    """Store model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    symbol = Column(String(10), nullable=False)
    metric_name = Column(String(20), nullable=False)  # RMSE, MAE, MAPE, R2, etc.
    metric_value = Column(Float, nullable=False)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    """Store backtesting results"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_return = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=True)
    avg_win = Column(Float, nullable=True)
    avg_loss = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradingSignal(Base):
    """Store trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    signal_strength = Column(Float, nullable=True)  # 0-1
    model_name = Column(String(50), nullable=False)
    price_at_signal = Column(Float, nullable=False)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    signal_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime, nullable=True)
    status = Column(String(20), default='ACTIVE')  # ACTIVE, EXECUTED, EXPIRED, CANCELLED
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketData(Base):
    """Store historical market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(20), default='binance')  # binance, yahoo, etc.
    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentData(Base):
    """Store sentiment analysis results"""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    source = Column(String(20), nullable=False)  # twitter, reddit, news
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    sentiment_label = Column(String(20), nullable=False)  # positive, negative, neutral
    content = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    data_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """Store system logs and events"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    log_level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR
    module = Column(String(50), nullable=True)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    log_date = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """Track model versions and deployments"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    file_path = Column(String(500), nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string
    training_date = Column(DateTime, nullable=False)
    performance_metrics = Column(Text, nullable=True)  # JSON string
    is_active = Column(Boolean, default=False)
    deployed_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_database(db_url: str = 'sqlite:///crypto_prediction.db'):
    """Initialize database with all tables"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Create database session"""
    Session = sessionmaker(bind=engine)
    return Session()
