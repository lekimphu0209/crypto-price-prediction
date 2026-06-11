"""
Configuration Management
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration"""
    
    # Data settings
    data_dir: str = "data"
    models_dir: str = "models"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Binance settings
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    
    # NewsAPI settings
    newsapi_key: Optional[str] = None
    
    # Twitter API settings
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    # OpenAI API settings
    openai_api_key: Optional[str] = None
    
    # Model settings
    default_sequence_length: int = 30
    train_test_split: float = 0.8
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            data_dir=os.getenv('DATA_DIR', 'data'),
            models_dir=os.getenv('MODELS_DIR', 'models'),
            api_host=os.getenv('API_HOST', '0.0.0.0'),
            api_port=int(os.getenv('API_PORT', '8000')),
            binance_api_key=os.getenv('BINANCE_API_KEY'),
            binance_api_secret=os.getenv('BINANCE_API_SECRET'),
            newsapi_key=os.getenv('NEWSAPI_KEY', 'e009882997db4c53bc86a3736a349c29'),
            twitter_api_key=os.getenv('TWITTER_API_KEY'),
            twitter_api_secret=os.getenv('TWITTER_API_SECRET'),
            twitter_access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            twitter_access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            default_sequence_length=int(os.getenv('SEQUENCE_LENGTH', '30')),
            train_test_split=float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))
        )


# Global config instance
config = Config.from_env()
