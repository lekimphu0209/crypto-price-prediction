"""
Sentiment Entity - Sentiment data from various sources
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class SentimentSource(Enum):
    """Sentiment source types"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    X_COM = "x_com"


@dataclass
class Sentiment:
    """Entity representing sentiment data"""
    
    source: SentimentSource
    score: float  # -1.0 to 1.0 (negative to positive)
    positive_ratio: float  # 0.0 to 1.0
    negative_ratio: float  # 0.0 to 1.0
    volume: int  # Number of posts/tweets
    timestamp: datetime
    symbol: str
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate sentiment data"""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("Score must be between -1.0 and 1.0")
        if not 0.0 <= self.positive_ratio <= 1.0:
            raise ValueError("Positive ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.negative_ratio <= 1.0:
            raise ValueError("Negative ratio must be between 0.0 and 1.0")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def is_bullish(self) -> bool:
        """Is sentiment bullish (positive)"""
        return self.score > 0
    
    @property
    def is_bearish(self) -> bool:
        """Is sentiment bearish (negative)"""
        return self.score < 0
    
    @property
    def strength(self) -> str:
        """Get sentiment strength"""
        abs_score = abs(self.score)
        if abs_score > 0.7:
            return "strong"
        elif abs_score > 0.3:
            return "moderate"
        else:
            return "weak"
