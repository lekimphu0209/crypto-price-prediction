"""
Prediction Entity - Prediction result
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Prediction:
    """Entity representing a prediction result"""
    
    symbol: str
    predicted_price: float
    confidence: float
    timestamp: datetime
    model_name: str
    metadata: Optional[Dict] = None
    
    @property
    def is_high_confidence(self) -> bool:
        """Is prediction high confidence (confidence > 0.7)"""
        return self.confidence > 0.7
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'metadata': self.metadata or {}
        }
