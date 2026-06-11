"""
Prediction DTO - Data Transfer Object for predictions
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class PredictionDTO:
    """Data Transfer Object for predictions"""
    
    symbol: str
    current_price: float
    predicted_price: float
    confidence: float
    model_name: str
    timestamp: datetime
    predictions_by_model: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'predictions_by_model': self.predictions_by_model or {},
            'metadata': self.metadata or {}
        }
