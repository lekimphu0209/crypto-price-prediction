"""
Trade Signal Entity - Trading signal generated from predictions
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """Entity representing a trading signal"""
    
    signal_type: SignalType
    symbol: str
    predicted_price: float
    current_price: float
    confidence: float
    predicted_change_pct: float
    timestamp: datetime
    model_name: str
    
    def __post_init__(self):
        """Validate trade signal"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.current_price <= 0 or self.predicted_price <= 0:
            raise ValueError("Prices must be positive")
    
    @property
    def is_strong_signal(self) -> bool:
        """Is this a strong signal (high confidence)"""
        return self.confidence > 0.7
    
    @property
    def potential_profit_pct(self) -> float:
        """Calculate potential profit percentage"""
        return ((self.predicted_price - self.current_price) / self.current_price) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'predicted_price': self.predicted_price,
            'current_price': self.current_price,
            'confidence': self.confidence,
            'predicted_change_pct': self.predicted_change_pct,
            'potential_profit_pct': self.potential_profit_pct,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name
        }
