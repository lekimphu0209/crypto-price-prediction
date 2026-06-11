"""
ModelMetrics Entity - Model performance metrics
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelMetrics:
    """Entity representing model performance metrics"""
    
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared
    direction_accuracy: float  # Direction prediction accuracy
    train_samples: int
    test_samples: int
    trained_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'direction_accuracy': self.direction_accuracy,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'trained_at': self.trained_at
        }
    
    @property
    def is_good_model(self) -> bool:
        """Is model performing well (R² > 0.8)"""
        return self.r2 > 0.8
