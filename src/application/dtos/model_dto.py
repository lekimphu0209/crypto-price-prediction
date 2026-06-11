"""
Model DTO - Data Transfer Object for models
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelDTO:
    """Data Transfer Object for models"""
    
    name: str
    type: str
    is_trained: bool
    metrics: Optional[Dict[str, float]] = None
    trained_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.type,
            'is_trained': self.is_trained,
            'metrics': self.metrics or {},
            'trained_at': self.trained_at
        }
