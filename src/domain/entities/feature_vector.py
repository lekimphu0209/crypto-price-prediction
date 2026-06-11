"""
Feature Vector Entity - Feature vector for model input
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FeatureVector:
    """Entity representing feature vector for model input"""
    
    features: Dict[str, float]
    timestamp: Optional[str] = None
    symbol: Optional[str] = None
    
    def __post_init__(self):
        """Validate feature vector"""
        if not self.features:
            raise ValueError("Features cannot be empty")
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """
        Convert to numpy array with specific feature order
        
        Args:
            feature_names: List of feature names in desired order
        
        Returns:
            Numpy array of features
        """
        return np.array([self.features.get(name, 0.0) for name in feature_names])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return list(self.features.keys())
    
    @property
    def size(self) -> int:
        """Get number of features"""
        return len(self.features)
