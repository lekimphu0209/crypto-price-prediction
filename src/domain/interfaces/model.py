"""
IModel Interface - Abstraction for all ML models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class IModel(ABC):
    """Interface for all ML models"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get model name"""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model is trained"""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model with data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from file"""
        pass
