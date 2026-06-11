"""
IModelRepository Interface - Abstraction for model storage
"""
from abc import ABC, abstractmethod
from typing import Optional
from src.domain.interfaces.model import IModel


class IModelRepository(ABC):
    """Interface for model repository"""
    
    @abstractmethod
    def save_model(self, model: IModel, name: str) -> None:
        """Save model to repository"""
        pass
    
    @abstractmethod
    def load_model(self, name: str) -> Optional[IModel]:
        """Load model from repository"""
        pass
    
    @abstractmethod
    def model_exists(self, name: str) -> bool:
        """Check if model exists in repository"""
        pass
    
    @abstractmethod
    def list_models(self) -> list[str]:
        """List all models in repository"""
        pass
