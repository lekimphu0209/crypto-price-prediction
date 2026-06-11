"""
Base Model - Abstract base class for all model implementations
"""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from src.domain.interfaces.model import IModel


class BaseModel(IModel, ABC):
    """Abstract base class for model implementations"""
    
    def __init__(self):
        self._model = None
        self._is_trained = False
        self._name = self.__class__.__name__
    
    @property
    def name(self) -> str:
        """Get model name"""
        return self._name
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def _check_trained(self) -> None:
        """Check if model is trained, raise error if not"""
        if not self._is_trained:
            raise RuntimeError(f"Model {self._name} is not trained yet")
