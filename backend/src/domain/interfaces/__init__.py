"""
Domain Interfaces - Abstractions for Infrastructure to implement
"""
from .data_source import IDataSource
from .feature_extractor import IFeatureExtractor
from .target_builder import ITargetBuilder
from .forecaster import IForecaster
from .model_registry import IModelRegistry

__all__ = [
    'IDataSource',
    'IFeatureExtractor',
    'ITargetBuilder',
    'IForecaster',
    'IModelRegistry',
]
