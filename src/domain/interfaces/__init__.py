"""
Domain Interfaces - Abstractions for Infrastructure to implement
"""
from .model import IModel
from .data_source import IDataSource
from .data_repository import IDataRepository
from .model_repository import IModelRepository

__all__ = ['IModel', 'IDataSource', 'IDataRepository', 'IModelRepository']
