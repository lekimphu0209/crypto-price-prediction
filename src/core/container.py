"""
Dependency Injection Container
"""
from dependency_injector import containers, providers

from src.core.config import Config, config
from src.infrastructure.models.linear_regression import LinearRegressionModel
from src.infrastructure.repositories.csv_repository import CSVDataRepository
from src.application.use_cases.predict import PredictUseCase


class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Configuration
    configuration = providers.Singleton(lambda: config)
    
    # Repositories
    data_repository = providers.Singleton(
        CSVDataRepository,
        data_dir=configuration.provided.data_dir
    )
    
    # Models
    linear_regression_model = providers.Factory(
        LinearRegressionModel
    )
    
    # Use Cases
    predict_use_case = providers.Factory(
        PredictUseCase,
        models=providers.List(
            linear_regression_model
        ),
        data_repository=data_repository
    )


# Global container instance
container = Container()
