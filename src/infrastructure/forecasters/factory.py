"""
ForecasterFactory - creates IForecaster instances by model name.

Centralizes the mapping from a model name to its concrete implementation,
so the rest of the system never imports concrete forecasters directly.
"""
from src.domain.interfaces.forecaster import IForecaster
from src.infrastructure.forecasters.linear_forecaster import LinearForecaster
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster, SUPPORTED as KERAS_MODELS

# All model names the system supports
SUPPORTED_MODELS = ["Linear Regression", "RNN", "LSTM", "BiLSTM", "Transformer"]


def create_forecaster(model_name: str) -> IForecaster:
    """
    Create a forecaster for the given model name.

    Raises:
        ValueError if the model name is unknown.
    """
    if model_name == "Linear Regression":
        return LinearForecaster()
    if model_name in KERAS_MODELS:
        return KerasForecaster(model_name)
    raise ValueError(f"Unknown model: {model_name}")
