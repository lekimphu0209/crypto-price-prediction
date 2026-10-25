"""
Forecaster registry - plugin hub for model implementations.

Replaces the old if/else factory: any class can self-register with
@register_forecaster("Name") and pipelines/scripts only ever call
create_forecaster(name). New models (GRU, XGBoost, LightGBM, ...) plug in
without touching any pipeline code.

Adding a new model:
    1. Create a class implementing IForecaster (fit/predict/save/load/name).
    2. Decorate it: @register_forecaster("GRU")  (for simple no-name classes)
       or append to _NAMED below for classes needing their name (Keras).
    3. Optionally add configs/models/<name>.yaml for hyperparams.
"""
from typing import Callable, Dict

from src.domain.interfaces.forecaster import IForecaster
from src.models.keras_forecaster import KerasForecaster, SUPPORTED as KERAS_MODELS
from src.models.linear_forecaster import LinearForecaster

# name -> callable(**hyperparams) -> IForecaster
_REGISTRY: Dict[str, Callable[..., IForecaster]] = {}


def register_forecaster(name: str, builder: Callable[..., IForecaster] = None):
    """
    Register a forecaster under `name`.

    Can be used as a decorator on classes taking no name argument:

        @register_forecaster("Logistic Regression")
        class LinearForecaster(IForecaster): ...

    or called directly with a factory callable:

        register_forecaster("RNN", lambda **kw: KerasForecaster("RNN", **kw))
    """
    if builder is not None:
        _REGISTRY[name] = builder
        return builder

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_forecaster(model_name: str, **hyperparams) -> IForecaster:
    """
    Create a forecaster for the given registered model name.

    Raises:
        ValueError if the model name is unknown.
    """
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[model_name](**hyperparams)


def registered_models() -> list:
    """Return all registered model names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

register_forecaster("Logistic Regression", LinearForecaster)
for _name in sorted(KERAS_MODELS):
    register_forecaster(_name, (lambda n: (lambda **kw: KerasForecaster(n, **kw)))(_name))

# Backwards-compatible alias: scripts import SUPPORTED_MODELS from the factory.
SUPPORTED_MODELS = ["Logistic Regression", "RNN", "LSTM", "BiLSTM", "Transformer"]
