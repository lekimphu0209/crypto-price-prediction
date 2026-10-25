"""
Extractor registry - maps a named feature extractor to its builder.

Any feature extractor can be registered with @register_extractor(name) so the
training and prediction pipelines (and model meta.json) can refer to it by
name + params instead of hard-coding classes.

Adding a new feature set:
    1. Implement IFeatureExtractor (extract + feature_names + to_config).
    2. Register a builder below.
    3. meta.json automatically stores {"name": ..., "params": ...} via to_config().
"""
from typing import Any, Callable, Dict

from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.features.extractors.technical import TechnicalExtractor
from src.features.market_regime.regime_feature_extractor import RegimeFeatureExtractor

# name -> callable(params: dict) -> IFeatureExtractor
_REGISTRY: Dict[str, Callable[[dict], IFeatureExtractor]] = {}


def register_extractor(name: str, builder: Callable[[dict], IFeatureExtractor]) -> None:
    """Register a builder that constructs an extractor from a params dict."""
    _REGISTRY[name] = builder


def build_extractor(config: dict) -> IFeatureExtractor:
    """
    Build a feature extractor from a serialized config.

    Args:
        config: {"name": <registered name>, "params": {...}}

    Raises:
        ValueError if the extractor name is unknown.
    """
    name = (config or {}).get("name", "technical")
    params = (config or {}).get("params", {})
    if name not in _REGISTRY:
        raise ValueError(f"Unknown extractor: {name}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](params)


def extractor_names() -> list:
    """Return the list of registered extractor names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------

def _build_technical(params: dict) -> TechnicalExtractor:
    return TechnicalExtractor(**params)


def _build_regime(params: dict) -> RegimeFeatureExtractor:
    params = dict(params)
    base_cfg = params.pop("base", {"name": "technical", "params": {"feature_set": "E"}})
    base = build_extractor(base_cfg) if isinstance(base_cfg, dict) else TechnicalExtractor(feature_set="E")
    return RegimeFeatureExtractor(base_extractor=base, **params)


register_extractor("technical", _build_technical)
register_extractor("regime", _build_regime)
