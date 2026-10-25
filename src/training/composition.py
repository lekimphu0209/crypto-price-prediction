"""
Composition Root - the single place that wires concrete adapters together.

This is the ONLY module that knows about concrete implementations
(BinanceProvider, extractor registry, target builders, FileModelRegistry).
Everything else depends on interfaces. Swap an implementation here to change
the whole system's behaviour.

Pipeline-level defaults live in configs/default.yaml; model hyperparams live
in configs/models/<name>.yaml so experiments don't require code edits.
"""
import os

from src.config import load_config
from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.features.extractors.registry import build_extractor
from src.training.targets.trend_target import TrendTargetBuilder
from src.training.targets.regression_target import RegressionTargetBuilder
from src.infrastructure.persistence.file_registry import FileModelRegistry
from src.training.datasets.market_data_loader import MarketDataLoader
from src.training.pipelines.train_pipeline import TrainingPipeline
from src.training.pipelines.predict_pipeline import PredictPipeline
from src.features.market_regime.regime_feature_extractor import RegimeFeatureExtractor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_CONFIG = load_config("default")
SEQ_LEN = int(_CONFIG.get("seq_len", 20))
HORIZON = int(_CONFIG.get("horizon", 7))  # predict up/down trend N days ahead
TRAIN_DAYS = int(_CONFIG.get("train_days", 1000))
DEFAULT_EXTRACTOR = _CONFIG.get(
    "extractor", {"name": "technical", "params": {"feature_set": "A"}}
)


def _build_common(extractor=None):
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=TRAIN_DAYS)
    extractor = extractor or build_extractor(DEFAULT_EXTRACTOR)
    target_builder = TrendTargetBuilder()
    registry = FileModelRegistry(MODELS_DIR)
    return loader, extractor, target_builder, registry


def build_training_pipeline(
    extractor=None,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    threshold_metric: str = "balanced_accuracy",
    mode: str = "classification",
    target_builder=None,
) -> TrainingPipeline:
    loader, extractor, default_target_builder, registry = _build_common(
        extractor=extractor
    )
    if mode == "regression" and target_builder is None:
        target_builder = RegressionTargetBuilder()
    else:
        target_builder = target_builder or default_target_builder
    return TrainingPipeline(
        loader=loader,
        extractor=extractor,
        target_builder=target_builder,
        registry=registry,
        seq_len=seq_len,
        data_dir=DATA_DIR,
        target_horizon=horizon,
        threshold_metric=threshold_metric,
        mode=mode,
    )


def build_predict_pipeline(extractor=None) -> PredictPipeline:
    loader, extractor, target_builder, registry = _build_common(extractor=extractor)
    return PredictPipeline(
        loader=loader,
        extractor=extractor,  # fallback for legacy bundles without extractor_config
        registry=registry,
    )


# Shared singleton predict pipeline (keeps the data cache warm across requests)
_predict_pipeline = None


def get_predict_pipeline() -> PredictPipeline:
    global _predict_pipeline
    if _predict_pipeline is None:
        _predict_pipeline = build_predict_pipeline()
    return _predict_pipeline


# ---------------------------------------------------------------------------
# Regime-aware composition helpers
# ---------------------------------------------------------------------------

def build_regime_extractor(
    base_feature_set: str = "E",
    short_window: int = 7,
    long_window: int = 30,
    slope_window: int = 3,
    onehot: bool = True,
    include_label: bool = False,
) -> RegimeFeatureExtractor:
    """Build a TechnicalExtractor + MA-regime wrapper."""
    return build_extractor({
        "name": "regime",
        "params": {
            "base": {"name": "technical", "params": {"feature_set": base_feature_set}},
            "short_window": short_window,
            "long_window": long_window,
            "slope_window": slope_window,
            "onehot": onehot,
            "include_label": include_label,
        },
    })


def build_training_pipeline_with_regime(
    base_feature_set: str = "E",
    short_window: int = 7,
    long_window: int = 30,
    slope_window: int = 3,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    threshold_metric: str = "balanced_accuracy",
    mode: str = "classification",
    target_builder=None,
) -> TrainingPipeline:
    """Training pipeline that includes MA-regime features."""
    extractor = build_regime_extractor(
        base_feature_set=base_feature_set,
        short_window=short_window,
        long_window=long_window,
        slope_window=slope_window,
        onehot=True,
        include_label=False,
    )
    return build_training_pipeline(
        extractor=extractor,
        seq_len=seq_len,
        horizon=horizon,
        threshold_metric=threshold_metric,
        mode=mode,
        target_builder=target_builder,
    )


def build_predict_pipeline_with_regime(
    base_feature_set: str = "E",
    short_window: int = 7,
    long_window: int = 30,
    slope_window: int = 3,
) -> PredictPipeline:
    """Prediction pipeline that includes MA-regime features."""
    extractor = build_regime_extractor(
        base_feature_set=base_feature_set,
        short_window=short_window,
        long_window=long_window,
        slope_window=slope_window,
        onehot=True,
        include_label=False,
    )
    return build_predict_pipeline(extractor=extractor)
