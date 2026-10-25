"""
Composition Root - the single place that wires concrete adapters together.

This is the ONLY module that knows about concrete implementations
(BinanceProvider, TechnicalExtractor, TrendTargetBuilder, FileModelRegistry).
Everything else depends on interfaces. Swap an implementation here to change
the whole system's behaviour.
"""
import os

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.regression_target_builder import RegressionTargetBuilder
from src.infrastructure.persistence.file_registry import FileModelRegistry
from src.training.market_data_loader import MarketDataLoader
from src.training.training_pipeline import TrainingPipeline
from src.training.predict_pipeline import PredictPipeline
from src.market_regime.regime_feature_extractor import RegimeFeatureExtractor

# Default config
SEQ_LEN = 20
HORIZON = 7  # predict up/down trend 7 days ahead
TRAIN_DAYS = 1000  # Increased from 400 for better generalization
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _build_common(extractor=None):
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=TRAIN_DAYS)
    extractor = extractor or TechnicalExtractor()
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
    loader, extractor, default_target_builder, registry = _build_common(extractor=extractor)
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
        extractor=extractor,
        target_builder=target_builder,
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
    base = TechnicalExtractor(feature_set=base_feature_set)
    return RegimeFeatureExtractor(
        base_extractor=base,
        short_window=short_window,
        long_window=long_window,
        slope_window=slope_window,
        onehot=onehot,
        include_label=include_label,
    )


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
