"""
Composition Root - the single place that wires concrete adapters together.

This is the ONLY module that knows about concrete implementations
(BinanceProvider, TechnicalExtractor, ReturnTargetBuilder, FileModelRegistry).
Everything else depends on interfaces. Swap an implementation here to change
the whole system's behaviour.
"""
import os

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import ReturnTargetBuilder
from src.infrastructure.persistence.file_registry import FileModelRegistry
from src.training.market_data_loader import MarketDataLoader
from src.training.training_pipeline import TrainingPipeline
from src.training.predict_pipeline import PredictPipeline

# Default config
SEQ_LEN = 20
TRAIN_DAYS = 1000  # Increased from 400 for better generalization
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _build_common():
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=TRAIN_DAYS)
    extractor = TechnicalExtractor()
    target_builder = ReturnTargetBuilder()
    registry = FileModelRegistry(MODELS_DIR)
    return loader, extractor, target_builder, registry


def build_training_pipeline() -> TrainingPipeline:
    loader, extractor, target_builder, registry = _build_common()
    return TrainingPipeline(
        loader=loader,
        extractor=extractor,
        target_builder=target_builder,
        registry=registry,
        seq_len=SEQ_LEN,
        data_dir=DATA_DIR,
    )


def build_predict_pipeline() -> PredictPipeline:
    loader, extractor, target_builder, registry = _build_common()
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
