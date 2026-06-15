"""
Train Models (FIXED pipeline - NO DATA LEAKAGE)

Uses TrainingPipelineFixed which splits train/test BEFORE scaling.
This prevents data leakage and gives more realistic metrics.

Usage:
    python scripts/train_models_fixed.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import ReturnTargetBuilder
from src.infrastructure.persistence.file_registry import FileModelRegistry
from src.training.market_data_loader import MarketDataLoader
from src.training.training_pipeline_fixed import TrainingPipelineFixed
from src.infrastructure.forecasters.factory import create_forecaster, SUPPORTED_MODELS

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SEQ_LEN = 20
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def build_training_pipeline_fixed():
    """Build the FIXED training pipeline (no data leakage)."""
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=1000)
    extractor = TechnicalExtractor()
    target_builder = ReturnTargetBuilder()
    registry = FileModelRegistry(MODELS_DIR)

    return TrainingPipelineFixed(
        loader=loader,
        extractor=extractor,
        target_builder=target_builder,
        registry=registry,
        seq_len=SEQ_LEN,
        data_dir=DATA_DIR,
    )


def main():
    print("=" * 70)
    print("TRAINING MODELS (FIXED - NO DATA LEAKAGE)")
    print("Split train/test BEFORE scaling to prevent leakage")
    print("=" * 70)

    pipeline = build_training_pipeline_fixed()
    summary = {}

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        summary[symbol] = {}
        for i, model_name in enumerate(SUPPORTED_MODELS):
            forecaster = create_forecaster(model_name)
            # Only save CSV for first model of each symbol
            save_csv = (i == 0)
            result = pipeline.train(symbol, forecaster, model_name=model_name, save_csv=save_csv)
            if result.get("ok"):
                m = result["metrics"]
                summary[symbol][model_name] = m
                print(f"  {model_name:<18} -> R2={m['r2']:.4f}  MAE=${m['mae']:,.2f}")
            else:
                err = result.get("error", "unknown")
                summary[symbol][model_name] = {"error": err}
                print(f"  {model_name:<18} -> FAILED: {err}")

    out_path = os.path.join(MODELS_DIR, "_summary_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
