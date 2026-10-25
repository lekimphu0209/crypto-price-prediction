"""
Demo / utility script: segment market regimes using MA.

Fetches recent daily data for a symbol, labels each day as bull / sideways / bear
from short/long moving averages, prints a summary, and saves the segmented CSV.

Usage:
    python scripts/detect_regimes.py BTCUSDT
    python scripts/detect_regimes.py ETHUSDT --output data/regimes.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.training.market_data_loader import MarketDataLoader
from src.market_regime.regime_detector import (
    add_regime_features,
    segment_regimes,
    regime_summary,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
)


def main():
    parser = argparse.ArgumentParser(description="Detect market regimes from MA.")
    parser.add_argument("symbol", default="BTCUSDT", nargs="?", help="Symbol to load.")
    parser.add_argument("--days", type=int, default=1000, help="Number of daily candles.")
    parser.add_argument("--short", type=int, default=7, help="Short MA window.")
    parser.add_argument("--long", type=int, default=30, help="Long MA window.")
    parser.add_argument("--slope", type=int, default=3, help="Slope lookback window.")
    parser.add_argument("--output", type=str, default=None, help="CSV output path.")
    args = parser.parse_args()

    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=args.days)
    df = loader.load(args.symbol)
    print(f"Loaded {len(df)} daily rows for {args.symbol}")

    df = add_regime_features(
        df,
        close_col="close",
        short_window=args.short,
        long_window=args.long,
        slope_window=args.slope,
        onehot=True,
    )

    segments = segment_regimes(df["ma_regime"])
    print(f"\nDetected {len(segments)} contiguous regime segments:")
    for seg in segments[-10:]:
        print(
            f"  {seg.label:10s} idx={seg.start_idx:5d}..{seg.end_idx:<5d} "
            f"len={seg.length:3d}"
        )

    print("\nRegime summary:")
    print(regime_summary(df["ma_regime"]).to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_cols = ["timestamp", "close", f"ma_{args.short}", f"ma_{args.long}", "ma_regime"]
        df[out_cols].to_csv(args.output, index=False)
        print(f"\nSaved regime labels to {args.output}")


if __name__ == "__main__":
    main()
