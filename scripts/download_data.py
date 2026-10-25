"""
Download and save all raw data (BTC OHLCV + Macro) to a single CSV file.

Output: data/btc_full_dataset.csv
Columns: timestamp, open, high, low, close, volume, gold, dxy, usd_rate
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from src.infrastructure.data_providers.binance_provider import BinanceProvider
from src.training.datasets.market_data_loader import MarketDataLoader

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "btc_full_dataset.csv",
)

MACRO_TICKERS = {
    "gold": ["GC=F", "GLD"],
    "dxy": ["DX-Y.NYB", "DX=F"],
    "usd_rate": ["^TNX"],
}


def fetch_btc(limit: int = 1000) -> pd.DataFrame:
    print(f"[BTC] Fetching {limit} daily candles from Binance...")
    loader = MarketDataLoader(BinanceProvider(), interval="1d", limit=limit)
    df = loader.load("BTCUSDT")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    print(f"[BTC] Got {len(df)} rows: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def fetch_macro() -> pd.DataFrame:
    frames = {}
    for col, tickers in MACRO_TICKERS.items():
        series = None
        for ticker in tickers:
            try:
                m = yf.Ticker(ticker).history(period="5y")[["Close"]]
            except Exception:
                m = None
            if m is not None and not m.empty:
                series = m.rename(columns={"Close": col})
                series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                print(f"[Macro] {col:8s} → fetched via {ticker}, {len(series)} rows")
                break
        if series is not None:
            frames[col] = series
        else:
            print(f"[Macro] {col:8s} → FAILED to fetch")

    if not frames:
        print("[Macro] WARNING: no macro data fetched")
        return pd.DataFrame(columns=["date", "gold", "dxy", "usd_rate"])

    out = None
    for col, s in frames.items():
        out = s if out is None else out.join(s, how="outer")
    out = out.sort_index().ffill().reset_index()
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def main():
    btc = fetch_btc(limit=1000)

    macro = fetch_macro()

    btc["date"] = btc["timestamp"]
    merged = btc.merge(macro, on="date", how="left")
    merged.drop(columns=["date"], inplace=True)

    for col in ("gold", "dxy", "usd_rate"):
        if col not in merged.columns:
            merged[col] = None
        else:
            merged[col] = merged[col].ffill().bfill()

    cols = ["timestamp", "open", "high", "low", "close", "volume", "gold", "dxy", "usd_rate"]
    merged = merged[cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[Done] Saved {len(merged)} rows to: {OUTPUT_PATH}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"  Date range: {merged['timestamp'].min()} → {merged['timestamp'].max()}")
    print(f"  Missing values:\n{merged.isnull().sum()}")


if __name__ == "__main__":
    main()
