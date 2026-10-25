import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.market_regime.regime_detector import (
    add_regime_features,
    regime_summary,
    segment_regimes,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "btc_1d.csv",
)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports",
)


def regime_color(label: str) -> str:
    return {
        REGIME_BULL: "green",
        REGIME_BEAR: "red",
        REGIME_SIDEWAYS: "gray",
    }.get(label, "blue")


def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Default regime parameters used by RegimeFeatureExtractor
    short_window = 7
    long_window = 30
    slope_window = 3

    df = add_regime_features(
        df,
        close_col="close",
        short_window=short_window,
        long_window=long_window,
        slope_window=slope_window,
        onehot=False,
    )

    regime = df["ma_regime"]
    print("\n" + "=" * 80)
    print("REGIME SUMMARY (per bar)")
    print("=" * 80)
    summary = regime_summary(regime)
    print(summary.to_string(index=False))

    segments = segment_regimes(regime)
    print("\n" + "=" * 80)
    print("SEGMENT STATISTICS")
    print("=" * 80)
    total_bars = len(df)
    rows = []
    for label in (REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR):
        seg_lens = [s.length for s in segments if s.label == label]
        count = len(seg_lens)
        total = sum(seg_lens)
        rows.append(
            {
                "regime": label,
                "n_segments": count,
                "total_bars": total,
                "pct_bars": total / total_bars * 100,
                "mean_length": sum(seg_lens) / count if count else 0,
                "max_length": max(seg_lens) if seg_lens else 0,
                "min_length": min(seg_lens) if seg_lens else 0,
            }
        )
    seg_df = pd.DataFrame(rows)
    print(seg_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TRANSITIONS (tổng số đoạn)")
    print("=" * 80)
    print(f"Tổng số đoạn liên tiếp: {len(segments)}")
    transitions = pd.Series([s.label for s in segments]).value_counts()
    print(transitions.to_string())

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save summary JSON
    report_path = os.path.join(REPORTS_DIR, "btcusdt_regime_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_bar_summary": summary.to_dict(orient="records"),
                "segment_summary": seg_df.to_dict(orient="records"),
                "n_segments": len(segments),
                "total_bars": total_bars,
            },
            f,
            indent=2,
        )

    # Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}
    )

    ts = df["timestamp"]
    close = df["close"]
    ma_short_col = f"ma_{short_window}"
    ma_long_col = f"ma_{long_window}"

    # Main price + MAs
    ax1.plot(ts, close, label="Close", color="black", linewidth=0.8)
    ax1.plot(ts, df[ma_short_col], label=f"MA {short_window}", color="blue", linewidth=0.7)
    ax1.plot(ts, df[ma_long_col], label=f"MA {long_window}", color="orange", linewidth=0.7)

    # Color background per segment
    for seg in segments:
        ax1.axvspan(ts.iloc[seg.start_idx], ts.iloc[seg.end_idx], alpha=0.15, color=regime_color(seg.label))

    ax1.set_ylabel("Price (USD)")
    ax1.set_title("BTC 1D Regime Segmentation (Bull=green, Bear=red, Sideways=gray)")
    ax1.legend(loc="upper left")

    # Regime ribbon
    regime_map = {REGIME_BULL: 1, REGIME_SIDEWAYS: 0, REGIME_BEAR: -1}
    ribbon = regime.map(regime_map)
    ax2.fill_between(ts, -1, ribbon, where=(ribbon == 1), color="green", alpha=0.6, label="Bull")
    ax2.fill_between(ts, -1, ribbon, where=(ribbon == -1), color="red", alpha=0.6, label="Bear")
    ax2.fill_between(ts, -1, ribbon, where=(ribbon == 0), color="gray", alpha=0.6, label="Sideways")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Bear", "Sideways", "Bull"])
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", ncol=3)

    fig.tight_layout()
    img_path = os.path.join(REPORTS_DIR, "btcusdt_regime_analysis.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nReport saved: {report_path}")
    print(f"Image saved: {img_path}")


if __name__ == "__main__":
    main()
