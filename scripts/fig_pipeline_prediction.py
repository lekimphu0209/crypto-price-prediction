"""
Generate a pipeline diagram: raw data -> prediction.

Output: docs/figures/pipeline_prediction.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def draw_box(ax, x, y, w, h, text, color="#E8F4FD", fontsize=9, text_color="#000000"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        wrap=True,
        linespacing=1.2,
    )
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def main():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(5, 11.4, "End-to-End Pipeline: Raw Data → Prediction", fontsize=14, ha="center", weight="bold")

    # Stages top to bottom
    draw_box(ax, 5, 10.2, 4.0, 0.7, "Binance API\nBTCUSDT 1d × 1000 candles", color="#D1ECF1")
    draw_box(ax, 5, 9.0, 4.0, 0.7, "OHLCV DataFrame\n(timestamp, open, high, low, close, volume)", color="#D4EDDA")
    draw_box(ax, 5, 7.8, 4.0, 0.7, "TechnicalExtractor (Feature Set B)\n26 features + macro (gold, DXY, rate)", color="#FFF3CD")
    draw_box(ax, 5, 6.6, 4.0, 0.7, "RobustScaler\nfit on train, transform all", color="#F8D7DA")
    draw_box(ax, 5, 5.4, 4.0, 0.7, "Sliding Window\n(seq_len=20, n_features=26)", color="#E2E3F3")

    # Model branch
    draw_box(ax, 5, 4.0, 4.0, 0.9, "KerasForecaster\nRNN / LSTM / BiLSTM / Transformer\nunits=32, dropout=0.1, sigmoid output", color="#D1ECF1")
    draw_box(ax, 5, 2.7, 3.0, 0.7, "Probability of UP trend\np ∈ [0, 1]", color="#FFF3CD")
    draw_box(ax, 5, 1.7, 3.0, 0.7, "Direction & Price Reconstruction\nUP if p ≥ 0.5 / next-day price", color="#D4EDDA")
    draw_box(ax, 5, 0.7, 3.6, 0.7, "7-Day Forecast\ndamped trend extension", color="#F8D7DA")

    # Arrows
    draw_arrow(ax, 5, 9.85, 5, 9.35)
    draw_arrow(ax, 5, 8.65, 5, 8.15)
    draw_arrow(ax, 5, 7.45, 5, 7.05)
    draw_arrow(ax, 5, 6.25, 5, 6.05)
    # side annotation: save model bundle
    draw_box(ax, 8.5, 5.4, 2.4, 0.6, "Model Bundle\n.keras + scaler.pkl\n+ meta.json", color="#E2E3F3", fontsize=8)
    draw_arrow(ax, 7.0, 5.4, 7.5, 5.4, color="#888888")

    draw_arrow(ax, 5, 4.9, 5, 4.55)
    draw_arrow(ax, 5, 3.85, 5, 3.35)
    draw_arrow(ax, 5, 3.05, 5, 2.35)
    draw_arrow(ax, 5, 2.05, 5, 1.35)
    draw_arrow(ax, 5, 1.05, 5, 1.05)

    # API response annotation
    draw_box(ax, 1.5, 0.7, 2.0, 0.6, "API Response\nforecast_dates\nforecast_values", color="#E8F4FD", fontsize=8)
    draw_arrow(ax, 3.0, 0.7, 3.2, 0.7, color="#888888")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pipeline_prediction.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
