"""
EDA: Macro Feature Correlation Analysis
========================================
Mục đích: Kiểm tra correlation giữa macro features (Gold, DXY, US 10Y Yield)
và BTC direction target TRƯỚC KHI đưa vào model chính.

Logic:
1. Tải BTC/USDT daily từ Binance (từ ngày đầu tiên có data đến hiện tại)
2. Tải Gold (GC=F), DXY (DX-Y.NYB), US 10Y Yield (^TNX) từ Yahoo Finance
   cùng khoảng thời gian với BTC
3. Tính các biến transform:
   - level: giá tuyệt đối
   - return (pct_change hoặc diff): % thay đổi ngày
   - lag returns: 1, 3, 5, 7 ngày trước
   - rolling correlation: cửa sổ 30, 60, 90 ngày
4. BTC direction target: 1 nếu BTC close tăng so với 7 ngày sau, 0 nếu không
5. Tính Pearson + Spearman correlation
6. Visualize heatmap + rolling correlation chart
7. Kết luận: nếu |corr| < 0.05 → tín hiệu rất yếu

Kết quả được in ra console và lưu chart vào exploration/charts/ (nếu matplotlib có sẵn).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
HORIZON = 7        # BTC direction: giá tăng sau HORIZON ngày?
LAG_DAYS = [0, 1, 3, 5, 7]   # macro lag trước BTC (lag=0 là cùng ngày)
ROLLING_WINDOWS = [30, 60, 90]
BTC_LIMIT = 1000   # số ngày BTC cần lấy (sẽ tự align với macro)
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")

MACRO_TICKERS = {
    "gold":     ["GC=F", "GLD"],
    "dxy":      ["DX-Y.NYB", "DX=F"],
    "us_yield": ["^TNX"],
}


# ── 1. Load BTC data ──────────────────────────────────────────────────────────
def load_btc(limit: int = 1000) -> pd.DataFrame:
    """Tải BTC/USDT daily từ Binance REST API."""
    import urllib.request, json as _json
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol=BTCUSDT&interval=1d&limit={limit}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = _json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Binance API failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_vol", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["date", "close", "high", "low", "volume"]].sort_values("date").reset_index(drop=True)


# ── 2. Load macro data ────────────────────────────────────────────────────────
def load_macro(start_date: str, end_date: str) -> pd.DataFrame:
    """Tải Gold / DXY / US10Y từ Yahoo Finance trong khoảng BTC."""
    print(f"\n[macro] Fetching Gold/DXY/Yield from {start_date} to {end_date}...")
    frames = {}
    for col, tickers in MACRO_TICKERS.items():
        series = None
        for ticker in tickers:
            try:
                m = yf.Ticker(ticker).history(start=start_date, end=end_date)[["Close"]]
                if m is not None and not m.empty:
                    series = m.rename(columns={"Close": col})
                    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                    print(f"  [{col}] OK via {ticker} — {len(series)} rows")
                    break
            except Exception as ex:
                print(f"  [{col}] {ticker} failed: {ex}")
        if series is not None:
            frames[col] = series[col]
        else:
            print(f"  [{col}] WARNING: all tickers failed")

    if not frames:
        return pd.DataFrame()

    macro_df = pd.DataFrame(frames)
    macro_df.index.name = "date"
    macro_df = macro_df.sort_index().ffill()
    macro_df = macro_df.reset_index()
    macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.normalize()
    return macro_df


# ── 3. Build analysis dataframe ───────────────────────────────────────────────
def build_analysis_df(btc: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge BTC + macro, tính returns, target, lags."""
    df = btc.copy()
    df = df.merge(macro, on="date", how="left")

    macro_cols = [c for c in MACRO_TICKERS.keys() if c in df.columns]
    print(f"\n[build] Macro columns available: {macro_cols}")

    # Fill macro level gaps (weekend/holiday → forward fill)
    for col in macro_cols:
        df[col] = df[col].ffill().bfill()

    # Macro returns (% change 1 ngày)
    for col in macro_cols:
        if col == "us_yield":
            df[f"{col}_ret"] = df[col].diff()          # dùng diff vì yield là %, không nhân 100
        else:
            df[f"{col}_ret"] = df[col].pct_change() * 100

    # BTC return
    df["btc_ret"] = df["close"].pct_change() * 100

    # BTC direction target: 1 nếu close[t+HORIZON] > close[t]
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(float)

    # Macro lags
    ret_cols = [f"{c}_ret" for c in macro_cols if f"{c}_ret" in df.columns]
    for lag in LAG_DAYS:
        if lag == 0:
            continue
        for rc in ret_cols:
            df[f"{rc}_lag{lag}"] = df[rc].shift(lag)

    # Chỉ drop rows thiếu target (HORIZON cuối) và macro return (đầu)
    essential = ["close", "target", "btc_ret"] + ret_cols
    df = df.dropna(subset=essential).reset_index(drop=True)
    print(f"[build] Final rows after dropna: {len(df)} | date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── 4. Correlation analysis ───────────────────────────────────────────────────
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Tính Pearson & Spearman corr giữa tất cả macro variants và target."""
    target = df["target"]

    # Tất cả cột macro (level, return, lag)
    macro_cols_all = [c for c in df.columns if any(
        c.startswith(m) for m in list(MACRO_TICKERS.keys()) + ["gold", "dxy", "us_yield"]
    ) and c != "target"]

    rows = []
    for col in macro_cols_all:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        # Drop NaN alignment
        valid = pd.concat([series, target], axis=1).dropna()
        if len(valid) < 30:
            continue
        s, t = valid[col], valid["target"]

        pearson_r, pearson_p = stats.pearsonr(s, t)
        spearman_r, spearman_p = stats.spearmanr(s, t)
        rows.append({
            "feature": col,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "|pearson_r|": round(abs(pearson_r), 4),
            "n": len(valid),
        })

    result = pd.DataFrame(rows).sort_values("|pearson_r|", ascending=False)
    return result


# ── 5. Rolling correlation ────────────────────────────────────────────────────
def rolling_corr_summary(df: pd.DataFrame, key_features: list) -> None:
    """In thống kê rolling correlation theo từng cửa sổ."""
    print("\n" + "─" * 70)
    print("ROLLING CORRELATION (mean ± std của |corr| theo từng cửa sổ)")
    print("─" * 70)
    target = df["target"].astype(float)

    for feat in key_features:
        if feat not in df.columns:
            continue
        series = df[feat].astype(float)
        print(f"\n  [{feat}]")
        for w in ROLLING_WINDOWS:
            rc = series.rolling(w).corr(target).dropna()
            if len(rc) < 5:
                continue
            print(f"    window={w:3d}: mean={rc.mean():+.4f}  std={rc.std():.4f}  "
                  f"pos%={100*(rc > 0).mean():.0f}%  |corr|_mean={rc.abs().mean():.4f}")


# ── 6. Chart ──────────────────────────────────────────────────────────────────
def save_charts(df: pd.DataFrame, corr_df: pd.DataFrame, key_features: list) -> None:
    """Lưu heatmap correlation và rolling corr chart (nếu matplotlib có)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        os.makedirs(CHARTS_DIR, exist_ok=True)

        # 1. Bar chart: correlation magnitude
        top = corr_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["green" if v > 0 else "red" for v in top["pearson_r"]]
        ax.barh(top["feature"], top["pearson_r"], color=colors, alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.05, color="gray", linewidth=0.8, linestyle="--", label="|r|=0.05")
        ax.axvline(-0.05, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"Pearson Correlation: Macro Features vs BTC Direction (horizon={HORIZON}d)")
        ax.set_xlabel("Pearson r")
        ax.legend()
        plt.tight_layout()
        path1 = os.path.join(CHARTS_DIR, "macro_correlation_bar.png")
        fig.savefig(path1, dpi=120)
        plt.close(fig)
        print(f"\n[chart] Saved: {path1}")

        # 2. Rolling correlation line chart for key features
        fig, axes = plt.subplots(len(key_features), 1, figsize=(12, 3 * len(key_features)), sharex=True)
        if len(key_features) == 1:
            axes = [axes]
        target = df["target"].astype(float)
        for ax, feat in zip(axes, key_features):
            if feat not in df.columns:
                continue
            rc60 = df[feat].astype(float).rolling(60).corr(target)
            ax.plot(df["date"], rc60, label=f"{feat} 60d corr", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.axhline(0.1, color="green", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.axhline(-0.1, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_ylabel("Rolling corr")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_ylim(-0.6, 0.6)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.suptitle(f"Rolling 60d Correlation with BTC Direction (horizon={HORIZON}d)")
        plt.tight_layout()
        path2 = os.path.join(CHARTS_DIR, "macro_rolling_correlation.png")
        fig.savefig(path2, dpi=120)
        plt.close(fig)
        print(f"[chart] Saved: {path2}")

    except ImportError:
        print("\n[chart] matplotlib not available — skipping charts")


# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MACRO CORRELATION ANALYSIS")
    print(f"BTC direction target: horizon={HORIZON} ngày | Lags thử: {LAG_DAYS}")
    print("=" * 70)

    # Load BTC
    print("\n[1/4] Loading BTC data from Binance...")
    btc = load_btc(limit=BTC_LIMIT)
    if btc.empty:
        print("[ERROR] Cannot load BTC data. Abort.")
        return
    start_date = btc["date"].min().strftime("%Y-%m-%d")
    end_date   = btc["date"].max().strftime("%Y-%m-%d")
    print(f"  BTC: {len(btc)} rows | {start_date} → {end_date}")

    # Load macro
    print("\n[2/4] Loading macro data (Gold / DXY / US10Y Yield)...")
    macro = load_macro(start_date, end_date)
    if macro.empty:
        print("[ERROR] Cannot load macro data. Abort.")
        return

    # Build analysis df
    print("\n[3/4] Building analysis dataframe...")
    df = build_analysis_df(btc, macro)
    if len(df) < 50:
        print("[ERROR] Not enough aligned rows. Abort.")
        return

    # Correlation analysis
    print("\n[4/4] Computing correlations...")
    corr_df = correlation_analysis(df)

    # ── Print results ──
    print("\n" + "=" * 70)
    print(f"PEARSON & SPEARMAN CORRELATION với BTC Direction (horizon={HORIZON}d)")
    print("Sắp xếp theo |pearson_r| giảm dần")
    print("=" * 70)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(corr_df.to_string(index=False))

    # ── Rolling corr summary ──
    key_features = [c for c in ["gold_ret", "dxy_ret", "us_yield_ret"] if c in df.columns]
    rolling_corr_summary(df, key_features)

    # ── Save charts ──
    save_charts(df, corr_df, key_features)

    # ── Final interpretation ──
    print("\n" + "=" * 70)
    print("KẾT LUẬN")
    print("=" * 70)
    threshold = 0.05
    strong = corr_df[corr_df["|pearson_r|"] >= threshold]
    weak   = corr_df[corr_df["|pearson_r|"] < threshold]

    if strong.empty:
        print(f"  → TẤT CẢ macro features có |pearson_r| < {threshold}")
        print("  → Tín hiệu từ Gold/DXY/Yield với BTC direction RẤT YẾU")
        print("  → Khuyến nghị: KHÔNG kỳ vọng macro features cải thiện accuracy đáng kể")
    else:
        print(f"  → Có {len(strong)} feature(s) có |pearson_r| ≥ {threshold}:")
        for _, row in strong.iterrows():
            sig = "(*)" if row["pearson_p"] < 0.05 else ""
            print(f"      {row['feature']:30s}: r={row['pearson_r']:+.4f}  p={row['pearson_p']:.4f} {sig}")
        if len(weak) > 0:
            print(f"  → {len(weak)} feature(s) còn lại có |r| < {threshold} — yếu, có thể bỏ qua")

    print("\n  Ghi chú thêm:")
    print("  - Correlation tuyến tính thấp ≠ không có quan hệ (có thể phi tuyến)")
    print("  - Nên thử XGBoost / feature importance để đánh giá phi tuyến")
    print("  - Macro lag 1-7 ngày thường phù hợp hơn cùng ngày (causal)")
    print("=" * 70)


if __name__ == "__main__":
    main()
