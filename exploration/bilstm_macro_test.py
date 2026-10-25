"""
Thử nghiệm nhỏ: BiLSTM + macro returns (stationary) vs BiLSTM gốc (Feature Set B)
===================================================================================
Câu hỏi: Thêm [gold_change, dxy_change, rate_change] dạng return vào Feature Set B
có cải thiện BiLSTM không?

Quy mô: 5 seeds × 5 folds = 25 mẫu/variant (đủ cho paired t-test)
Baseline: BiLSTM gốc Feature Set B → mean BalAcc = 0.573

Kết quả:
- Nếu diff > 0 và p < 0.05 → macro có ích → mở rộng toàn bộ 4 model
- Nếu diff ≤ 0 hoặc p ≥ 0.05 → không có ích → dừng hướng này
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset
from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON, MODELS_DIR
from src.infrastructure.forecasters.keras_forecaster import KerasForecaster
from src.utils.determinism import set_full_determinism

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
N_FOLDS = 5
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50
SEEDS = list(range(5))   # 5 seeds × 5 folds = 25 mẫu

BILSTM_BASELINE_MEAN = 0.5730  # từ btc_walkforward_bilstm_results.json (10s×5f)

# Feature Set B không macro (loại bỏ level macro, chỉ giữ return)
MACRO_LEVEL_COLS = ["gold", "dxy", "usd_rate"]   # absolute level → loại
MACRO_RETURN_COLS = ["gold_change", "dxy_change", "rate_change"]  # stationary → giữ

# ── BiLSTM config (giống script gốc) ─────────────────────────────────────────
def make_bilstm():
    return KerasForecaster(
        name="BiLSTM",
        epochs=40,
        batch_size=16,
        units=32,
        dropout=0.1,
        learning_rate=0.001,
        early_stopping_patience=0,
        validation_split=0.0,
        mode="classification",
    )


# ── Core walk-forward runner ──────────────────────────────────────────────────
def run_walkforward(feature_names: list, values: np.ndarray, close: np.ndarray,
                    label: str) -> np.ndarray:
    """Chạy walk-forward 5 seeds × 5 folds, trả về mảng balanced_accuracy."""
    window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)
    target_builder = TrendTargetBuilder()

    # Scale toàn bộ 1 lần (fit trên initial window, giống script gốc)
    scaler = RobustScaler()
    scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
    feature_matrix = scaler.transform(values)

    X, base_close, future_close = window.build(feature_matrix, close)
    y = target_builder.build(base_close, future_close)
    mask = ~pd.isna(y)
    X, y = X[mask], y[mask]

    print(f"\n  [{label}] X={X.shape}, n_features={X.shape[2]}")

    all_ba = []
    total = len(SEEDS) * N_FOLDS
    run_idx = 0

    for seed in SEEDS:
        set_full_determinism(seed)
        for fold in range(N_FOLDS):
            run_idx += 1
            train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
            test_end = train_end + TEST_SIZE
            if test_end >= len(X):
                print(f"    [seed{seed} f{fold+1}] skip: not enough data")
                continue

            X_train = X[:train_end]
            X_test  = X[train_end:test_end]
            y_train = y[:train_end]
            y_test  = y[train_end:test_end]

            t0 = time.time()
            try:
                model = make_bilstm()
                model.fit(X_train, y_train)
                proba = np.asarray(model.predict(X_test))
                y_pred = (proba >= 0.5).astype(int)
                ba = balanced_accuracy_score(y_test, y_pred)
                all_ba.append(ba)
                elapsed = time.time() - t0
                print(f"    [{run_idx:2d}/{total}] seed{seed} fold{fold+1}: BalAcc={ba:.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"    [{run_idx:2d}/{total}] seed{seed} fold{fold+1}: ERROR {e}")

    return np.array(all_ba)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BILSTM MACRO RETURN TEST  (5 seeds × 5 folds = 25 mẫu/variant)")
    print(f"Baseline BiLSTM gốc: mean BalAcc = {BILSTM_BASELINE_MEAN:.4f} (n=50)")
    print("=" * 70)

    # Load data một lần duy nhất qua extractor đầy đủ (có macro)
    extractor_full = TechnicalExtractor(feature_set="B")
    pipeline = build_training_pipeline(extractor=extractor_full, seq_len=SEQ_LEN, horizon=HORIZON)
    df = pipeline.loader.load(SYMBOL)
    print(f"\nData: {df.shape[0]} rows | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # ── Variant A: Feature Set B không có macro (baseline nhỏ, 25 mẫu) ──
    print("\n" + "─" * 70)
    print("VARIANT A: Feature Set B — KHÔNG macro (stationary-only baseline)")
    print("─" * 70)
    extractor_no_macro = TechnicalExtractor(
        feature_set="B",
        exclude_features=MACRO_LEVEL_COLS + MACRO_RETURN_COLS,
    )
    feat_no_macro = extractor_no_macro.extract(df)
    names_no_macro = extractor_no_macro.feature_names()
    values_no_macro = feat_no_macro[names_no_macro].values
    close = feat_no_macro["close"].values
    print(f"  Features ({len(names_no_macro)}): {names_no_macro}")

    ba_no_macro = run_walkforward(names_no_macro, values_no_macro, close,
                                  "B_no_macro")

    # ── Variant B: Feature Set B + macro returns (stationary) ──
    print("\n" + "─" * 70)
    print("VARIANT B: Feature Set B + macro RETURNS (gold_change, dxy_change, rate_change)")
    print("─" * 70)
    extractor_ret = TechnicalExtractor(
        feature_set="B",
        exclude_features=MACRO_LEVEL_COLS,   # bỏ absolute level, giữ returns
    )
    feat_ret = extractor_ret.extract(df)
    names_ret = extractor_ret.feature_names()
    values_ret = feat_ret[names_ret].values
    print(f"  Features ({len(names_ret)}): {names_ret}")
    # close không đổi, lấy lại từ feat_ret để đảm bảo align
    close_ret = feat_ret["close"].values

    ba_with_macro = run_walkforward(names_ret, values_ret, close_ret,
                                    "B+macro_returns")

    # ── So sánh ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH")
    print("=" * 70)

    n = min(len(ba_no_macro), len(ba_with_macro))
    if n < 10:
        print(f"  [WARNING] Quá ít mẫu (n={n}) để kết luận")
        return

    print(f"  Variant A (không macro) : n={len(ba_no_macro):2d}, "
          f"mean={ba_no_macro.mean():.4f} ± {ba_no_macro.std():.4f}")
    print(f"  Variant B (+macro ret)  : n={len(ba_with_macro):2d}, "
          f"mean={ba_with_macro.mean():.4f} ± {ba_with_macro.std():.4f}")
    print(f"  Baseline gốc (n=50)     : mean={BILSTM_BASELINE_MEAN:.4f}")

    diff_vs_no_macro = ba_with_macro[:n].mean() - ba_no_macro[:n].mean()
    diff_vs_baseline = ba_with_macro.mean() - BILSTM_BASELINE_MEAN

    # Paired t-test: Variant B vs Variant A
    t_ab, p_ab = stats.ttest_rel(ba_with_macro[:n], ba_no_macro[:n])
    sig_ab = "***" if p_ab < 0.001 else "**" if p_ab < 0.01 else "*" if p_ab < 0.05 else "ns"

    # One-sample vs baseline gốc
    t_base, p_base = stats.ttest_1samp(ba_with_macro, BILSTM_BASELINE_MEAN)

    print(f"\n  Paired t-test (B vs A)  : diff={diff_vs_no_macro:+.4f}, "
          f"t={t_ab:.4f}, p={p_ab:.4f} {sig_ab}")
    print(f"  vs baseline gốc 0.573  : diff={diff_vs_baseline:+.4f}, "
          f"t={t_base:.4f}, p={p_base:.4f}")

    # Lưu kết quả
    out = {
        "experiment": "BiLSTM macro return test",
        "n_seeds": len(SEEDS),
        "n_folds": N_FOLDS,
        "variant_A_no_macro": {
            "n": len(ba_no_macro),
            "mean": float(ba_no_macro.mean()),
            "std": float(ba_no_macro.std()),
            "values": ba_no_macro.tolist(),
        },
        "variant_B_macro_returns": {
            "n": len(ba_with_macro),
            "mean": float(ba_with_macro.mean()),
            "std": float(ba_with_macro.std()),
            "values": ba_with_macro.tolist(),
        },
        "baseline_original_n50": BILSTM_BASELINE_MEAN,
        "paired_ttest_B_vs_A": {"diff": float(diff_vs_no_macro), "t": float(t_ab), "p": float(p_ab)},
        "ttest_vs_baseline": {"diff": float(diff_vs_baseline), "t": float(t_base), "p": float(p_base)},
    }
    out_path = os.path.join(MODELS_DIR, "btc_bilstm_macro_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ── Kết luận ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KẾT LUẬN")
    print("=" * 70)
    if p_ab < 0.05 and diff_vs_no_macro > 0:
        print("  ✓ Macro returns CÓ CẢI THIỆN so với không macro (p<0.05)")
        if diff_vs_baseline > 0:
            print(f"  ✓ Vượt cả baseline gốc 0.573 (+{diff_vs_baseline:.4f})")
            print("  → Khuyến nghị: MỞ RỘNG thử nghiệm với toàn bộ 4 model")
        else:
            print(f"  ~ Nhưng vẫn kém baseline gốc 0.573 ({diff_vs_baseline:.4f})")
            print("  → Khuyến nghị: Xem xét lại cách thêm macro hoặc dừng hướng này")
    elif p_ab < 0.05 and diff_vs_no_macro < 0:
        print(f"  ✗ Macro returns LÀM GIẢM BalAcc ({diff_vs_no_macro:.4f}, p={p_ab:.4f})")
        print("  → Khuyến nghị: DỪNG hướng này, tiết kiệm thời gian")
    else:
        print(f"  ~ Không có ý nghĩa thống kê (diff={diff_vs_no_macro:+.4f}, p={p_ab:.4f} ns)")
        print("  → Khuyến nghị: DỪNG hướng này, macro không mang thêm tín hiệu")
    print("=" * 70)


if __name__ == "__main__":
    main()
