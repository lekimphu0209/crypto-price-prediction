"""Check number of training samples before and after applying 1% threshold."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.training.composition import build_training_pipeline, SEQ_LEN, HORIZON
from src.features.technical_extractor import TechnicalExtractor
from src.training.return_target_builder import TrendTargetBuilder
from src.training.window_dataset import WindowDataset

SYMBOL = "BTCUSDT"
FEATURE_SET = "B"
INITIAL_TRAIN_SIZE = 200
TEST_SIZE = 50

extractor = TechnicalExtractor(feature_set=FEATURE_SET)
window = WindowDataset(seq_len=SEQ_LEN, target_horizon=HORIZON)
pipeline = build_training_pipeline(extractor=extractor, seq_len=SEQ_LEN, horizon=HORIZON)

df = pipeline.loader.load(SYMBOL)
feat_df = pipeline.extractor.extract(df)
feature_names = pipeline.extractor.feature_names()

values = feat_df[feature_names].values
close = feat_df["close"].values
scaler = RobustScaler()
scaler.fit(values[:INITIAL_TRAIN_SIZE + TEST_SIZE])
feature_matrix = scaler.transform(values)

X, base_close, future_close = window.build(feature_matrix, close)

# Original labels (threshold=0)
y_original = TrendTargetBuilder(threshold=0.0).build(base_close, future_close)
mask_original = ~pd.isna(y_original)

# Threshold labels (1%)
y_threshold = TrendTargetBuilder(threshold=0.01).build(base_close, future_close)
mask_threshold = ~pd.isna(y_threshold)

print("=" * 80)
print("SAMPLE COUNTS: BEFORE vs AFTER 1% THRESHOLD")
print("=" * 80)
print(f"Trước threshold: {mask_original.sum()} mẫu")
print(f"Sau threshold 1%: {mask_threshold.sum()} mẫu")
print(f"Tỷ lệ loại bỏ: {1 - mask_threshold.sum() / mask_original.sum():.1%}")

print("\nChi tiết từng fold:")
for fold in range(5):
    train_end = INITIAL_TRAIN_SIZE + fold * TEST_SIZE
    test_start = train_end
    test_end = test_start + TEST_SIZE

    y_train_original = y_original[:train_end]
    y_train_threshold = y_threshold[:train_end]

    original_count = (~pd.isna(y_train_original)).sum()
    threshold_count = (~pd.isna(y_train_threshold)).sum()
    removed_ratio = 1 - threshold_count / original_count

    print(f"  Fold {fold+1}: train_end={train_end} | Trước={original_count} | Sau={threshold_count} | Loại bỏ={removed_ratio:.1%}")
