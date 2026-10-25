# Nghiên cứu mô hình dự đoán xu hướng giá crypto

> Nguồn: ACM/IEEE/Springer/arXiv + benchmark Monash & GIFT-Eval (2024–2025).
> Bài toán của project: **phân loại hướng xu hướng 7 ngày** (UP/DOWN) trên dữ liệu
> ngày BTC/ETH, seq_len=20, ~1000 ngày dữ liệu. Ưu tiên mô hình có bằng chứng
> đúng với task classification hướng giá — không phải forecast giá nói chung.

## 1. Bằng chứng từ nghiên cứu

### Direction classification (đúng task của project)

| Mô hình | Bằng chứng | Nguồn |
|---|---|---|
| **CNN-LSTM + Boruta** | 82.44% direction accuracy — cao nhất | Springer *Financial Innovation* 2024 (s40854-024-00643-1) |
| **LightGBM** | Buy/Sell/Hold: Sharpe 2.64, +366%, thắng buy-and-hold | ISD 2025 (10.62036/isd.2025.42) |
| **XGBoost** | Trend accuracy 0.930, AUC 0.983; thắng RF/DT | JDSIS 2023 (10.47852/bonviewjdsis3202677) |
| **CatBoost** | F1 cao nhất binary UP/DOWN (60 cấu hình) | ACM 2026 (10.1145/3816713.3818216) |
| **SVM / LogisticReg** | Macro-F1 0.83 trên time-series features, thắng gradient boosting ở một số setup | Bitcasting 2025 (10.21528/cbic2025-1175712) |

→ **Gradient boosting (LightGBM/CatBoost/XGBoost) có bằng chứng mạnh nhất và rẻ nhất
cho đúng bài toán direction classification.** Logistic Regression hiện tại đã là
baseline đúng chuẩn (Bitcasting cho thấy LR/SVM cạnh tranh trên TS features).

### Deep learning cho crypto price/direction

| Mô hình | Kết quả | Nguồn |
|---|---|---|
| **CNN-GRU** | Best accuracy-complexity tradeoff BTC/ETH | ACM DLT 2025 (10.1145/3699966) |
| **ConvLSTM multivariate** | Tốt nhất giữa các DL variants | arXiv 2405.11431 (Chandra group) |
| **Transformer** | MAPE thấp nhất 0.20–0.36% (BTC/ETH/BNB/XRP); best short-term | Sci. J. Informatics 2025; ACM DLT 2025 |
| **Transformer+GRU hybrid** | Thắng BiLSTM/BiGRU có kiểm định Friedman | MDPI Mathematics 2025 (13/9/1484) |
| **TCN-LSTM** | Vượt LSTM/CNN/Transformer/CNN-LSTM | IEEE ICEEIE 2025 |
| **BiLSTM** | Hạng 2 sau Transformer | Sci. J. Informatics 2025 |
| **GRU** | Ổn định, ≈ LSTM với ít tham số hơn | nhiều nguồn |

→ **Hybrid CNN/RNN là xu hướng mạnh**: CNN trích local pattern + RNN nắm phụ thuộc
dài hạn. BiLSTM hiện có đã đúng hướng; Transformer hiện có là lựa chọn tốt.

### Time-series architectures hiện đại (benchmark tổng quát)

| Mô hình | Vị thế | Nguồn |
|---|---|---|
| **PatchTST** | Consistent performer nhất toàn bộ GIFT-Eval; SOTA transformer | arXiv 2211.14730; GIFT-Eval 2410.10393 |
| **iTransformer** | Tốt nhất multivariate (inverted attention) | arXiv 2310.06625 (ICLR 2024) |
| **N-BEATS / N-HiTS** | Best trên 27/41 Monash datasets (univariate) | Monash Benchmarking 04-2025 |
| **TFT** | Thắng N-BEATS: MAPE 6.58% vs 14.18% trên stocks; có feature importance | ICT (arc.cct.ie); arXiv 1912.09363 |
| **DLinear/NLinear** | Baseline linear mạnh, rẻ | arXiv 2205.13504 |

### Foundation models — cảnh báo

- **TimesFM zero-shot trên giá tài chính: THẤT BẠI** — dưới mức random 4/7 horizons
  (Preferred Networks + MIT, arXiv 2412.09880).
- **TimesFM fine-tuned trên 100M financial points: thắng benchmark** (Sharpe 1.68
  trên S&P500) → chỉ có giá trị nếu fine-tune.
- Chronos/Moirai tương tự: mạnh zero-shot chung, nhưng giá crypto là chuỗi nhiễu
  cao, low signal-to-noise → không tin dùng thẳng.

## 2. Khuyến nghị theo tầng (mapping vào registry hiện tại)

### Tier 1 — nên thêm ngay (bằng chứng trực tiếp, rẻ)

- **LightGBM** — đã có trong `backend/requirements.txt`; class mới ~60 dòng,
  `@register_forecaster("LightGBM")`. Bằng chứng mạnh nhất cho Buy/Sell/Hold.
- **XGBoost / CatBoost** — cùng family, swap một dòng config.
- **CNN-LSTM / CNN-GRU** — thêm vào `keras_forecaster.py` (Conv1D+LSTM stack);
  bằng chứng cao nhất trong DL cho direction (82.44%).
- **GRU** — thêm 1 dòng vào `SUPPORTED` của KerasForecaster; nhẹ hơn LSTM.

### Tier 2 — research-worthy (SOTA TS nhưng chưa có bằng chứng crypto-direction)

- **PatchTST** — patching + channel-independent; SOTA benchmark; implement Keras.
- **TFT** — interpretable (variable importance — vừa hay cho trang Feature
  Importance thật thay DEMO!); multi-horizon; support covariates.
- **iTransformer** — multivariate attention; phù hợp khi feature set lớn (set E + macro).

### Tier 3 — để sau / cần hạ tầng

- **N-BEATS/N-HiTS** — chỉ univariate, ít phù hợp multi-feature hiện tại.
- **TimesFM/Chronos fine-tuned** — cần GPU + corpus tài chính; zero-shot không uy tín.
- **TCN(-LSTM)** — thay thế khả dĩ nếu CNN-LSTM hạn chế.

### Không khuyến nghị

- Zero-shot foundation models cho giá (bằng chứng thất bại trên financial data).
- ARIMA/statistical — chỉ nên làm baseline, không phải production model.

## 3. Lưu ý phương pháp (quy trình > mô hình)

Nhiều paper crypto có rò rỉ dữ liệu (leakage) → accuracy ảo. Project đã có:
- `test_2_lookahead_bias`, `test_3_data_leakage` → giữ bắt buộc khi thêm model mới.
- Walk-forward/expanding window thay vì 1 lần train/test split (LightGBM paper
  dùng expanding-window — đây là lý do nó đáng tin).
- Feature selection (Boruta/LightGBM importance) là yếu tố tạo khác biệt lớn
  trong paper 82.44% → feature pipeline quan trọng không kém model.
- Hold/neutral class (±threshold) giảm overtrading — đã có `threshold` trong
  TrendTargetBuilder, có thể mở rộng 3 lớp sau.

## 4. Lộ trình đề xuất

1. `LightGBM` + `GRU` + `CNN-LSTM` (1 buổi, rủi ro thấp nhất).
2. So sánh trên cùng protocol (walk-forward) → có số liệu thật thay giả định.
3. `PatchTST` hoặc `TFT` nếu muốn model "đẹp" cho báo cáo/demo (TFT cho
   interpretability → làm trang Feature Importance thật).
4. Ensemble: voting kết hợp LightGBM + CNN-LSTM (nhiều paper cho thấy ensemble
   hướng đa dạng cải thiện ổn định).
