# Dự báo Hướng Giá Bitcoin bằng Deep Learning

> **Nghiên cứu thực nghiệm**: So sánh Logistic Regression, RNN, LSTM và BiLSTM cho bài toán phân loại xu hướng giá BTC/USDT trên dữ liệu daily, đánh giá bằng Walk-Forward Validation và kiểm định thống kê t-test.

---

## Tóm tắt kết quả (Abstract)

Nghiên cứu so sánh bốn mô hình học máy — Logistic Regression, RNN, LSTM và BiLSTM — trong bài toán dự báo hướng giá Bitcoin (lên/xuống) theo horizon 7 ngày. Dữ liệu gồm 1.000 nến daily BTCUSDT (Binance, Oct 2023 – Jul 2026). Đánh giá sử dụng **Walk-Forward Validation** (5 folds × 10 seeds = 50 mẫu/mô hình) và **Balanced Accuracy** làm chỉ số chính.

**Kết quả chính:**

| Mô hình | Balanced Accuracy | Std | p-value vs 0.5 |
|---|---|---|---|
| **BiLSTM** | **0.5730** | 0.0801 | < 0.001 \*\*\* |
| LSTM | 0.5524 | 0.0556 | < 0.001 \*\*\* |
| RNN | 0.4727 | 0.0941 | 0.048 \* (kém baseline) |
| Logistic Regression | 0.4619 | 0.0854 | 0.003 \*\* (kém baseline) |

BiLSTM vượt baseline ngẫu nhiên có ý nghĩa thống kê (p < 0.001), nhưng chưa vượt LSTM có ý nghĩa (p = 0.087). Macro features (Gold, DXY, Yield) không cải thiện hiệu suất.

---

## Mục lục

- [Bài toán nghiên cứu](#bài-toán-nghiên-cứu)
- [Dữ liệu](#dữ-liệu)
- [Phương pháp](#phương-pháp)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Phân tích thống kê](#phân-tích-thống-kê)
- [Nghiên cứu ablation: Macro features](#nghiên-cứu-ablation-macro-features)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Bài toán nghiên cứu

**Input:** Chuỗi 20 nến OHLCV daily của BTC/USDT + 20 chỉ báo kỹ thuật  
**Output:** Nhãn nhị phân — giá tăng (`UP=1`) hay giảm (`DOWN=0`) sau 7 ngày  
**Baseline ngẫu nhiên:** Balanced Accuracy = 0.500

Bài toán là **classification không cân bằng** trong môi trường phi tham chiếu (non-stationary). Không thể dùng cross-validation thông thường vì sẽ gây data leakage từ tương lai.

---

## Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Symbol | BTCUSDT |
| Nguồn | Binance REST API |
| Khung thời gian | Daily (1d) |
| Số nến | 1.000 |
| Khoảng thời gian | Oct 2023 – Jul 2026 |
| Chiều dài chuỗi (seq_len) | 20 ngày |
| Horizon dự báo | 7 ngày |

### Feature Set B (26 features)

| Nhóm | Features |
|---|---|
| Price | `close`, `ma_7`, `ma_14` |
| Volume | `volume`, `log_volume`, `volume_change` |
| Momentum | `rsi`, `macd`, `roc` |
| Volatility | `atr`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_position`, `bb_width`, `atr_close_ratio` |
| Returns | `log_return_1`, `log_return_7` |
| Ratios | `close_to_ma_7`, `close_to_ma_14` |
| Macro | `gold`, `dxy`, `usd_rate`, `gold_change`, `dxy_change`, `rate_change` |

> **Lưu ý:** Kết quả ablation (xem mục [Macro features](#nghiên-cứu-ablation-macro-features)) cho thấy việc loại bỏ macro features không làm giảm hiệu suất.

---

## Phương pháp

### Walk-Forward Validation

Để tránh data leakage và mô phỏng điều kiện thực tế, nghiên cứu sử dụng **Walk-Forward Validation** theo sơ đồ expanding window:

```
Fold 1: Train[0:200]  → Test[200:250]
Fold 2: Train[0:250]  → Test[250:300]
Fold 3: Train[0:300]  → Test[300:350]
Fold 4: Train[0:350]  → Test[350:400]
Fold 5: Train[0:400]  → Test[400:450]
```

Mỗi mô hình được huấn luyện với **10 seeds khác nhau** → 50 giá trị Balanced Accuracy độc lập/mô hình, đủ mẫu cho paired t-test.

### Xử lý tính xác định (Determinism)

- Neural networks (RNN/LSTM/BiLSTM): stochastic theo seed TensorFlow
- Logistic Regression: sử dụng `SGDClassifier(loss='log_loss')` thay vì `LogisticRegression(solver='lbfgs')` để đảm bảo stochastic qua các seed (tránh n hiệu lực chỉ là 5 thay vì 50)

### Kiến trúc mô hình

| Mô hình | Cấu hình |
|---|---|
| Logistic Regression | `SGDClassifier(loss='log_loss')`, flatten (seq_len × n_features) |
| RNN | `SimpleRNN(units=32)`, dropout=0.1 |
| LSTM | `LSTM(units=32)`, dropout=0.1 |
| BiLSTM | `Bidirectional(LSTM(units=32))`, dropout=0.1 |

**Hyperparameters chung (KerasForecaster):** epochs=40, batch_size=16, lr=0.001, optimizer=Adam, loss=binary_crossentropy

### Tiền xử lý

- **Scaling:** `RobustScaler` — fit trên tập train của mỗi fold, transform test (không data leakage)
- **Windowing:** `WindowDataset(seq_len=20)` tạo tensor 3D `(N, 20, n_features)`
- **Target:** `TrendTargetBuilder` — nhãn 1 nếu `close[t+7] > close[t]`, ngược lại 0

---

## Kết quả thực nghiệm

### Balanced Accuracy — Walk-Forward Validation (n=50 mỗi mô hình)

| Mô hình | Mean | Std | Min | Max |
|---|---|---|---|---|
| **BiLSTM** | **0.5730** | 0.0801 | 0.3815 | 0.7597 |
| LSTM | 0.5524 | 0.0556 | 0.3815 | 0.7241 |
| RNN | 0.4727 | 0.0941 | 0.2791 | 0.6715 |
| Logistic Regression | 0.4619 | 0.0854 | 0.3036 | 0.6051 |
| *Baseline ngẫu nhiên* | *0.5000* | — | — | — |

### Diễn giải

- **BiLSTM** và **LSTM** vượt baseline ngẫu nhiên một cách có ý nghĩa thống kê
- **RNN** và **Logistic Regression** thực tế kém hơn baseline (do học lệch về một lớp)
- Độ biến động (std) lớn cho thấy thị trường crypto có giai đoạn dự báo được và giai đoạn gần ngẫu nhiên

---

## Phân tích thống kê

### One-sample t-test: mô hình vs baseline 0.5

| Mô hình | t-statistic | p-value | Kết luận |
|---|---|---|---|
| BiLSTM | +6.384 | < 0.001 | Vượt baseline \*\*\* |
| LSTM | +6.596 | < 0.001 | Vượt baseline \*\*\* |
| RNN | −2.030 | 0.048 | Kém baseline \* |
| Logistic Regression | −3.123 | 0.003 | Kém baseline \*\* |

### Paired t-test: so sánh cặp (n=50)

| So sánh | Diff | t | p-value | Kết luận |
|---|---|---|---|---|
| BiLSTM vs LSTM | +0.0207 | 1.746 | 0.087 | Không significant |
| BiLSTM vs RNN | +0.1003 | 5.019 | < 0.001 | BiLSTM tốt hơn \*\*\* |
| BiLSTM vs Logistic | +0.1112 | 7.506 | < 0.001 | BiLSTM tốt hơn \*\*\* |
| LSTM vs RNN | — | — | < 0.001 | LSTM tốt hơn \*\*\* |
| LSTM vs Logistic | — | — | < 0.001 | LSTM tốt hơn \*\*\* |
| RNN vs Logistic | — | — | 0.622 | Không khác biệt |

> *Mức ý nghĩa: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

**Kết luận chính:** BiLSTM là mô hình tốt nhất nhưng không thể khẳng định vượt LSTM (p=0.087 > 0.05). LSTM và BiLSTM là nhóm hiệu quả; RNN và Logistic là nhóm không hiệu quả trên bộ dữ liệu này.

---

## Nghiên cứu Ablation: Macro Features

Nghiên cứu kiểm tra liệu Gold price, US Dollar Index (DXY) và US 10Y Treasury Yield có cải thiện mô hình không, qua 3 thử nghiệm độc lập:

### Thử nghiệm 1 — Correlation analysis (n=992 ngày)

| Feature | Pearson r | p-value |
|---|---|---|
| Gold level | −0.1245 | < 0.001 \*\*\* |
| US Yield return lag-1 | −0.0483 | 0.127 |
| DXY level | +0.0438 | 0.167 |
| Gold/DXY/Yield returns | < 0.04 | > 0.2 |

Gold level có tương quan có ý nghĩa thống kê (p<0.001) nhưng |r|=0.12 quá nhỏ để có giá trị thực tiễn. Rolling 60-day correlation dao động mạnh (std=0.13) và không nhất quán về chiều.

### Thử nghiệm 2 — Walk-forward Logistic (n=25)

| Variant | Mean BalAcc | Std |
|---|---|---|
| Feature Set B (có macro) | 0.4577 | 0.0864 |
| Feature Set B (không macro) | 0.5250 | 0.0806 |

Paired t-test: diff=−0.0673, **p=0.012** — macro làm **giảm** accuracy đáng kể với Logistic.

### Thử nghiệm 3 — Walk-forward BiLSTM (n=25)

| Variant | Mean BalAcc | Std |
|---|---|---|
| B không macro | 0.5706 | 0.0807 |
| B + macro returns | 0.5703 | 0.0851 |

Paired t-test: diff=−0.0003, **p=0.989** — không có tác động nào.

**Kết luận ablation:** Macro features (Gold/DXY/Yield) **không cải thiện** và có thể làm giảm hiệu suất. Nguyên nhân: (1) tương quan tuyến tính gần 0 với target, (2) macro data là weekend-fill nên chứa nhiều giá trị lặp, (3) mối quan hệ BTC–macro phi tuyến và thay đổi theo regime thị trường.

---

## Cài đặt và chạy

### Yêu cầu

- Python 3.10+
- pip

### Cài đặt

```bash
git clone https://github.com/lekimphu0209/crypto-price-prediction.git
cd crypto-price-prediction
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### Chạy thực nghiệm

```bash
# Walk-forward BiLSTM (10 seeds × 5 folds)
python scripts/train_btc_walkforward_bilstm.py

# Walk-forward RNN + LSTM
python scripts/train_btc_walkforward.py

# Walk-forward Logistic Regression
python scripts/train_btc_walkforward_logistic.py

# So sánh tất cả mô hình bằng t-test
python scripts/compare_all_models_ttest.py

# EDA macro features
python exploration/macro_correlation_analysis.py

# Ablation test: BiLSTM + macro
python exploration/bilstm_macro_test.py
```

### Chạy API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8502
# Endpoint: GET /api/predict/?symbol=BTCUSDT
# Endpoint: GET /api/models/
```

### Chạy Dashboard

```bash
streamlit run dashboard/streamlit_app.py
# Mở: http://localhost:8501
```

---

## Cấu trúc dự án

```
crypto-price-prediction/
│
├── src/
│   ├── domain/
│   │   ├── interfaces/          # IFeatureExtractor, IForecaster, ...
│   │   ├── entities/            # OHLCV
│   │   └── value_objects/       # Symbol, Interval
│   │
│   ├── features/
│   │   ├── indicators.py        # MA, RSI, MACD, BB, ATR, ...
│   │   └── technical_extractor.py  # TechnicalExtractor (feature sets A/B/C/E)
│   │
│   ├── training/
│   │   ├── training_pipeline.py    # Orchestrator
│   │   ├── window_dataset.py       # Sliding window (seq_len=20)
│   │   ├── return_target_builder.py  # TrendTargetBuilder (UP/DOWN)
│   │   ├── market_data_loader.py
│   │   └── composition.py          # Composition root / wiring
│   │
│   ├── infrastructure/
│   │   ├── forecasters/
│   │   │   ├── keras_forecaster.py    # RNN / LSTM / BiLSTM / Transformer
│   │   │   ├── linear_forecaster.py   # SGDClassifier (log_loss)
│   │   │   └── factory.py
│   │   ├── data_providers/
│   │   │   ├── binance_provider.py
│   │   │   └── yfinance_provider.py
│   │   └── persistence/
│   │       └── file_registry.py       # Lưu model.keras / scaler.pkl / meta.json
│   │
│   ├── api/
│   │   ├── main.py                    # FastAPI app
│   │   └── routes/
│   │       ├── predict.py
│   │       └── models.py
│   │
│   └── utils/
│       └── determinism.py             # set_full_determinism(seed)
│
├── scripts/
│   ├── train_btc_walkforward.py          # RNN + LSTM
│   ├── train_btc_walkforward_bilstm.py   # BiLSTM
│   ├── train_btc_walkforward_logistic.py # Logistic
│   └── compare_all_models_ttest.py       # T-test comparison
│
├── exploration/                          # EDA / ablation (độc lập với pipeline)
│   ├── macro_correlation_analysis.py     # Correlation: macro vs BTC direction
│   ├── bilstm_macro_test.py              # Ablation: BiLSTM + macro features
│   └── charts/                           # Hình ảnh EDA output
│
├── tests/
│   ├── test_1_window_dataset.py
│   ├── test_2_lookahead_bias.py
│   ├── test_3_data_leakage.py
│   ├── test_4_target_builder.py
│   └── test_5_majority_baseline.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── docs/
│   └── bao_cao.md                        # Nội dung báo cáo chi tiết
│
├── models/                               # Kết quả thực nghiệm (gitignored)
│   ├── btc_walkforward_bilstm_results.json
│   ├── btc_walkforward_results_detailed.json
│   ├── btc_walkforward_logistic_results.json
│   └── btc_bilstm_macro_test.json
│
├── requirements.txt
└── README.md
```

---

## Chạy kiểm thử

```bash
python -m pytest tests/ -v
```

| Test | Mục đích |
|---|---|
| `test_1_window_dataset.py` | WindowDataset không tạo overlap hay gap |
| `test_2_lookahead_bias.py` | Scaler fit chỉ trên train, không dùng future data |
| `test_3_data_leakage.py` | Target không bị leak vào features |
| `test_4_target_builder.py` | TrendTargetBuilder tính đúng nhãn UP/DOWN |
| `test_5_majority_baseline.py` | Baseline majority classifier = 0.5 balanced accuracy |

---

## Tài liệu tham khảo

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
4. Prado, M. L. de (2018). *Advances in Financial Machine Learning*. Wiley.
5. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
6. Scikit-learn: Pedregosa et al. (2011). *JMLR*, 12, 2825–2830. https://scikit-learn.org/
7. TensorFlow: Abadi et al. (2016). https://www.tensorflow.org/

---

## License

Dự án này được xây dựng cho mục đích học thuật và nghiên cứu.

## Tác giả

- GitHub: [@lekimphu0209](https://github.com/lekimphu0209)
- Repository: https://github.com/lekimphu0209/crypto-price-prediction
