# Backend — Engine (training / API / models / data)

Phần chức năng của hệ thống: pipeline huấn luyện, FastAPI phục vụ dự đoán,
model registry plugin, và artifact model đã train. Giao diện Streamlit nằm
tách riêng trong `gui/`.

## Cấu trúc

```
backend/
├── src/
│   ├── api/            # FastAPI: /api/predict, /api/models
│   ├── domain/         # interfaces + entities (OHLCV, Symbol, Interval)
│   ├── features/       # TechnicalExtractor, market-regime extractors
│   ├── models/         # forecaster registry + implementations
│   │   ├── keras_forecaster.py       # RNN / LSTM / BiLSTM / Transformer
│   │   ├── linear_forecaster.py      # Logistic Regression
│   │   ├── lightgbm_forecaster.py    # LightGBM (mới, 10/2026)
│   │   └── registry.py               # plugin registry + auto yaml hyperparams
│   ├── training/       # pipelines, datasets, targets, composition root
│   ├── evaluation/     # metrics (classification + regression)
│   └── infrastructure/ # binance/yfinance providers, FileModelRegistry
├── configs/            # default.yaml + models/<model>.yaml (hyperparams)
├── scripts/            # train_models.py (all), train_one.py (single)
├── tests/              # pytest suite (lookahead/leakage/reproducibility)
├── models/             # artifact: {SYMBOL}/{model}/model + scaler + meta.json
└── data/               # raw OHLCV + features + results CSV
```

Chạy API (từ repo root):
```bash
.venv\Scripts\uvicorn src.api.main:app --app-dir backend --host 0.0.0.0 --port 8502
```

---

# Huấn luyện 1 mô hình: LightGBM — quá trình thực hiện

## 1. Tại sao chọn LightGBM

Dựa trên khảo sát tài liệu (`docs/model_research.md`), cho đúng bài toán
**phân loại hướng xu hướng** của project (UP/DOWN 7 ngày, ~1000 ngày dữ liệu):

- **Bằng chứng trực tiếp mạnh nhất**: LightGBM đạt Sharpe 2.64, +366% trên
  BTC Buy/Sell/Hold và *thắng buy-and-hold* (ISD 2025); XGBoost/CatBoost
  cùng family đạt accuracy 0.93 / F1 cao nhất trong các study tương đương.
- **Rẻ và nhanh**: fit < 1 giây trên CPU — phù hợp vòng lặp thử nghiệm nhanh,
  không cần GPU như các mô hình Keras.
- **Diễn giải được**: `feature_importances_` cho phép làm trang Feature
  Importance *thật* thay dữ liệu DEMO hiện tại trên dashboard.
- Baseline hiện có (`Logistic Regression`) là linear; cần một model phi tuyến
  để so sánh trên cùng pipeline.

## 2. Các file đã thêm/sửa

| File | Thay đổi |
|---|---|
| `backend/src/models/lightgbm_forecaster.py` | Class mới implement `IForecaster` — flatten window `(seq_len, n_features)` rồi fit `LGBMClassifier`/`LGBMRegressor`; save/load bằng `joblib`; expose `feature_importances()` |
| `backend/src/models/registry.py` | Đăng ký `"LightGBM"`; `create_forecaster` giờ tự merge `configs/models/<slug>.yaml` vào hyperparams (explicit kwargs override; key lạ bị drop theo signature `__init__`) |
| `backend/configs/models/lightgbm.yaml` | Hyperparams mặc định (n_estimators=300, lr=0.05, num_leaves=31, …) |
| `backend/scripts/train_one.py` | Script mới: train đúng 1 cặp `{symbol, model}` qua cùng pipeline (thay vì loop toàn bộ như `train_models.py`) |
| `backend/requirements.txt` | `lightgbm` đã khai báo sẵn; cài `pip install lightgbm` → 4.7.0 |

## 3. Các bước thực hiện (đã chạy)

```bash
# 1. Cài dependency
.venv\Scripts\python.exe -m pip install lightgbm        # -> 4.7.0

# 2. Kiểm tra registry nhận model mới + merge yaml
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'backend'); \
  from src.models.registry import create_forecaster, registered_models; \
  print(registered_models()); m=create_forecaster('LightGBM'); print(m._params)"
# -> ['BiLSTM','LSTM','LightGBM','Logistic Regression','RNN','Transformer']
# -> params đúng như lightgbm.yaml

# 3. Train trên dữ liệu Binance thật — TOÀN BỘ lịch sử BTCUSDT
#    (train_days=10000 trong default.yaml → fetch cap ở max khả dụng)
.venv\Scripts\python.exe backend/scripts/train_one.py --symbol BTCUSDT --model LightGBM
```

**Lưu ý về dữ liệu**: `train_days: 10000` yêu cầu lấy tới 10.000 nến ngày từ
hôm nay lùi về — nhưng BTCUSDT trên Binance chỉ tồn tại từ **2017-08-17**, nên
kết quả thực tế là **3,357 nến** (2017-08-17 → 2026-10-25, toàn bộ lịch sử
available). Để vượt cap ~1000 nến/request của Binance API, `fetch_ohlcv` đã
được nâng cấp thêm:
- **Pagination** lùi theo thời gian (loop `since`/`endTime`), đủ cho limit bất kỳ.
- **Raw `klines` endpoint** (`publicGetKlines`) thay `fetch_ohlcv` chuẩn — bỏ
  qua `load_markets()`/`exchangeInfo` vốn intermittent timeout trên mạng chập chờn.
- **`enableRateLimit` + retry/backoff** cho NetworkError.

Pipeline (`build_training_pipeline` trong `src/training/composition.py`) thực hiện:
`Binance OHLCV → TechnicalExtractor (feature_set A: MA, RSI, MACD, BB, ATR, ROC)
→ WindowDataset seq_len=20 → TrendTarget horizon=7 → StandardScaler (fit TRAIN
chỉ — không leakage) → LGBMClassifier → tối ưu threshold → FileModelRegistry
.save()`.

Artifact tạo ra:
- `backend/models/BTCUSDT/lightgbm/` → `model.pkl` + `scaler.pkl` + `meta.json`
  (meta chứa `mode=classification`, `horizon=7`, `threshold`, `extractor_config`
  → bundle tự khôi phục đúng extractor lúc predict).
- `backend/data/BTCUSDT_raw_*.csv`, `*_features_*.csv`, `*_results_*.csv`.

## 4. Kết quả thật

```
LightGBM on BTCUSDT (train 80% / test 20%, chronological, 3,357 nến ngày):
  accuracy           0.5249
  balanced_accuracy  0.5241     <- threshold 0.45 tối ưu cho balanced acc
  precision          0.5260
  recall             0.6030
  f1                 0.5619
  threshold          0.4500

So với lần train trước trên 1000 ngày:  bal_acc 0.5094 -> 0.5241, f1 0.6179 -> 0.5619
(recall giảm 0.80 -> 0.60 nhưng precision + balanced acc tăng -> cân bằng hơn)
```

**Diễn giải trung thực**: thêm ~2.4x dữ liệu cải thiện nhẹ balanced accuracy
(+0.015) nhưng vẫn chỉ nhỉnh baseline ngẫu nhiên (0.50) — nhất quán với
literature: trên feature set hiện tại crypto rất khó dự đoán, và gradient
boosting chỉ thực sự mạnh khi đi kèm **feature selection** (Boruta/LightGBM
importance) và **expanding-window retraining** (thay vì 1 lần split). Đây là
bước cải tiến tiếp theo — hiện tại model hoạt động đúng pipeline và tích hợp
được API/dashboard, nhưng chưa là cải thiện accuracy rõ rệt so với BiLSTM/LSTM.

Verify bundle tự khôi phục đúng:
```python
FileModelRegistry('backend/models').load('BTCUSDT', 'LightGBM')
# -> forecaster=LightGBM, mode=classification, horizon=7,
#    threshold=0.45, extractor_config={'name':'technical', params:{feature_set:'A',...}}
```

## 5. Đánh giá đúng chuẩn: walk-forward (26/10/2026)

Single split 80/20 chỉ cho **1 con số lạc quan** trên 1 đoạn test. Đúng chuẩn
cho time series là **expanding-window** (repeated hold-out — xem
`docs/slide_mapping.md`):

```bash
.venv\Scripts\python.exe backend/scripts/walk_forward_eval.py \
    --model LightGBM --feature-set A --folds 5
```

`walk_forward_eval.py` mỗi fold: refit scaler chỉ trên train rows → tune
threshold trên validation slice → forecaster **mới** → đánh giá fold tiếp theo
chưa thấy. Kết quả thật (BTCUSDT, 3,357 nến, 5 folds):

| Config | bal_acc | acc | f1 |
|---|---|---|---|
| feature_set E (42 feat) | 0.5054±0.018 | 0.4812 | 0.397 |
| E pruned -7 feat (LGBM importance) | 0.5215±0.022 | 0.5055 | 0.435 |
| **feature_set A (18 feat, production)** | **0.5296±0.028** | 0.5202 | 0.543 |

→ **Feature selection qua `feature_importances_`** (loại 7 feature đuôi:
dxy_change, bb_middle, gold_change, dxy, ma_14, log_volume, ma_7) cải thiện
E +0.016 — nhưng bộ A chọn sẵn vẫn tối ưu → giữ config production. Con số
công bằng của LightGBM là **0.5296±0.028**, không phải 0.5241 của single split.
Kết quả JSON: `backend/data/*_walkforward_*.json`.

## 6. Cách mở rộng thêm model mới (plugin pattern)

1. Tạo class implement `IForecaster` (`fit`/`predict`/`save`/`load`/`name`) trong
   `src/models/` — xem `linear_forecaster.py` hoặc `lightgbm_forecaster.py`.
2. `register_forecaster("Tên Model", MyForecaster)` trong `registry.py`.
3. (Tùy chọn) `configs/models/<slug>.yaml` cho hyperparams — tự động merge.
4. Train: `python backend/scripts/train_one.py -s BTCUSDT -m "Tên Model"`.
   Không cần sửa pipeline/API/dashboard — registry lo hết.

## 6. Model đã train có sẵn

```
models/BTCUSDT/  logistic_regression/  rnn/  lstm/  bilstm/  transformer/  lightgbm/
models/ETHUSDT/  (cùng cấu trúc)
models/_summary.json   # metrics tổng hợp của lần train_models.py gần nhất
```
