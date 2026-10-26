# Tóm tắt slide IT3190 (Nhập môn Học máy & Khai phá dữ liệu)
# và ứng dụng vào quy trình xử lý + huấn luyện dữ liệu của project

Nguồn: `D:\2026.1\NMHM\Slide` (14 file PDF, khóa IT3190).
Mục đích: ánh xạ khái niệm lý thuyết trong slide lên từng bước cụ thể trong
pipeline `backend/` của project dự đoán xu hướng crypto.

## 1. Tóm tắt theo nhóm slide

### L0 + L1 + L1.2 — Nhập môn & quy trình KDD

- Phân biệt **dữ liệu → thông tin → tri thức** (dữ liệu thô = tín hiệu; tri thức =
  thông tin đã tích hợp có định hướng quyết định).
- **Quy trình phát hiện tri thức 5 bước**: (1) thu thập/lựa chọn dữ liệu →
  (2) tiền xử lý → (3) chuyển đổi → (4) khai phá dữ liệu (học máy) →
  (5) giải thích/đánh giá.
- Phân biệt **predictive task** (phân loại, hồi quy) vs **descriptive task**
  (phân cụm, luật kết hợp) — project thuộc predictive/classification.

### L2 — Tiền xử lý dữ liệu

- Vì sao phải tiền xử lý: dữ liệu **incomplete** (thiếu), **noisy** (lỗi),
  **inconsistent** (không nhất quán).
- **Làm sạch** 3 bước: điền giá trị thiếu (bỏ/mean/mode/hồi quy/Bayes) →
  chuốt nhiễu (hồi quy, phân cụm) → sửa không nhất quán (unique/consecutive/null rules).
- **Tích hợp/chuyển đổi**: tập hợp, tổng quát hóa, **chuẩn hóa** (đưa về phạm vi
  chuẩn), **xây dựng thuộc tính** (attribute construction — tạo feature mới).
- **Giảm chiều**: chọn tập con thuộc tính, mã hóa giảm chiều, rời rạc hóa.

### L3 — Hồi quy tuyến tính

Baseline đơn giản — trong project là `Logistic Regression` (linear classifier).

### L4-5, L7, L9, L9.1, L11, L12 — Mô hình phụ trợ

Clustering, KNN, Naive Bayes/EM, SVM, association rules — descriptive/predictive
không trực tiếp dùng trong pipeline hiện tại, nhưng là lý thuyết nền
(khoảng cách, xác suất có điều kiện, margin).

### L8 — Cây quyết định & Rừng ngẫu nhiên

- Decision tree: duyệt gốc→lá, disjunction của điều kiện; ID3/C4.5 (entropy, gini).
- **Random Forest**: ensemble qua **bootstrap sampling** (lấy mẫu lặp lại),
  mỗi cây học trên mẫu bootstrap + **tập con ngẫu nhiên các feature**,
  quyết định bằng **đa số phiếu** → giảm variance, chống overfit.
- **Feature importance** từ ensemble cây — ý tưởng LightGBM dùng.

### L6 — Đánh giá hiệu năng (slide quan trọng nhất cho training)

- **Các kỹ thuật chia dữ liệu**: hold-out; **stratified sampling** (giữ phân bố
  lớp — chỉ áp dụng cho phân loại); repeated hold-out; **k-fold CV** (k=10/5);
  LOOCV; **bootstrap** (train ≈63.2%, test ≈36.8% không trùng).
- **Model selection**: tách **validation set** khỏi train; quy trình 3 tập
  `Dtrain → Tvalidation (chọn λ*) → retrain trên train+val → đánh Ttest`.
- **Tiêu chí đánh giá hệ thống**: accuracy, efficiency, robustness, scalability,
  interpretability, complexity.
- **Metrics phân loại**: confusion matrix (TP/FP/TN/FN), precision, recall,
  micro/macro averaging, **F1 = trung bình điều hòa** (thiên về giá trị nhỏ hơn
  → cả 2 phải cao thì F1 mới cao).

### L10 — Mạng nơron

- Perceptron → MLP nhiều tầng ẩn; activation (ReLU); feedforward vs feedback.
- Training = tối ưu loss bằng các phương pháp gradient (backprop, AdaGrad...);
  phân biệt parameter learning vs structure learning.
- Nền cho RNN/LSTM/BiLSTM/Transformer trong project (feedback/recurrent nets).

---

## 2. Ánh xạ slide → pipeline thực tế của project

| Bước KDD (slide) | Code trong `backend/` | Ghi chú |
|---|---|---|
| **1. Thu thập dữ liệu** | `BinanceProvider.fetch_ohlcv` → `MarketDataLoader` | OHLCV ngày, pagination >1000 nến, retry/backoff |
| **2. Làm sạch** | loader `dropna`/lọc ngày | Crypto gần như không missing; inconsistency thấp |
| **3. Chuyển đổi** | `TechnicalExtractor` (**attribute construction**: MA/RSI/MACD/BB/ATR/ROC) → `WindowDataset` (seq_len=20) | Xây dựng thuộc tính + tạo cấu trúc sequence |
| **Chuẩn hóa** | `StandardScaler` **fit chỉ trên train** | Đúng nguyên tắc tránh leakage |
| **4. Khai phá** | forecaster: LogReg/LightGBM/RNN/LSTM/BiLSTM/Transformer | Classification (predictive task) |
| **5. Đánh giá** | `train_pipeline` tính acc/bal_acc/P/R/F1/threshold; `test_2`,`test_3` | Confusion-matrix-based metrics |

### Chi tiết từng khái niệm

- **Attribute construction (L2)** = `TechnicalExtractor.feature_set A` — tạo
  MA7/MA14/RSI/MACD/BB/ATR/ROC từ OHLCV thô. Đây chính là "xây dựng thuộc tính
  giúp khai phá tốt hơn" theo đúng nghĩa slide.
- **Normalization (L2)** = `StandardScaler` trong `train_pipeline` — fit TRAIN
  only, transform cả val/test (tránh lookahead — gắn với `test_2_lookahead_bias`).
- **Giảm chiều/chọn feature (L2)** = `feature_set A|B|C|E` trong config extractor
  + `exclude_features`; hướng tới Boruta/LightGBM importance selection.
- **Hold-out split (L6)** = `train_split=0.8` chronological trong
  `TrainingPipeline` — hold-out theo **thời gian** (không shuffle, do time series).
- **3-way train/val/test (L6)** = pipeline dùng train/val cho early stopping &
  threshold optimization; test giữ nguyên → đúng model selection protocol.
- **Bootstrap sampling (L6)** = nền của `n=50 (seed × fold)` trong
  `test_4/5/6` — LinearForecaster dùng SGDClassifier theo seed để có biến động
  thực; LightGBM cũng hưởng bootstrap ngầm qua `subsample=0.8` (giống bagging RF).
- **Stratified sampling (L6)** — chưa áp dụng (split chronological giữ phân bố
  lớp thời gian); `class_weight="balanced"` trong LogReg thay thế vai trò xử lý
  lệch lớp.
- **Confusion matrix + P/R/F1 (L6)** = đúng metrics pipeline in ra
  `[TN FP FN TP]` + `precision`/`recall`/`f1` + `balanced_accuracy` —
  `balanced_accuracy` = trung bình macro của recall theo lớp (chuẩn micro/macro
  trong slide).
- **Threshold optimization** = `threshold_metric="balanced_accuracy"` quét
  threshold → chính là **model selection trên validation** ở cấp ngưỡng.
- **Ensemble (L8)** = Random Forest ↔ LightGBM: cùng family tree ensemble;
  RF bagging (bootstrap+vote) còn LightGBM **boosting** (cây sau sửa lỗi cây
  trước) — `n_estimators=300`, `subsample=0.8` gần khái niệm bagging.
- **Gradient + backprop (L10)** = optimizer Adam/early stopping trong
  `KerasForecaster`; `num_leaves`/`learning_rate` của LGBM là hyperparams λ
  theo nghĩa "lựa chọn tham số" của slide 6.
- **Tiêu chí đánh giá ngoài accuracy (L6)**: efficiency (LGBM <1s vs LSTM ~30s),
  interpretability (`feature_importances_` của LGBM/RF) — đã ghi trong
  `docs/model_research.md` là lý do ưu tiên LGBM.

## 3. Điểm pipeline chưa khớp slide → hướng cải thiện

Trạng thái sau đợt cải tiến (26/10/2026):

- **[ĐÃ LÀM] Repeated/expanding hold-out**: `scripts/walk_forward_eval.py`
  — expanding-window 5 folds, scaler refit mỗi fold chỉ trên train rows,
  threshold tune trên validation slice, forecaster mới mỗi fold. Đúng chuẩn
  "repeated hold-out" + model selection qua validation của slide 6.
- **[ĐÃ LÀM] Feature selection chủ động** (L2 giảm chiều): dùng LightGBM
  `feature_importances_` (tổng hợp qua folds, reshape về per-feature) để prune
  — loại 7 feature đuôi (dxy_change, bb_middle, gold_change, dxy, ma_14,
  log_volume, ma_7) khỏi feature_set E → bal_acc +0.016.
- **k-fold CV**: cố tình không dùng (đúng cho time series — thay bằng
  expanding-window; stratified CV chỉ nên áp dụng nếu bỏ qua thứ tự thời gian).
- **Rời rạc hóa (L2)**: chưa dùng — có thể thử binning price/return thành
  regime (đã có `market_regime` extractor — đúng hướng "trừu tượng khái niệm").

### Kết quả walk-forward thật (LightGBM, BTCUSDT, 3,357 nến, 5 folds)

| Config | bal_acc | acc | f1 | Nhận xét |
|---|---|---|---|---|
| feature_set E (42 feat) | 0.5054±0.018 | 0.4812 | 0.397 | Nhiều feature macro/level nhiễu |
| **E pruned (-7)** | **0.5215±0.022** | 0.5055 | 0.435 | Feature selection +0.016 |
| **feature_set A (18 feat)** | **0.5296±0.028** | 0.5202 | 0.543 | **Config tốt nhất — giữ nguyên** |

→ Feature selection chứng minh giá trị trên E (đúng tinh thần paper Boruta +
  slide L2 "lựa chọn tập con thuộc tính"), nhưng bộ A được chọn sẵn vẫn tối ưu
  → giữ config production. Single-split 0.5241 trước đây hơi lạc quan; con số
  công bằng là 0.5296±0.028 trên 5 folds (vẫn > baseline ngẫu nhiên 0.50,
  với fold 4 đạt 0.5766 — đúng nhận xét README: crypto có giai đoạn dự báo được).
