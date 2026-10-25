# Báo cáo Nghiên cứu: Dự báo Hướng Giá Bitcoin bằng Deep Learning

**Tác giả:** Le Kim Phu  
**Repository:** https://github.com/lekimphu0209/crypto-price-prediction  
**Ngày hoàn thành:** Tháng 7, 2026

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Thu thập và khảo sát dữ liệu](#2-thu-thập-và-khảo-sát-dữ-liệu)
3. [Tiền xử lý và trích xuất đặc trưng](#3-tiền-xử-lý-và-trích-xuất-đặc-trưng)
4. [Thiết kế thực nghiệm](#4-thiết-kế-thực-nghiệm)
5. [Kiến trúc mô hình](#5-kiến-trúc-mô-hình)
6. [Kết quả thực nghiệm — Mô hình gốc](#6-kết-quả-thực-nghiệm--mô-hình-gốc)
7. [Kết quả thực nghiệm — Kỹ thuật Ensemble](#7-kết-quả-thực-nghiệm--kỹ-thuật-ensemble)
8. [Nghiên cứu Ablation — Macro Features](#8-nghiên-cứu-ablation--macro-features)
9. [Phân tích và Thảo luận](#9-phân-tích-và-thảo-luận)
10. [Kết luận](#10-kết-luận)
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1. Bối cảnh và Động lực

Thị trường tiền điện tử (cryptocurrency) là một trong những thị trường tài chính có tính biến động cao nhất, với Bitcoin (BTC) là tài sản có vốn hoá lớn nhất và thanh khoản tốt nhất. Đặc điểm phi tham chiếu (*non-stationary*), hoạt động 24/7 và chịu ảnh hưởng đồng thời từ yếu tố kỹ thuật, vĩ mô và tâm lý thị trường khiến việc dự báo giá crypto trở thành bài toán có độ khó cao trong lĩnh vực học máy tài chính.

Nghiên cứu này tập trung vào bài toán **phân loại hướng giá** (*directional classification*): dự báo Bitcoin sẽ **tăng hay giảm** sau 7 ngày, dựa trên dữ liệu lịch sử OHLCV và các chỉ báo kỹ thuật.

### 1.2. Câu hỏi nghiên cứu

1. Trong số các mô hình Logistic Regression, RNN, LSTM và BiLSTM, mô hình nào cho Balanced Accuracy cao nhất và có ý nghĩa thống kê vượt baseline ngẫu nhiên?
2. Các kỹ thuật ensemble (seed ensemble, bagging) có cải thiện hiệu suất đáng kể không?
3. Dữ liệu vĩ mô (Gold, DXY, US 10Y Yield) có đóng góp tín hiệu hữu ích cho mô hình không?

### 1.3. Đóng góp chính

- Đánh giá nghiêm ngặt 4 mô hình bằng Walk-Forward Validation với kiểm định thống kê t-test (n=50 mẫu/mô hình).
- Xây dựng pipeline hoàn chỉnh theo Clean Architecture, đảm bảo không có data leakage.
- Ablation study có hệ thống về macro features qua 3 thử nghiệm độc lập (correlation, Logistic walk-forward, BiLSTM walk-forward).

---

## 2. Thu thập và Khảo sát Dữ liệu

### 2.1. Nguồn dữ liệu

| Loại dữ liệu | Nguồn | Ghi chú |
|---|---|---|
| BTC/USDT OHLCV | Binance REST API (`/api/v3/klines`) | 1.000 nến daily |
| Gold price | Yahoo Finance (`GC=F`, fallback `GLD`) | Dùng cho ablation study |
| US Dollar Index | Yahoo Finance (`DX-Y.NYB`, fallback `DX=F`) | Dùng cho ablation study |
| US 10Y Treasury Yield | Yahoo Finance (`^TNX`) | Dùng cho ablation study |

### 2.2. Đặc điểm bộ dữ liệu chính

| Thuộc tính | Giá trị |
|---|---|
| Symbol | BTCUSDT |
| Khung thời gian | Daily (1d) |
| Số nến | 1.000 |
| Khoảng thời gian | 17/10/2023 – 12/07/2026 |
| Các trường | Open, High, Low, Close, Volume |

### 2.3. Phân tích thống kê mô tả dữ liệu thô

Giá đóng cửa BTC trong giai đoạn nghiên cứu dao động từ khoảng $25.000 đến $110.000, thể hiện tính phi tham chiếu rõ ràng — điều này là lý do chính để sử dụng Walk-Forward Validation thay vì cross-validation ngẫu nhiên.

Phân phối nhãn mục tiêu (horizon=7 ngày):
- UP (giá tăng sau 7 ngày): ~55%
- DOWN (giá giảm sau 7 ngày): ~45%

Mất cân bằng nhẹ → sử dụng **Balanced Accuracy** thay vì Accuracy thông thường.

---

## 3. Tiền xử lý và Trích xuất Đặc trưng

### 3.1. Pipeline trích xuất đặc trưng

`TechnicalExtractor` (lớp `src/features/technical_extractor.py`) tính toán các chỉ báo kỹ thuật từ dữ liệu OHLCV thô. Nghiên cứu sử dụng **Feature Set B** gồm 26 đặc trưng:

| Nhóm | Tên đặc trưng | Mô tả |
|---|---|---|
| Price | `close`, `ma_7`, `ma_14` | Giá đóng cửa, trung bình động 7 và 14 ngày |
| Volume | `volume`, `log_volume`, `volume_change` | Khối lượng giao dịch và biến động khối lượng |
| Momentum | `rsi`, `macd`, `roc` | RSI(14), MACD, Rate of Change(12) |
| Volatility | `atr`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_position`, `bb_width`, `atr_close_ratio` | ATR(14), Bollinger Bands(20), vị trí và độ rộng BB |
| Returns | `log_return_1`, `log_return_7` | Log return 1 ngày và 7 ngày |
| Ratios | `close_to_ma_7`, `close_to_ma_14` | Độ lệch giá so với trung bình động |
| Macro | `gold`, `dxy`, `usd_rate`, `gold_change`, `dxy_change`, `rate_change` | Dữ liệu vĩ mô và biến động (xem Mục 8) |

### 3.2. Xây dựng biến mục tiêu

`TrendTargetBuilder` xây dựng nhãn nhị phân:

```
y[t] = 1  nếu close[t + 7] > close[t]   (UP)
y[t] = 0  nếu close[t + 7] ≤ close[t]   (DOWN)
```

Horizon = **7 ngày** được chọn để tránh nhiễu quá mức của horizon 1 ngày và đủ ngắn để tín hiệu kỹ thuật còn có ý nghĩa.

### 3.3. Chuẩn hoá dữ liệu

- **Phương pháp:** `RobustScaler` (scikit-learn) — ít nhạy cảm với outliers hơn `StandardScaler`
- **Nguyên tắc chống data leakage:** Scaler được **fit chỉ trên tập train** của mỗi fold, sau đó transform cả train lẫn test. Tuyệt đối không fit trên toàn bộ dữ liệu trước khi chia.

### 3.4. Tạo chuỗi thời gian (Windowing)

`WindowDataset(seq_len=20)` chuyển ma trận đặc trưng 2D `(T, F)` thành tensor 3D phù hợp cho các mô hình sequence:

```
Input shape:  (N, 20, 26)   — N mẫu, cửa sổ 20 ngày, 26 đặc trưng
Target shape: (N,)           — nhãn UP/DOWN tương ứng
```

---

## 4. Thiết kế Thực nghiệm

### 4.1. Walk-Forward Validation

Để mô phỏng điều kiện triển khai thực tế và tránh data leakage từ tương lai, nghiên cứu áp dụng **Walk-Forward Validation** theo sơ đồ expanding window:

```
Fold 1: Train[0  :200] → Test[200:250]   (train=200, test=50)
Fold 2: Train[0  :250] → Test[250:300]   (train=250, test=50)
Fold 3: Train[0  :300] → Test[300:350]   (train=300, test=50)
Fold 4: Train[0  :350] → Test[350:400]   (train=350, test=50)
Fold 5: Train[0  :400] → Test[400:450]   (train=400, test=50)
```

Tập test của mỗi fold là **50 ngày liên tiếp**, không có overlap. Mô hình được huấn luyện lại từ đầu cho mỗi fold (không fine-tuning liên fold).

### 4.2. Đa khởi tạo ngẫu nhiên (Multi-seed)

Để đánh giá tính ổn định và giảm thiên lệch do khởi tạo, mỗi mô hình được huấn luyện với **10 seeds** (0–9) trên 5 folds → **50 giá trị Balanced Accuracy** độc lập/mô hình. Đây là cỡ mẫu đủ lớn để thực hiện kiểm định t-test có độ tin cậy cao.

`set_full_determinism(seed)` kiểm soát: Python random, NumPy, TensorFlow/Keras và CUDA (nếu có).

**Lưu ý thiết kế quan trọng:** Với Logistic Regression, `solver='lbfgs'` là deterministic hoàn toàn — 10 seeds cho ra cùng 1 kết quả, làm n hiệu lực chỉ là 5. Để khắc phục, nghiên cứu chuyển sang `SGDClassifier(loss='log_loss')`, vốn stochastic theo seed, đảm bảo n=50 thực sự.

### 4.3. Chỉ số đánh giá

| Chỉ số | Công thức | Lý do chọn |
|---|---|---|
| **Balanced Accuracy** | `(Recall_UP + Recall_DOWN) / 2` | Chỉ số chính — không bị ảnh hưởng bởi mất cân bằng nhãn |
| Accuracy | `(TP+TN) / N` | Tham khảo |
| Precision, Recall, F1 | Chuẩn | Phân tích lỗi chi tiết |

**Baseline ngẫu nhiên:** Balanced Accuracy = **0.500** (classifier đoán ngẫu nhiên đều tay).

### 4.4. Kiểm định thống kê

- **One-sample t-test** (`ttest_1samp`): mô hình có thực sự vượt baseline 0.5 không?
- **Paired t-test** (`ttest_rel`): so sánh hai mô hình trên cùng 50 fold × seed, loại bỏ variance do dữ liệu.
- Mức ý nghĩa: α = 0.05 (\* p<0.05, \*\* p<0.01, \*\*\* p<0.001)

---

## 5. Kiến trúc Mô hình

### 5.1. Logistic Regression

`LinearForecaster` flatten cửa sổ `(20, 26)` thành vector 520 chiều, sau đó huấn luyện `SGDClassifier(loss='log_loss')`:

```
Input: (N, 20×26=520) → SGD Logistic → sigmoid → P(UP)
```

### 5.2. RNN (Simple RNN)

```
Input (N,20,26) → SimpleRNN(32, return_sequences=False) → Dropout(0.1) → Dense(1, sigmoid)
```

### 5.3. LSTM

```
Input (N,20,26) → LSTM(32, return_sequences=False) → Dropout(0.1) → Dense(1, sigmoid)
```

LSTM sử dụng các cổng forget/input/output để ghi nhớ phụ thuộc dài hạn, giải quyết vanishing gradient của RNN thuần.

### 5.4. BiLSTM (Bidirectional LSTM)

```
Input (N,20,26) → Bidirectional(LSTM(32)) → Dropout(0.1) → Dense(1, sigmoid)
```

Lớp `Bidirectional` chạy hai LSTM song song: một xuôi chiều thời gian, một ngược chiều. Output được nối (concatenate) → vector 64 chiều. Trong cửa sổ 20 ngày, điều này cho phép mô hình khai thác pattern từ cả hai đầu mút của window.

### 5.5. Hyperparameters chung (KerasForecaster)

| Tham số | Giá trị |
|---|---|
| Units | 32 |
| Dropout | 0.1 |
| Epochs | 40 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Binary crossentropy |
| Early stopping | Tắt (không có validation set trong mỗi fold) |

---

## 6. Kết quả Thực nghiệm — Mô hình Gốc

### 6.1. Balanced Accuracy — Walk-Forward Validation

Kết quả trên **50 quan sát độc lập** (5 folds × 10 seeds) cho mỗi mô hình:

| Mô hình | Mean BalAcc | Std | Min | Max |
|---|---|---|---|---|
| **BiLSTM** | **0.5730** | 0.0801 | 0.3815 | 0.7597 |
| LSTM | 0.5524 | 0.0556 | 0.3815 | 0.7241 |
| RNN | 0.4727 | 0.0941 | 0.2791 | 0.6715 |
| Logistic Regression | 0.4619 | 0.0854 | 0.3036 | 0.6051 |
| *Baseline ngẫu nhiên* | *0.5000* | — | — | — |

### 6.2. Kiểm định thống kê — One-sample t-test vs baseline 0.5

| Mô hình | t-statistic | p-value | Kết luận |
|---|---|---|---|
| BiLSTM | +6.384 | < 0.001 | Vượt baseline \*\*\* |
| LSTM | +6.596 | < 0.001 | Vượt baseline \*\*\* |
| RNN | −2.030 | 0.048 | **Kém** baseline \* |
| Logistic Regression | −3.123 | 0.003 | **Kém** baseline \*\* |

BiLSTM và LSTM vượt baseline ngẫu nhiên với mức ý nghĩa rất cao. RNN và Logistic Regression thực tế kém hơn baseline — do mô hình có xu hướng thiên về dự báo một lớp duy nhất trong nhiều fold.

### 6.3. Kiểm định thống kê — Paired t-test giữa các mô hình (n=50)

| So sánh | Diff (Mean) | t-statistic | p-value | Kết luận |
|---|---|---|---|---|
| BiLSTM vs LSTM | +0.0207 | +1.746 | 0.087 | Không significant (p>0.05) |
| BiLSTM vs RNN | +0.1003 | +5.019 | < 0.001 | BiLSTM tốt hơn rõ rệt \*\*\* |
| BiLSTM vs Logistic | +0.1112 | +7.506 | < 0.001 | BiLSTM tốt hơn rõ rệt \*\*\* |
| LSTM vs RNN | — | — | < 0.001 | LSTM tốt hơn \*\*\* |
| LSTM vs Logistic | — | — | < 0.001 | LSTM tốt hơn \*\*\* |
| RNN vs Logistic | — | — | 0.622 | Không khác biệt |

**Phát hiện chính:** BiLSTM không thể khẳng định vượt LSTM tại α=0.05 (p=0.087). Hai mô hình này tạo thành **nhóm dẫn đầu** tách biệt với nhóm RNN/Logistic.

---

## 7. Kết quả Thực nghiệm — Kỹ thuật Ensemble

Sau khi xác định BiLSTM là mô hình tốt nhất, nghiên cứu thử nghiệm các kỹ thuật ensemble nhằm giảm variance và cải thiện mean Balanced Accuracy.

### 7.1. Seed Ensemble (Trung bình hoá nhiều seeds)

Thay vì dùng một model đơn, seed ensemble lấy **trung bình xác suất** từ nhiều seeds cùng fold.

**Filtered Ensemble** (chỉ giữ các fold có train size ≥ 300 để đảm bảo đủ dữ liệu):

| Mô hình | n folds | Mean BalAcc |
|---|---|---|
| BiLSTM Seed Ensemble (full, filtered folds) | 3 | 0.5298 |
| BiLSTM Group-5 Ensemble | 12 | 0.5449 |
| LSTM Seed Ensemble (filtered folds) | 3 | 0.5318 |
| RNN Seed Ensemble (filtered folds) | 3 | 0.5273 |

Ensemble với nhóm 5 seeds (Group-5) cho kết quả 0.5449, thấp hơn BiLSTM đơn (0.5730). Nguyên nhân: số folds sau khi lọc chỉ còn 3, không đủ để đánh giá.

### 7.2. Bagging (Bootstrap Aggregating)

BiLSTM Bagging (5 folds, trung bình nhiều bootstrap samples):

| Chỉ số | Giá trị |
|---|---|
| Mean Balanced Accuracy | 0.5544 |
| Std | 0.0581 |
| n (folds) | 5 |

Bagging (0.5544) thấp hơn BiLSTM đơn tốt nhất (0.5730). Std giảm nhẹ so với đơn (0.0801 → 0.0581), cho thấy bagging có hiệu quả ổn định hoá nhưng không tăng mean.

### 7.3. Các biến thể khác của BiLSTM

| Biến thể | Mean BalAcc | Std | Nhận xét |
|---|---|---|---|
| **BiLSTM Gốc** | **0.5730** | 0.0801 | Tốt nhất |
| BiLSTM More Data (train lớn hơn) | 0.5176 | 0.0854 | Không cải thiện |
| BiLSTM Stable (dropout cao hơn) | 0.4944 | 0.0903 | Kém hơn rõ |
| BiLSTM Regularized (L2) | 0.4913 | 0.0590 | Std nhỏ nhưng mean kém |

### 7.4. Tổng hợp so sánh ensemble vs gốc

| Phương pháp | Mean BalAcc | So với BiLSTM gốc |
|---|---|---|
| BiLSTM gốc (best baseline) | **0.5730** | — |
| BiLSTM Bagging | 0.5544 | −0.0186 |
| BiLSTM Group-5 Ensemble | 0.5449 | −0.0281 |
| BiLSTM Seed Ensemble (filtered) | 0.5298 | −0.0432 |

**Kết luận:** Không có kỹ thuật ensemble nào vượt BiLSTM đơn trên bộ dữ liệu này. Ensemble giúp giảm std nhưng không cải thiện mean Balanced Accuracy.

---

## 8. Nghiên cứu Ablation — Macro Features

### 8.1. Động lực

Nghiên cứu đặt câu hỏi: liệu dữ liệu vĩ mô (Gold price, US Dollar Index, US 10Y Yield) có mang thêm tín hiệu dự báo không? Thay vì tích hợp trực tiếp, một ablation study hệ thống gồm 3 thử nghiệm độc lập được thực hiện.

### 8.2. Thử nghiệm 1 — Phân tích tương quan (n=992 ngày)

Tính Pearson correlation và Spearman correlation giữa từng macro feature (levels, daily returns, lagged returns 1–7 ngày) và BTC direction target (horizon=7 ngày):

| Feature | Pearson r | p-value |
|---|---|---|
| Gold (level) | −0.1245 | < 0.001 \*\*\* |
| US Yield return, lag-1 | −0.0483 | 0.127 |
| DXY (level) | +0.0438 | 0.167 |
| Gold/DXY/Yield daily return | < 0.040 | > 0.20 |
| Tất cả lagged returns (lag 3,5,7) | < 0.040 | > 0.22 |

Rolling 60-day correlation của `gold_return` với target: mean=+0.031, **std=0.133**, pos%=57% — dao động mạnh và không nhất quán về chiều, cho thấy không có tín hiệu ổn định.

**Kết luận thử nghiệm 1:** Gold level có tương quan thống kê (p<0.001) nhưng |r|=0.12 quá nhỏ (R²≈1.5%) để có ý nghĩa thực tiễn. Hầu hết macro features không có tương quan tuyến tính với target.

### 8.3. Thử nghiệm 2 — Walk-forward với Logistic Regression (n=25)

So sánh Logistic Regression với và không có macro features (5 seeds × 5 folds):

| Variant | Mean BalAcc | Std |
|---|---|---|
| Feature Set B, **không macro** | 0.5250 | 0.0806 |
| Feature Set B, **có macro** (level + return) | 0.4577 | 0.0864 |

Paired t-test: diff = −0.0673, t = −2.80, **p = 0.012** \*

Macro features làm **giảm** Balanced Accuracy của Logistic Regression một cách có ý nghĩa thống kê.

### 8.4. Thử nghiệm 3 — Walk-forward với BiLSTM (n=25)

So sánh BiLSTM với và không có macro returns (dạng stationary: `gold_change`, `dxy_change`, `rate_change`):

| Variant | n features | Mean BalAcc | Std |
|---|---|---|---|
| Feature Set B, **không macro** (20 features) | 20 | 0.5706 | 0.0807 |
| Feature Set B, **+ macro returns** (23 features) | 23 | 0.5703 | 0.0851 |

Paired t-test: diff = −0.0003, t = −0.015, **p = 0.989** (ns)

### 8.5. Tổng hợp Ablation

| Thử nghiệm | Kết quả | Kết luận |
|---|---|---|
| Correlation | |r| < 0.05 với hầu hết features | Không có tín hiệu tuyến tính |
| Logistic walk-forward | −0.0673, p=0.012 | Macro làm **giảm** accuracy |
| BiLSTM walk-forward | −0.0003, p=0.989 | Macro không có tác động |

**Kết luận ablation:** Macro features (Gold/DXY/Yield) không cải thiện và có thể làm hại hiệu suất mô hình. Ba nguyên nhân chính:
1. Tương quan tuyến tính gần 0 với target (không có predictive power trực tiếp)
2. Macro data chứa nhiều giá trị weekend-fill (forward-fill từ Thứ Sáu), làm giả tạo autocorrelation
3. Mối quan hệ BTC–macro thay đổi theo regime thị trường, không ổn định trong suốt giai đoạn nghiên cứu

---

## 9. Phân tích và Thảo luận

### 9.1. Tại sao BiLSTM và LSTM hoạt động tốt hơn?

BiLSTM học được pattern từ cả hai chiều trong cửa sổ 20 ngày — ví dụ, một đảo chiều đang diễn ra có thể được phát hiện rõ hơn khi nhìn từ cả hai đầu. Tuy nhiên, sự chênh lệch so với LSTM đơn hướng (diff=+0.021, p=0.087) chưa đạt ý nghĩa thống kê, cho thấy lợi thế của BiLSTM có thể không đủ ổn định để kết luận tổng quát.

### 9.2. Tại sao RNN và Logistic Regression kém baseline?

- **RNN:** Vanishing gradient khiến mô hình không học được pattern xa trong chuỗi 20 ngày. Trong nhiều fold, RNN hội tụ về giải pháp dự báo toàn bộ một lớp (recall một lớp ~1.0, lớp kia ~0.0).
- **Logistic Regression:** Mô hình tuyến tính không đủ khả năng nắm bắt mối quan hệ phi tuyến trong thị trường crypto. Flatten chuỗi 20×26 thành vector 520 chiều tạo không gian quá lớn cho dữ liệu train nhỏ (200–400 mẫu), dẫn đến overfitting hoặc underfitting.

### 9.3. Tại sao ensemble không cải thiện?

Trong lý thuyết, ensemble cải thiện khi các mô hình thành phần có correlation lỗi thấp và mean accuracy cao. Trên bộ dữ liệu này:
- Các fold train nhỏ (200–250 mẫu) khiến các seed khác nhau học ra các model không đủ đa dạng
- Sau khi lọc chỉ còn 3 folds (train ≥ 300), không đủ dữ liệu để đánh giá

### 9.4. Hạn chế của nghiên cứu

1. **Dữ liệu ngắn:** 1.000 ngày (~2.75 năm) bao gồm cả giai đoạn bull và bear, nhưng chưa đủ để kiểm tra tính tổng quát qua nhiều chu kỳ thị trường.
2. **Chi phí giao dịch:** Balanced Accuracy đo khả năng phân loại, chưa tính đến spread, phí và slippage trong thực tế giao dịch.
3. **Lookback cố định:** Cửa sổ seq_len=20 ngày chưa được tối ưu hoá — có thể 30 hoặc 60 ngày phù hợp hơn cho horizon 7 ngày.
4. **Tham số chưa tối ưu:** Hyperparameters (units, dropout, lr, epochs) dùng cấu hình cố định, chưa qua Bayesian optimization hoặc grid search nghiêm ngặt.

---

## 10. Kết luận

### 10.1. Tóm tắt kết quả

| Câu hỏi nghiên cứu | Kết quả |
|---|---|
| Mô hình tốt nhất? | **BiLSTM** (BalAcc=0.573, p<0.001) — nhưng không vượt LSTM có ý nghĩa |
| Ensemble có giúp? | Không — không có kỹ thuật ensemble nào vượt BiLSTM đơn |
| Macro features có giúp? | Không — không có tác động (BiLSTM) hoặc làm hại (Logistic) |

### 10.2. Kết luận chính

1. **BiLSTM và LSTM** là hai mô hình duy nhất vượt baseline ngẫu nhiên có ý nghĩa thống kê. Balanced Accuracy ~0.55–0.57 cho thấy khả năng dự báo nhất định nhưng còn khiêm tốn.

2. **Mức độ khả năng dự báo thực tế:** Balanced Accuracy 0.573 nghĩa là trung bình BiLSTM đúng khoảng 57.3% trong việc phân loại chiều hướng BTC — cao hơn ngẫu nhiên nhưng chưa đủ cao cho ứng dụng giao dịch thực tế mà không có quản lý rủi ro bổ sung.

3. **Macro features không hữu ích** với dữ liệu daily và horizon 7 ngày trong bối cảnh này. Đây là kết quả nhất quán qua cả 3 thử nghiệm độc lập.

4. **Phương pháp đánh giá nghiêm ngặt** (Walk-Forward + multi-seed + t-test) là cần thiết: đánh giá trên single split hoặc không kiểm định thống kê sẽ đưa ra kết luận sai lệch.

### 10.3. Hướng phát triển tiếp theo

- **Mô hình:** Thử nghiệm Transformer với multi-head attention; kết hợp TCN (Temporal Convolutional Network)
- **Dữ liệu:** Bổ sung on-chain metrics (hash rate, exchange flows, whale activity); dữ liệu sentiment từ mạng xã hội
- **Feature engineering:** Tối ưu seq_len và horizon; market regime detection để train mô hình có điều kiện
- **Đánh giá:** Thêm backtest thực tế với transaction cost và position sizing; Sharpe ratio làm chỉ số bổ sung
- **Ensemble nâng cao:** Stacking, boosting với Gradient Boosted Trees làm meta-learner

---

## 11. Tài liệu Tham khảo

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

2. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

4. Prado, M. L. de (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

5. Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4).

6. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825–2830.

8. Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. *12th USENIX Symposium on Operating Systems Design and Implementation*.

9. Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System. *bitcoin.org/bitcoin.pdf*.

10. Achelis, S. B. (2001). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.

---

*Toàn bộ code, dữ liệu thực nghiệm và kết quả có thể tái lập tại:*  
*https://github.com/lekimphu0209/crypto-price-prediction*
