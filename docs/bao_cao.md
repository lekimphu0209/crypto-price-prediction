# BÁO CÁO ĐỒ ÁN: DỰ BÁO HƯỚNG GIÁ CRYPTOCURRENCY BẰNG MÔ HÌNH MACHINE LEARNING

> **Lưu ý**: File `docs/bao_cao.md` được tạo vì thư mục `report/` nằm trong `.gitignore`. Nội dung này có thể được sao chép vào `DAMH_1.docx` để chỉnh sửa.

---

## MỤC LỤC

- CHƯƠNG 1: GIỚI THIỆU
- CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
- CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
- CHƯƠNG 4: CÀI ĐẶT VÀ THỰC NGHIỆM
- CHƯƠNG 5: ĐÁNH GIÁ VÀ KẾT LUẬN
- TÀI LIỆU THAM KHẢO
- PHỤ LỤC

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1. Đặt vấn đề

Thị trường tiền điện tử (Cryptocurrency) là một trong những thị trường tài chính có tính biến động cao nhất hiện nay. Giá của Bitcoin, Ethereum và các đồng tiền số khác chịu ảnh hưởng bởi nhiều yếu tố như cung cầu, tâm lý thị trường, tin tức vĩ mô, chính sách quản lý và dòng tiền đầu tư. Vì vậy, bài toán dự báo hướng đi của giá crypto là một bài toán có ý nghĩa thực tiễn nhưng cũng đầy thách thức.

Đồ án này tập trung vào việc dự báo **hướng giá** (lên/xuống) của Bitcoin trong ngắn hạn dựa trên dữ liệu OHLCV và các chỉ báo kỹ thuật, sử dụng các mô hình Machine Learning và Deep Learning.

## 1.2. Mục tiêu

- Xây dựng pipeline thu thập, xử lý dữ liệu và huấn luyện mô hình.
- So sánh hiệu suất của các mô hình: Logistic Regression, RNN, LSTM, BiLSTM.
- Đánh giá mô hình bằng Walk-Forward Validation và các kiểm định thống kê (t-test).
- Cải thiện độ ổn định của mô hình bằng các kỹ thuật ensemble.

## 1.3. Phạm vi

- **Dữ liệu**: BTCUSDT, khung thời gian daily, từ Binance.
- **Mục tiêu dự báo**: Hướng giá ngày tiếp theo (Up/Down).
- **Mô hình**: Logistic Regression, RNN, LSTM, BiLSTM.
- **Đánh giá**: Balanced Accuracy, Accuracy, Precision, Recall, F1, t-test.

## 1.4. Phương pháp

- Thu thập dữ liệu OHLCV từ Binance.
- Trích xuất đặc trưng kỹ thuật (MA, RSI, MACD, Bollinger Bands, Volume).
- Chuẩn hóa dữ liệu bằng RobustScaler.
- Xây dựng cửa sổ thời gian (seq_len=20) cho các mô hình sequence.
- Huấn luyện với Walk-Forward Validation (5 folds, 10 seeds).
- So sánh bằng t-test một mẫu và t-test cặp.

## 1.5. Cấu trúc báo cáo

- **Chương 2**: Cơ sở lý thuyết về crypto, time series, các mô hình và metric.
- **Chương 3**: Phân tích và thiết kế hệ thống.
- **Chương 4**: Cài đặt và thực nghiệm.
- **Chương 5**: Đánh giá, kết luận và hướng phát triển.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Tổng quan về Cryptocurrency

Cryptocurrency là tài sản số sử dụng công nghệ blockchain để ghi nhận giao dịch. Bitcoin (BTC) và Ethereum (ETH) là hai đồng có vốn hóa lớn, thanh khoản cao và dữ liệu lịch sử đầy đủ, phù hợp cho nghiên cứu.

## 2.2. Chuỗi thời gian (Time Series)

Bài toán dự báo giá crypto là bài toán **time series**. Trong đó, thứ tự thời gian rất quan trọng. Không thể xáo trộn dữ liệu như các bài toán ML thông thường vì sẽ gây rò rỉ thông tin tương lai.

## 2.3. Các chỉ số đánh giá

### 2.3.1. MAE

Mean Absolute Error (MAE) là sai số tuyệt đối trung bình. MAE càng thấp thì dự báo càng gần giá trị thực.

### 2.3.2. MAPE

Mean Absolute Percentage Error (MAPE) là sai số phần trăm trung bình, giúp so sánh giữa các tài sản có scale giá khác nhau.

### 2.3.3. RMSE

Root Mean Squared Error (RMSE) là căn bậc hai của sai số bình phương trung bình, phạt mạnh các lỗi lớn.

### 2.3.4. R²

R² đo mức độ mô hình giải thích được biến động dữ liệu. R² gần 1 thì fit tốt. Tuy nhiên, trong tài chính, R² cao có thể do autocorrelation giá nên cần đánh giá kỹ.

### 2.3.5. MASE

Mean Absolute Scaled Error (MASE) so sánh lỗi mô hình với lỗi của naive baseline. MASE < 1 nghĩa là mô hình tốt hơn baseline.

### 2.3.6. Balanced Accuracy

Trong phân loại hướng giá, dữ liệu thường mất cân bằng. **Balanced Accuracy** là trung bình của recall cho mỗi lớp, phù hợp để đánh giá mô hình phân loại lên/xuống.

## 2.4. Mô hình Logistic Regression

Logistic Regression là mô hình tuyến tính cho bài toán phân loại. Mô hình tính xác suất của lớp dương (UP) thông qua hàm sigmoid.

Trong đồ án, `LinearForecaster` sử dụng `LogisticRegression` hoặc `SGDClassifier(loss='log_loss')` để dự báo hướng giá.

## 2.5. Mô hình RNN

Recurrent Neural Network (RNN) là mạng nơ-ron tuần tự, truyền hidden state qua từng bước thời gian. RNN phù hợp với dữ liệu chuỗi thời gian nhưng gặp vấn đề vanishing gradient khi chuỗi dài.

## 2.6. Mô hình LSTM

Long Short-Term Memory (LSTM) cải thiện RNN bằng các cổng (forget, input, output) để ghi nhớ thông tin dài hạn. LSTM phù hợp với time series tài chính.

## 2.7. Mô hình BiLSTM

Bidirectional LSTM (BiLSTM) sử dụng hai LSTM: một chạy xuôi, một chạy ngược trong cửa sổ dữ liệu. Kết quả được nối lại để khai thác cả pattern quá khứ và tương lai trong window.

## 2.8. Mô hình Transformer

Transformer sử dụng attention mechanism để học mức độ quan trọng giữa các timestep. Mô hình này có tiềm năng tốt cho sequence dài nhưng trong đồ án chưa được tối ưu đánh giá chi tiết.

---

# CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1. Yêu cầu hệ thống

Hệ thống cần:
- Thu thập dữ liệu OHLCV real-time từ Binance.
- Trích xuất đặc trưng kỹ thuật.
- Huấn luyện nhiều mô hình dự báo hướng giá.
- Đánh giá bằng Walk-Forward Validation.
- Lưu trữ model, scaler và metadata.
- Hỗ trợ API và Dashboard.

## 3.2. Kiến trúc hệ thống

Hệ thống được xây dựng theo **Clean Architecture** và **SOLID principles**:
- `src/domain/`: interfaces và entities.
- `src/features/`: indicators và technical extractor.
- `src/training/`: pipeline, data loader, target builder, window dataset.
- `src/infrastructure/`: data providers, forecasters, model registry.
- `src/api/`: FastAPI routes.
- `dashboard/`: Streamlit dashboard.

## 3.3. Thiết kế mô hình

### 3.3.1. Mô hình Logistic Regression

`LinearForecaster` flatten cửa sổ `(seq_len, n_features)` thành vector và huấn luyện Logistic Regression. Khi truyền `seed`, mô hình sử dụng `SGDClassifier(loss='log_loss')` để đảm bảo tính ngẫu nhiên giữa các seed.

### 3.3.2. Mô hình RNN

`KerasForecaster` xây dựng `SimpleRNN` với `units=32`, `dropout=0.1`, `epochs=40`, `batch_size=16`, `learning_rate=0.001`.

### 3.3.3. Mô hình LSTM

`KerasForecaster` xây dựng `LSTM` với cùng cấu hình trên.

### 3.3.4. Mô hình BiLSTM

`KerasForecaster` xây dựng `Bidirectional(LSTM)` để học hai chiều trong cửa sổ 20 ngày.

### 3.3.5. Mô hình Transformer

`KerasForecaster` hỗ trợ Transformer nhưng chưa được tối ưu trong thực nghiệm chính.

### 3.3.6. Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| SYMBOL | BTCUSDT |
| FEATURE_SET | B |
| SEQ_LEN | 20 |
| HORIZON | 1 |
| N_FOLDS | 5 |
| INITIAL_TRAIN_SIZE | 200 |
| TEST_SIZE | 50 |
| N_SEEDS | 10 |
| Units | 32 |
| Dropout | 0.1 |
| Learning rate | 0.001 |
| Epochs | 40 |
| Batch size | 16 |

### 3.4.7. Xây dựng biến mục tiêu dự báo

`TrendTargetBuilder` tính giá tương lai và gán nhãn UP (1) nếu giá tăng, DOWN (0) nếu giá giảm so với giá hiện tại.

---

# CHƯƠNG 4: CÀI ĐẶT VÀ THỰC NGHIỆM

## 4.1. Môi trường thực nghiệm

- Python 3.10+
- TensorFlow/Keras, scikit-learn, pandas, numpy
- Windows 11, CPU
- Thư viện chính: `ccxt`, `yfinance`, `ta`, `scikit-learn`, `tensorflow`

## 4.2.1. Thu thập dữ liệu

`MarketDataLoader` sử dụng `BinanceProvider` để lấy dữ liệu OHLCV. Dữ liệu gồm 1000 nến daily, bao gồm Open, High, Low, Close, Volume.

## 4.2.2. Trích xuất đặc trưng kỹ thuật

`TechnicalExtractor` tạo feature set B gồm:
- close, volume
- ma_7, ma_14
- rsi
- macd
- bb_upper, bb_middle, bb_lower

## 4.3. Tiền xử lý dữ liệu

### 4.3.1. Phân chia tập dữ liệu

Dữ liệu được chia theo thứ tự thời gian. Walk-Forward Validation chia thành 5 folds:
- Fold 1: Train[0:200], Test[200:250]
- Fold 2: Train[0:250], Test[250:300]
- Fold 3: Train[0:300], Test[300:350]
- Fold 4: Train[0:350], Test[350:400]
- Fold 5: Train[0:400], Test[400:450]

### 4.3.2. Chuẩn hóa dữ liệu

`RobustScaler` được fit trên tập train và transform toàn bộ dữ liệu. Điều này tránh data leakage từ tập test.

### 4.3.3. Xây dựng chuỗi thời gian

`WindowDataset` tạo các chuỗi `(seq_len=20, n_features=26)` để đưa vào các mô hình sequence.

## 4.4. Huấn luyện mô hình

### 4.4.1. Thiết lập thực nghiệm

Các mô hình được huấn luyện với 10 seed khác nhau trên 5 folds, tạo thành 50 quan sát cho mỗi mô hình.

### 4.4.2. Cấu hình huấn luyện

- `KerasForecaster` hỗ trợ RNN, LSTM, BiLSTM, Transformer.
- `LinearForecaster` hỗ trợ Logistic Regression.
- `set_full_determinism(seed)` đảm bảo reproducibility.

### 4.4.3. Quy trình huấn luyện

1. Load dữ liệu.
2. Trích xuất đặc trưng.
3. Scale.
4. Windowing.
5. Train/test split theo walk-forward.
6. Huấn luyện model với từng seed.
7. Lưu metrics và so sánh.

### 4.4.4. Cải tiến

- **Ensemble methods**: seed ensemble, weighted ensemble, bagging.
- **Filtered ensemble**: bỏ qua các fold có train size nhỏ hơn 300.
- **Logistic n=50**: sửa lỗi deterministic Logistic bằng `SGDClassifier(log_loss)` để 50 seed/fold thực sự khác nhau.

## 4.5. Kết quả thực nghiệm

### 4.5.1. Kết quả đánh giá trên tập kiểm thử (In-Sample)

Trên tập test thông thường, các mô hình regression đạt R² cao (0.91–0.97). Tuy nhiên, do autocorrelation giá, R² cao chưa đủ để kết luận khả năng dự báo thực tế.

### 4.5.2. Đánh giá Walk-Forward Validation (Out-of-Sample)

Kết quả classification (hướng lên/xuống) trên walk-forward:

| Model | n | Mean BalAcc | Std | p vs 0.5 |
|---|---|---|---|---|
| BiLSTM | 50 | 0.5730 | 0.0801 | <0.001 *** |
| LSTM | 50 | 0.5524 | 0.0556 | <0.001 *** |
| RNN | 50 | 0.4727 | 0.0941 | 0.0478 * |
| Logistic | 50 | 0.4619 | 0.0854 | 0.0030 ** |

### 4.5.3. So sánh các mô hình

Bảng t-test cặp (n=50):

| So sánh | p | Ý nghĩa |
|---|---|---|
| BiLSTM vs LSTM | 0.087 | Không significant |
| BiLSTM vs RNN | <0.001 | BiLSTM thắng |
| BiLSTM vs Logistic | <0.001 | BiLSTM thắng |
| LSTM vs RNN | <0.001 | LSTM thắng |
| LSTM vs Logistic | <0.001 | LSTM thắng |
| RNN vs Logistic | 0.622 | Không khác biệt |

---

# CHƯƠNG 5: ĐÁNH GIÁ VÀ KẾT LUẬN

## 5.1. Đánh giá kết quả

- **BiLSTM** là mô hình tốt nhất với mean Balanced Accuracy 0.5730, vượt baseline rõ rệt.
- **LSTM** xếp thứ 2 (0.5524), vẫn vượt baseline.
- **RNN** và **Logistic Regression** đều kém hơn baseline, mặc dù Logistic có n=50 thực sự sau khi sửa lỗi deterministic.
- BiLSTM không thể khẳng định vượt LSTM với p=0.087 (chưa đạt ý nghĩa thống kê ở α=0.05).
- Ensemble giúp giảm variance nhưng chưa cải thiện mean balanced accuracy.

## 5.2. Kết luận

Đồ án đã xây dựng thành công một pipeline dự báo hướng giá cryptocurrency theo Clean Architecture, với nhiều mô hình và đánh giá nghiêm ngặt bằng Walk-Forward Validation và t-test. Kết quả cho thấy LSTM và BiLSTM vượt baseline nhưng hiệu quả còn hạn chế. Các mô hình đơn giản như Logistic Regression không đủ khả năng phân loại hướng giá trong bối cảnh dữ liệu nhiễu cao.

## 5.3. Hướng phát triển

- Bổ sung dữ liệu sentiment từ tin tức và mạng xã hội.
- Bổ sung dữ liệu vĩ mô (Gold, DXY, lãi suất).
- Bổ sung on-chain metrics (cho BTC/ETH).
- Tối ưu Transformer và thử nghiệm Attention-based models.
- Thêm backtesting với transaction cost và position sizing.
- Thử nghiệm thêm các kỹ thuật ensemble nâng cao (stacking, boosting).

---

# TÀI LIỆU THAM KHẢO

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE TP.
3. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
4. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. IJF.
5. Scikit-learn documentation: https://scikit-learn.org/
6. TensorFlow/Keras documentation: https://www.tensorflow.org/

---

# PHỤ LỤC

## Phụ lục A: Cấu trúc thư mục dự án

```
crypto-price-prediction/
├── src/
│   ├── api/
│   ├── domain/
│   ├── features/
│   ├── infrastructure/
│   ├── training/
│   └── utils/
├── scripts/
├── models/
├── dashboard/
├── report/
├── docs/
└── tests/
```

## Phụ lục B: Các script thực nghiệm chính

- `scripts/train_btc_walkforward.py`: RNN, LSTM.
- `scripts/train_btc_walkforward_bilstm.py`: BiLSTM.
- `scripts/train_btc_walkforward_logistic.py`: Logistic Regression.
- `scripts/compare_all_models_ttest.py`: So sánh t-test.
- `scripts/train_btc_walkforward_bilstm_ensemble_filtered.py`: Ensemble BiLSTM.

## Phụ lục C: Kết quả chi tiết Logistic sau cải tiến

Sau khi sửa `LinearForecaster` để dùng `SGDClassifier(log_loss)` với seed, kết quả Logistic trở nên stochastic:
- Mean BalAcc: 0.4619
- Std: 0.0854
- n: 50
- p-value vs 0.5: 0.0030

Điều này chứng minh n=50 giờ đã là thực sự, không còn bị gấp đôi do 10 seed cho ra kết quả giống nhau.
