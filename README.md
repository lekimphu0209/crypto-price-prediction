# Dự Đoán Giá Bitcoin Sử Dụng Hồi Quy và RNN

Một dự án machine learning hoàn chỉnh từ đầu đến cuối để dự đoán giá Bitcoin sử dụng các mô hình Hồi quy Tuyến tính (Linear Regression), RNN Đơn giản (Simple RNN), và LSTM. Dự án này được thiết kế cho mục đích học thuật và có thể được sử dụng làm nền tảng cho nghiên cứu về AI/Deep Learning áp dụng vào chuỗi thời gian tài chính.

## 📋 Mục Lục

- [Tổng quan dự án](#tổng-quan-dự án)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Các mô hình](#các-mô-hình)
- [Chỉ số kỹ thuật](#chỉ-số-kỹ-thuật)
- [Đánh giá mô hình](#đánh-giá-mô-hình)
- [Kết quả](#kết-quả)
- [Hướng phát triển](#hướng-phát-triển)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Giấy phép](#giấy-phép)

## 🎯 Tổng quan dự án

Dự án này triển khai các phương pháp khác nhau để dự đoán giá Bitcoin:

1. **Hồi quy Tuyến tính (Linear Regression)** - Mô hình cơ bản (đã triển khai)
2. **RNN Đơn giản (Simple RNN)** - Mạng nơ-ron hồi quy (tùy chọn, cần TensorFlow)
3. **LSTM** - Mạng nhớ ngắn hạn dài hạn (tùy chọn, cần TensorFlow)

Dự án tuân theo quy trình machine learning hoàn chỉnh:

- Thu thập dữ liệu từ Yahoo Finance
- Kỹ thuật đặc trưng với chỉ số kỹ thuật
- Tiền xử lý và chuẩn hóa dữ liệu
- Huấn luyện và đánh giá mô hình
- Trực quan hóa và so sánh

## ✨ Tính năng

- **Quy trình hoàn chỉnh**: Triển khai từ đầu đến cuối từ thu thập dữ liệu đến đánh giá mô hình
- **Nhiều mô hình**: Linear Regression (cơ bản), RNN và LSTM (tùy chọn)
- **Chỉ số kỹ thuật**: RSI, MA20, MA50, MACD, ATR và nhiều hơn nữa
- **Đánh giá toàn diện**: Các chỉ số MSE, RMSE, MAE, MAPE, R²
- **Trực quan hóa chuyên nghiệp**: Lịch sử giá, chỉ số, dự đoán, biểu đồ phần dư
- **Code mô-đun**: Tổ chức theo module (data, models, evaluation, visualization)
- **Chất lượng học thuật**: Phù hợp cho các dự án đại học và nghiên cứu
- **Cấu trúc tách biệt**: Dữ liệu thô, dữ liệu xử lý, models, results được phân chia rõ ràng

## 📁 Cấu trúc dự án

```
project/
│
├── data/                          # Thư mục dữ liệu
│   ├── raw/                       # Dữ liệu thô từ nguồn
│   │   └── bitcoin_data.csv       # Dữ liệu OHLCV gốc
│   ├── processed/                 # Dữ liệu đã xử lý
│   │   ├── bitcoin_data_with_features.csv  # Dữ liệu với đặc trưng
│   │   ├── X_train.npy            # Dữ liệu huấn luyện
│   │   ├── X_test.npy             # Dữ liệu kiểm tra
│   │   ├── y_train.npy            # Nhãn huấn luyện
│   │   └── y_test.npy             # Nhãn kiểm tra
│   └── external/                  # Dữ liệu bên ngoài (nếu có)
│
├── src/                           # Source code
│   ├── data/                      # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_collection.py     # Thu thập dữ liệu
│   │   ├── feature_engineering.py # Kỹ thuật đặc trưng
│   │   └── data_preprocessing.py  # Tiền xử lý
│   ├── models/                    # Module mô hình
│   │   ├── __init__.py
│   │   ├── linear_regression_model.py
│   │   ├── rnn_model.py
│   │   ├── lstm_model.py
│   │   └── *.pkl                  # Mô hình đã huấn luyện
│   ├── evaluation/                # Module đánh giá
│   │   ├── __init__.py
│   │   └── evaluation.py
│   └── visualization/             # Module trực quan hóa
│       ├── __init__.py
│       └── visualization.py
│
├── notebooks/                     # Jupyter Notebooks
│   └── bitcoin_price_prediction_vn.ipynb
│
├── results/                       # Kết quả và biểu đồ
│   ├── *.png                      # Biểu đồ trực quan hóa
│   ├── *.csv                      # Bảng so sánh
│   └── *.npy                      # Dự đoán
│
├── report/                        # Báo cáo học thuật
│   └── report_outline_vn.md
│
├── main.py                        # Script chính chạy pipeline
├── requirements.txt               # Dependencies
├── Chuẩn bị dữ liệu.pdf           # Tài liệu hướng dẫn chuẩn bị dữ liệu
├── Pythonic Quant A Comprehensive Guide to Python in Finance.pdf # Sách tham khảo
└── README.md                      # Tài liệu dự án
```

## 🚀 Cài đặt

### Yêu cầu tiên quyết

- Python 3.8 trở lên (khuyến nghị Python 3.10)
- Trình quản lý gói pip
- Kết nối internet (để tải dữ liệu từ Yahoo Finance)

### Thiết lập chi tiết

#### Bước 1: Clone hoặc tải dự án

```bash
# Nếu dùng Git
git clone <repository-url>
cd project

# Hoặc tải và giải nén thư mục dự án
```

#### Bước 2: Tạo môi trường ảo (Khuyến nghị)

Môi trường ảo giúp tách biệt dependencies của dự án với hệ thống:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

Kiểm tra môi trường ảo đã kích hoạt (xem `(venv)` ở đầu dòng lệnh).

#### Bước 3: Cập nhật pip

```bash
python -m pip install --upgrade pip
```

#### Bước 4: Cài đặt dependencies

**Cách 1: Cài đặt dependencies cơ bản (Khuyên dùng cho Linear Regression)**

```bash
pip install -r requirements_basic.txt
```

Đây là cách nhanh nhất và tránh xung đột với TensorFlow. Chỉ cần cho:

- Thu thập dữ liệu
- Kỹ thuật đặc trưng
- Tiền xử lý
- Mô hình Linear Regression
- Đánh giá và trực quan hóa

**Cách 2: Cài đặt đầy đủ (bao gồm TensorFlow cho RNN/LSTM)**

```bash
pip install -r requirements.txt
```

**Lưu ý:** TensorFlow có thể gây xung đột dependencies trên một số hệ thống. Nếu gặp lỗi, hãy dùng Cách 1.

#### Bước 5: Xác minh cài đặt

```bash
# Kiểm tra Python
python --version

# Kiểm tra các thư viện chính
python -c "import pandas; import numpy; import sklearn; print('Dependencies OK')"
```

### Xử lý lỗi thường gặp

#### Lỗi 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Nguyên nhân:** TensorFlow chưa được cài đặt hoặc có xung đột dependencies.

**Giải pháp:**

```bash
# Chỉ cần Linear Regression (không cần TensorFlow)
pip install -r requirements_basic.txt

# Hoặc cài TensorFlow riêng
pip install tensorflow==2.10.0
```

#### Lỗi 2: "resolution-too-deep" hoặc "ResolutionImpossible"

**Nguyên nhân:** Xung đột dependencies khi cài từ requirements.txt.

**Giải pháp:**

```bash
# Cài từng thư viện riêng
pip install yfinance pandas numpy ta scikit-learn matplotlib seaborn tqdm
```

#### Lỗi 3: "Length mismatch: Expected axis has 6 elements, new values have 7 elements"

**Nguyên nhân:** Thay đổi API của yfinance trả về thêm cột 'Adj Close'.

**Giải pháp:** Đã được sửa trong code mới nhất. Hãy cập nhật code hoặc chạy lại.

#### Lỗi 4: Lỗi hiển thị tiếng Việt trên Windows Console

**Nguyên nhân:** Encoding mặc định của Windows Console không hỗ trợ UTF-8.

**Giải pháp:**

```bash
# Trên PowerShell
chcp 65001

# Hoặc thiết lập biến môi trường
set PYTHONIOENCODING=utf-8
```

### Các phụ thuộc

**Cơ bản (cho Linear Regression):**

| Thư viện       | Phiên bản | Mục đích                                                                  |
| ---------------- | ----------- | ---------------------------------------------------------------------------- |
| `yfinance`     | Latest      | Thu thập dữ liệu tài chính từ Yahoo Finance (giá Bitcoin, vàng, DXY) |
| `pandas`       | Latest      | Xử lý dữ liệu dạng DataFrame, thao tác dữ liệu thời gian            |
| `numpy`        | Latest      | Tính toán số học, xử lý mảng đa chiều cho machine learning          |
| `scikit-learn` | Latest      | Tiện ích machine learning: train/test split, metrics, scaler               |
| `matplotlib`   | Latest      | Vẽ biểu đồ cơ bản: lịch sử giá, dự đoán, phần dư               |
| `seaborn`      | Latest      | Trực quan hóa thống kê nâng cao: heatmap, biểu đồ đẹp hơn         |
| `ta`           | Latest      | Thư viện phân tích kỹ thuật: RSI, MACD, ATR, MA                        |
| `tqdm`         | Latest      | Hiển thị thanh tiến trình khi chạy pipeline                             |

**Nâng cao (cho RNN/LSTM):**

| Thư viện     | Phiên bản | Tác dụng                                                 |
| -------------- | ----------- | ---------------------------------------------------------- |
| `tensorflow` | 2.10+       | Framework deep learning cho RNN, LSTM, Transformer         |
| `keras`      | 2.10+       | API mạng nơ-ron cấp cao, xây dựng mô hình dễ dàng |

### Kiểm tra cài đặt thành công

Chạy script chính để kiểm tra:

```bash
python main.py
```

Nếu chạy thành công, bạn sẽ thấy:

```
============================================================
DỰ ĐOÁN GIÁ BITCOIN - PIPELINE HOÀN CHỈNH
============================================================

[1/6] Thu thập dữ liệu...
✓ Dữ liệu đã lưu: data\raw\bitcoin_data.csv

[2/6] Kỹ thuật đặc trưng...
✓ Đặc trưng đã lưu: data\processed\bitcoin_data_with_features.csv

...

============================================================
HOÀN THÀNH!
============================================================
```

## 📖 Sử dụng

Chạy toàn bộ pipeline:

```bash
python main.py
```

Script sẽ tự động:

1. Thu thập dữ liệu từ Yahoo Finance
2. Kỹ thuật đặc trưng với chỉ số kỹ thuật
3. Tiền xử lý và chuẩn hóa dữ liệu
4. Huấn luyện mô hình Linear Regression
5. Đánh giá mô hình
6. Tạo biểu đồ trực quan hóa

Kết quả sẽ được lưu trong thư mục `results/`.

## 🤖 Các mô hình

### 1. Hồi quy Tuyến tính (Linear Regression) - Mô hình cơ bản

**Công thức:**

```
y = w·x + b
```

**Mô tả:**

- Mô hình cơ bản đơn giản và dễ hiểu
- Huấn luyện nhanh
- Cung cấp tiêu chuẩn so sánh cho các mô hình phức tạp hơn
- Sử dụng đặc trưng của ngày hiện tại để dự đoán giá ngày tiếp theo

**Ưu điểm:**

- Dễ hiểu và giải thích
- Hiệu quả về mặt tính toán
- Không cần tinh chỉnh siêu tham số
- Có thể trích xuất tầm quan trọng của đặc trưng

**Nhược điểm:**

- Không thể nắm bắt các mối quan hệ phi tuyến tính
- Không xem xét phụ thuộc thời gian
- Khả năng dự đoán hạn chế cho các mẫu phức tạp

### 2. RNN Đơn giản (Simple RNN)

**Công thức trạng thái ẩn:**

```
hₜ = f(W·xₜ + U·hₜ₋₁ + b)
```

**Kiến trúc:**

- SimpleRNN(50) - 50 đơn vị hồi quy
- Dropout(0.2) - Điều chuẩn (regularization)
- Dense(1) - Lớp đầu ra

**Mô tả:**

- Xử lý chuỗi với bộ nhớ nội bộ
- Nắm bắt phụ thuộc thời gian
- Trạng thái ẩn duy trì thông tin từ các bước thời gian trước đó

**Ưu điểm:**

- Có thể mô hình hóa dữ liệu tuần tự
- Nắm bắt các mẫu thời gian
- Linh hoạt hơn so với hồi quy tuyến tính

**Nhược điểm:**

- Bị vấn đề gradient biến mất (vanishing gradient)
- Khó học các phụ thuộc dài hạn
- Có thể không ổn định cho các chuỗi dài

### 3. LSTM (Long Short-Term Memory)

**Công thức ô nhớ:**

```
fₜ = sigmoid(Wf·[hₜ₋₁, xₜ] + bf)  # Cổng quên (forget gate)
iₜ = sigmoid(Wi·[hₜ₋₁, xₜ] + bi)  # Cổng đầu vào (input gate)
C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)    # Trạng thái ô nhớ ứng viên
Cₜ = fₜ·Cₜ₋₁ + iₜ·C̃ₜ            # Trạng thái ô nhớ
oₜ = sigmoid(Wo·[hₜ₋₁, xₜ] + bo)  # Cổng đầu ra (output gate)
hₜ = oₜ·tanh(Cₜ)                   # Trạng thái ẩn
```

**Kiến trúc:**

- LSTM(50) - 50 đơn vị LSTM với ô nhớ
- Dropout(0.2) - Điều chuẩn (regularization)
- Dense(1) - Lớp đầu ra

**Mô tả:**

- RNN nâng cao với ô nhớ
- Ba cổng kiểm soát luồng thông tin
- Có thể học các phụ thuộc dài hạn

**Ưu điểm:**

- Giải quyết vấn đề gradient biến mất
- Học các phụ thuộc dài hạn
- Huấn luyện ổn định hơn
- Tốt hơn cho các mẫu thời gian phức tạp

**Nhược điểm:**

- Tốn kém hơn về mặt tính toán
- Nhiều tham số để huấn luyện
- Thời gian huấn luyện dài hơn
- Phức tạp hơn để hiểu

## 📊 Chỉ số kỹ thuật

### Đường trung bình (Moving Averages - MA20, MA50)

- **Công thức**: MA = (Tổng giá trong n kỳ) / n
- **Mục đích**: Làm mịn dữ liệu giá để xác định xu hướng
- **Sử dụng**: Tín hiệu cắt nhau của MA (golden cross, death cross)

### RSI (Chỉ số sức mạnh tương đối - Relative Strength Index)

- **Công thức**: RSI = 100 - (100 / (1 + RS)), trong đó RS = Lợi nhuận trung bình / Mất mát trung bình
- **Mục đích**: Đo động lượng và điều kiện mua quá/bán quá
- **Sử dụng**: RSI > 70 (mua quá), RSI < 30 (bán quá)

### MACD (Hội tụ phân kỳ đường trung bình - Moving Average Convergence Divergence)

- **Thành phần**: Đường MACD, Đường tín hiệu, Biểu đồ (Histogram)
- **Mục đích**: Hiển thị sức mạnh và hướng của xu hướng
- **Sử dụng**: MACD cắt lên/xuống đường tín hiệu cho tín hiệu mua/bán

### ATR (Phạm vi thực tế trung bình - Average True Range)

- **Công thức**: ATR = Đường trung bình của True Range
- **Mục đích**: Đo độ biến động
- **Sử dụng**: ATR cao hơn cho thấy độ biến động cao hơn

### Thay đổi giá

- **Công thức**: Thay đổi giá % = ((Giá_t - Giá_{t-n}) / Giá_{t-n}) × 100
- **Mục đích**: Nắm bắt động lượng qua các giai đoạn khác nhau
- **Sử dụng**: Thay đổi ngắn hạn (1 ngày), trung hạn (5 ngày), dài hạn (10 ngày)

### Đặc trưng khối lượng

- **Thành phần**: Thay đổi khối lượng, Khối lượng MA, Tỷ lệ khối lượng
- **Mục đích**: Đo sức mạnh giao dịch
- **Sử dụng**: Khối lượng cao kèm tăng giá = tín hiệu tăng mạnh

## 📈 Chỉ số đánh giá

### MSE (Sai số bình phương trung bình - Mean Squared Error)

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

- Phạt nặng các sai số lớn
- Đơn vị: bình phương (ví dụ: USD²)
- Càng thấp càng tốt

### RMSE (Căn bậc hai của sai số bình phương trung bình - Root Mean Squared Error)

```
RMSE = √MSE
```

- Cùng đơn vị với mục tiêu (ví dụ: USD)
- Dễ hiểu hơn MSE
- Càng thấp càng tốt

### MAE (Sai số tuyệt đối trung bình - Mean Absolute Error)

```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

- Ít nhạy cảm với giá trị ngoại lai
- Dễ hiểu hơn
- Càng thấp càng tốt

### MAPE (Sai số phần trăm tuyệt đối trung bình - Mean Absolute Percentage Error)

```
MAPE = (1/n) × Σ|yᵢ - ŷᵢ| / yᵢ × 100
```

- Độc lập với quy mô
- Được biểu diễn dưới dạng phần trăm
- Càng thấp càng tốt

### R² (Hệ số xác định - R-squared)

```
R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
```

- Tỷ lệ phương sai được giải thích
- Phạm vi: -∞ đến 1
- Càng cao càng tốt (gần 1)

## 📊 Kết quả

Dự án tạo ra các kết quả toàn diện bao gồm:

- **Bảng so sánh mô hình**: File CSV so sánh tất cả mô hình trên tất cả chỉ số
- **Biểu đồ dự đoán**: So sánh trực quan giữa giá dự đoán và giá thực tế
- **Đường cong mất mát**: Mất mát huấn luyện và kiểm tra qua các epoch
- **Biểu đồ phần dư**: Phân phối lỗi dự đoán
- **Bản đồ nhiệt tương quan**: Tương quan giữa các đặc trưng
- **Biểu đồ chỉ số kỹ thuật**: Giá với chỉ số kỹ thuật chồng lên

Ví dụ đầu ra:

```
So sánh mô hình:
+-------------------+----------+----------+----------+----------+----------+
| Mô hình             |      MSE |     RMSE |      MAE |     MAPE |       R2 |
+-------------------+----------+----------+----------+----------+----------+
| Hồi quy Tuyến tính |  0.001234 |  0.035128 |  0.028456 |   2.3456 |  0.8765 |
| RNN               |  0.000987 |  0.031416 |  0.025123 |   2.1234 |  0.9012 |
| LSTM              |  0.000876 |  0.029597 |  0.023456 |   1.9876 |  0.9123 |
+-------------------+----------+----------+----------+----------+----------+
```

## 🔮 Hướng phát triển

### Nguồn dữ liệu bổ sung

- Phân tích tâm lý mạng xã hội (Twitter/X)
- Phân tích tin tức sử dụng NLP/LLM
- Chỉ số on-chain (tỷ lệ băm, địa chỉ hoạt động)
- Chỉ số kinh tế vĩ mô (DXY, Vàng, S&P 500)
- Chỉ số Sợ hãi và Tham lam (Fear and Greed Index)

### Mô hình nâng cao

- BiLSTM (LSTM hai chiều)
- GRU (Đơn vị hồi quy có cổng - Gated Recurrent Unit)
- Cơ chế Transformer/Attention
- Mô hình lai CNN-LSTM
- Phương pháp ensemble (Random Forest, XGBoost)
- Autoencoder để phát hiện bất thường

### Ứng dụng giao dịch

- Tạo tín hiệu giao dịch
- Hệ thống quản lý rủi ro
- Tối ưu hóa danh mục đầu tư
- Khung kiểm tra ngược (backtesting)
- Hệ thống dự đoán thời gian thực

### Mở rộng nghiên cứu

- Tinh chỉnh siêu tham số (Grid Search, Tối ưu hóa Bayesian)
- Cross-validation cho chuỗi thời gian (TimeSeriesSplit)
- Phân tích tầm quan trọng đặc trưng (giá trị SHAP)
- Kỹ thuật AI có thể giải thích
- Học chuyển từ các tiền điện tử khác
- Dự đoán nhiều bước phía trước

## 📚 Tài liệu tham khảo

1. **Yahoo Finance API** - [Tài liệu yfinance](https://github.com/ranaroussi/yfinance)
2. **TensorFlow/Keras** - [Tài liệu TensorFlow](https://www.tensorflow.org/)
3. **scikit-learn** - [Tài liệu scikit-learn](https://scikit-learn.org/)
4. **Phân tích kỹ thuật** - Murphy, J. J. (1999). Technical Analysis of the Financial Markets
5. **Deep Learning** - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
6. **Phân tích chuỗi thời gian** - Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice

## 📝 Sử dụng học thuật

Dự án này được thiết kế cho mục đích học thuật và có thể được sử dụng cho:

- Dự án khóa học đại học
- Luận văn cử nhân/thạc sĩ
- Bài báo nghiên cứu
- Dự án danh mục đầu tư cho đơn xin việc
- Học machine learning và deep learning

### Cấu trúc báo cáo

Xem `report/report_outline.md` để biết cấu trúc báo cáo học thuật hoàn chỉnh bao gồm:

- Giới thiệu
- Phát biểu bài toán
- Công việc liên quan
- Mô tả tập dữ liệu
- Phương pháp luận
- Thí nghiệm
- Kết quả
- Thảo luận
- Kết luận
- Hướng phát triển

## ⚠️ Tuyên bố miễn trừ trách nhiệm

**Dự án này chỉ dành cho mục đích giáo dục và nghiên cứu.**

- **Không phải lời khuyên tài chính**: Các dự đoán không nên được sử dụng cho quyết định giao dịch thực tế
- **Rủi ro thị trường**: Thị trường tiền điện tử rất biến động và khó dự đoán
- **Không đảm bảo**: Mô hình không thể đảm bảo dự đoán chính xác
- **Sử dụng với rủi ro của riêng bạn**: Bất kỳ sử dụng code này cho giao dịch đều là rủi ro của bạn

## 🤝 Đóng góp

Đóng góp được chào đón! Các lĩnh vực cần cải thiện:

- Chỉ số kỹ thuật bổ sung
- Mô hình nâng cao hơn
- Kỹ thuật đặc trưng tốt hơn
- Trực quan hóa được cải thiện
- Nâng cấp tài liệu
- Sửa lỗi

## 🔄 Nhật ký cập nhật

- **Tháng 5/2026**:
  - Khắc phục lỗi tương thích do thay đổi API của thư viện `yfinance` (MultiIndex columns).
  - Khắc phục lỗi `UnicodeEncodeError` khi in các ký tự tiếng Việt trên Windows Console.
  - Xóa bỏ các file cấu hình dependencies dư thừa (`requirements_basic.txt`).

## 📄 Giấy phép

Dự án này là mã nguồn mở và có sẵn cho mục đích giáo dục.

## 👥 Tác giả

- Dự án Dự đoán Giá Bitcoin - 2026

## 🙏 Lời cảm ơn

- Yahoo Finance vì cung cấp dữ liệu tài chính miễn phí
- Đội ngũ TensorFlow/Keras vì framework deep learning
- Cộng đồng mã nguồn mở vì các thư viện khác nhau

---

**Lưu ý**: Dự án này đóng vai trò là nền tảng cho học tập và nghiên cứu. Thị trường tài chính phức tạp và chịu ảnh hưởng bởi nhiều yếu tố không được nắm bắt trong mô hình này. Luôn tự nghiên cứu trước khi đưa ra bất kỳ quyết định đầu tư nào.
