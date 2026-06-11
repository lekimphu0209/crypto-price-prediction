# Đề Cương Báo Cáo Học Thuật
## Dự Đoán Giá Bitcoin Sử Dụng Hồi Quy và Mạng Nơ-ron Hồi Quy

---

## Mục Lục

1. [Trang bìa](#1-trang-bìa)
2. [Tóm tắt](#2-tóm-tắt)
3. [Giới thiệu](#3-giới-thiệu)
4. [Phát biểu bài toán](#4-phát-biểu-bài-toán)
5. [Công việc liên quan](#5-công-việc-liên-quan)
6. [Mô tả tập dữ liệu](#6-mô-tả-tập-dữ-liệu)
7. [Phương pháp luận](#7-phương-pháp-luận)
8. [Thí nghiệm](#8-thí-nghiệm)
9. [Kết quả](#9-kết-quả)
10. [Thảo luận](#10-thảo-luận)
11. [Kết luận](#11-kết-luận)
12. [Hướng phát triển](#12-hướng-phát-triển)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)
14. [Phụ lục](#14-phụ-lục)

---

## 1. Trang Bìa

**Tiêu đề:**  
Dự đoán giá Bitcoin bằng mô hình Hồi quy và Recurrent Neural Network

**Tiêu đề tiếng Anh:**  
Bitcoin Price Prediction using Regression and Recurrent Neural Networks

**Tác giả:**  
[Tên của bạn]

**Cơ sở:**  
[Tên trường đại học]

**Khoa:**  
[Tên khoa]

**Ngày tháng:**  
[Tháng, Năm]

**Giáo viên hướng dẫn:**  
[Tên giáo viên hướng dẫn]

---

## 2. Tóm Tắt

### Tóm tắt tiếng Việt
*(Khoảng 200-300 từ)*

Tóm tắt:
- Giới thiệu bài toán dự đoán giá Bitcoin
- Phương pháp tiếp cận (Hồi quy, RNN, LSTM)
- Kết quả chính
- Đóng góp của nghiên cứu
- Hướng phát triển

### Tóm tắt tiếng Anh
*(Khoảng 200-300 từ)*

Summary:
- Introduction to Bitcoin price prediction problem
- Methodology (Regression, RNN, LSTM)
- Key results
- Research contributions
- Future directions

**Từ khóa:**  
Bitcoin, Dự đoán giá, Machine Learning, Deep Learning, RNN, LSTM, Chuỗi thời gian, Phân tích kỹ thuật

---

## 3. Giới Thiệu

### 3.1 Bối cảnh

#### 3.1.1 Bitcoin và thị trường tiền điện tử
- Lịch sử ngắn gọn của Bitcoin (2009 - hiện tại)
- Tăng trưởng vốn hóa thị trường
- Đặc tính biến động (volatility)
- Tầm quan trọng trong thị trường tài chính

#### 3.1.2 Dự đoán giá trong thị trường tài chính
- Tại sao dự đoán giá lại quan trọng
- Phương pháp truyền thống so với Machine Learning
- Thách thức trong dự đoán chuỗi thời gian tài chính

#### 3.1.3 Machine Learning trong tài chính
- Sự phát triển của AI trong tài chính
- Ứng dụng: giao dịch, quản lý rủi ro, tối ưu hóa danh mục đầu tư
- Lợi thế của deep learning cho chuỗi thời gian

### 3.2 Động cơ

#### 3.2.1 Động cơ nghiên cứu
- Biến động cao của thị trường tiền điện tử
- Tiềm năng lợi nhuận (với quản lý rủi ro phù hợp)
- Sự quan tâm học thuật trong việc áp dụng AI vào tài chính
- Khoảng trống trong nghiên cứu dự đoán giá Bitcoin

#### 3.2.2 Ứng dụng thực tế
- Tín hiệu giao dịch
- Đánh giá rủi ro
- Quản lý danh mục đầu tư
- Hỗ trợ quyết định đầu tư

### 3.3 Mục tiêu nghiên cứu

**Mục tiêu chính:**
1. Xây dựng quy trình hoàn chỉnh cho dự đoán giá Bitcoin
2. So sánh mô hình Hồi quy Tuyến tính, RNN và LSTM
3. Đánh giá hiệu suất mô hình sử dụng nhiều chỉ số
4. Cung cấp thông tin chi tiết về tầm quan trọng của đặc trưng

**Mục tiêu phụ:**
1. Tạo khung có thể tái sử dụng cho dự đoán giá crypto
2. Tài liệu hóa các phương pháp tốt nhất cho chuỗi thời gian ML
3. Cung cấp tài liệu giáo dục cho sinh viên

### 3.4 Câu hỏi nghiên cứu

1. Mô hình machine learning có thể dự đoán giá Bitcoin chính xác như thế nào?
2. Mô hình nào hoạt động tốt nhất: Hồi quy Tuyến tính, RNN hay LSTM?
3. Chỉ số kỹ thuật nào có khả năng dự đoán tốt nhất?
4. Những hạn chế của các phương pháp hiện tại là gì?

### 3.5 Đóng góp

1. Triển khai hoàn chỉnh, tài liệu hóa đầy đủ
2. So sánh toàn diện ba mô hình
3. Phân tích tầm quan trọng đặc trưng chi tiết
4. Tài liệu giáo dục cho sinh viên
5. Xác định các hạn chế và hướng phát triển

---

## 4. Phát Biểu Bài Toán

### 4.1 Định nghĩa bài toán

**Định nghĩa chính thức:**

Cho chuỗi thời gian dữ liệu giá Bitcoin:
```
D = {P₁, P₂, ..., Pₙ}
```
trong đó Pₜ là giá tại thời điểm t

**Mục tiêu:** Dự đoán giá tại thời điểm t+1:
```
P̂_{t+1} = f(P_{t}, P_{t-1}, ..., P_{t-k})
```
trong đó k là cửa sổ hồi quy (ví dụ: 30 ngày)

### 4.2 Đặc điểm bài toán

#### 4.2.1 Thuộc tính chuỗi thời gian
- **Phụ thuộc thời gian:** Giá hiện tại phụ thuộc vào các giá quá khứ
- **Không dừng (Non-stationarity):** Các thuộc tính thống kê thay đổi theo thời gian
- **Tính mùa vụ (Seasonality):** Các mẫu định kỳ có thể có
- **Xu hướng:** Sự chuyển động dài hạn tăng/giảm

#### 4.2.2 Thách thức
- **Biến động cao:** Thay đổi giá nhanh chóng
- **Nhiễu (Noise):** Biến động ngẫu nhiên
- **Yếu tố bên ngoài:** Tin tức, quy định, tâm lý thị trường
- **Phi tuyến tính:** Các mối quan hệ phức tạp, phi tuyến tính

### 4.3 Phạm vi và hạn chế

#### 4.3.1 Phạm vi
- Dữ liệu theo ngày (daily timeframe)
- 5 năm dữ liệu lịch sử
- Chỉ số kỹ thuật
- Dự đoán một bước (next day)

#### 4.3.2 Hạn chế
- Không bao gồm các yếu tố bên ngoài (tin tức, tâm lý)
- Không phù hợp cho giao dịch thời gian thực
- Chỉ giới hạn Bitcoin (không phải crypto khác)
- Không tính đến chi phí giao dịch

---

## 5. Công Việc Liên Quan

### 5.1 Phương pháp truyền thống

#### 5.1.1 Phân tích kỹ thuật
- Đường trung bình (Moving Averages)
- Chỉ số sức mạnh tương đối (RSI)
- MACD
- Mức hỗ trợ và kháng cự
- **Tài liệu tham khảo:** Murphy (1999), Pring (2002)

#### 5.1.2 Phương pháp thống kê
- ARIMA (AutoRegressive Integrated Moving Average)
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- Bộ lọc Kalman
- **Tài liệu tham khảo:** Box & Jenkins (1970), Hamilton (1994)

### 5.2 Phương pháp Machine Learning

#### 5.2.1 ML truyền thống
- **Support Vector Regression (SVR)**
  - Tài liệu tham khảo: Smola & Schölkopf (2004)
  
- **Random Forest**
  - Tài liệu tham khảo: Breiman (2001)
  
- **Gradient Boosting (XGBoost, LightGBM)**
  - Tài liệu tham khảo: Chen & Guestrin (2016)

#### 5.2.2 Mạng nơ-ron
- **Mạng nơ-ron feedforward**
  - Tài liệu tham khảo: Hornik et al. (1989)
  
- **Recurrent Neural Network (RNN)**
  - Tài liệu tham khảo: Rumelhart et al. (1986)
  
- **Long Short-Term Memory (LSTM)**
  - Tài liệu tham khảo: Hochreiter & Schmidhuber (1997)
  
- **Gated Recurrent Unit (GRU)**
  - Tài liệu tham khảo: Cho et al. (2014)

### 5.3 Văn học dự đoán tiền điện tử

#### 5.3.1 Khảo sát các nghiên cứu gần đây
- Nghiên cứu 1: LSTM cho dự đoán Bitcoin (McNally et al., 2018)
- Nghiên cứu 2: Phương pháp ensemble cho crypto (Araque et al., 2021)
- Nghiên cứu 3: Phân tích tâm lý + dự đoán giá (Kraaijeveld & De Smedt, 2020)

#### 5.3.2 So sánh với phương pháp của chúng tôi
- Tương đồng: Sử dụng chỉ số kỹ thuật, mô hình LSTM
- Khác biệt: So sánh mô hình toàn diện, tài liệu hóa quy trình chi tiết

### 5.4 Khoảng trống nghiên cứu

- Thiếu so sánh toàn diện giữa mô hình đơn giản và phức tạp
- Tài liệu hóa quy trình kỹ thuật đặc trưng không đầy đủ
- Phân tích hạn chế mô hình không đủ
- Cần triển khai thân thiện với người mới, giáo dục

---

## 6. Mô Tả Tập Dữ Liệu

### 6.1 Nguồn dữ liệu

- **Nguồn:** Yahoo Finance thông qua API yfinance
- **Mã chứng khoán:** BTC-USD
- **Khung thời gian:** Hàng ngày (1D)
- **Giai đoạn:** 5 năm (khoảng 2019-2024)
- **Điểm dữ liệu:** ~1250 ngày giao dịch

### 6.2 Cấu trúc dữ liệu

#### 6.2.1 Dữ liệu OHLCV

| Cột | Mô tả | Kiểu |
|--------|-------------|------|
| Date | Ngày giao dịch | datetime |
| Open | Giá mở cửa | float |
| High | Giá cao nhất | float |
| Low | Giá thấp nhất | float |
| Close | Giá đóng cửa | float |
| Volume | Khối lượng giao dịch | float |

#### 6.2.2 Dữ liệu mẫu

```
Date        Open        High        Low         Close       Volume
2019-01-01  3750.25     3800.50     3720.10     3785.30     5.2B
2019-01-02  3785.30     3850.00     3770.25     3820.15     4.8B
...
```

### 6.3 Thống kê dữ liệu

#### 6.3.1 Thống kê mô tả

| Thống kê | Open | High | Low | Close | Volume |
|-----------|------|------|-----|-------|--------|
| Mean | ... | ... | ... | ... | ... |
| Std | ... | ... | ... | ... | ... |
| Min | ... | ... | ... | ... | ... |
| Max | ... | ... | ... | ... | ... |
| Median | ... | ... | ... | ... | ... |

#### 6.3.2 Chất lượng dữ liệu
- Giá trị thiếu: 0 (sau tiền xử lý)
- Hàng trùng lặp: 0
- Giá trị ngoại lai: Xử lý bằng phương pháp robust

### 6.4 Kỹ thuật đặc trưng

#### 6.4.1 Chỉ số kỹ thuật

**Đường trung bình (MA20, MA50)**
```
MAₙ = (1/n) × Σᵢ₌₁ⁿ Pᵢ
```
- Mục đích: Xác định xu hướng
- Sử dụng: Tín hiệu crossover

**Chỉ số sức mạnh tương đối (RSI)**
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```
- Mục đích: Đo động lượng và điều kiện mua quá/bán quá
- Phạm vi: 0-100
- Giải thích: >70 mua quá, <30 bán quá

**MACD (Hội tụ phân kỳ đường trung bình)**
```
MACD = EMA₁₂ - EMA₂₆
Signal = EMA₉(MACD)
Histogram = MACD - Signal
```
- Mục đích: Hiển thị sức mạnh và hướng của xu hướng
- Thành phần: Đường MACD, Đường tín hiệu, Biểu đồ

**Phạm vi thực tế trung bình (ATR)**
```
TR = max(H-L, |H-Cₚᵣₑᵥ|, |L-Cₚᵣₑᵥ|)
ATR = MA(TR)
```
- Mục đích: Đo độ biến động
- ATR cao hơn = độ biến động cao hơn

**Thay đổi giá**
```
Changeₙ = ((Pₜ - Pₜ₋ₙ) / Pₜ₋ₙ) × 100
```
- Mục đích: Nắm bắt động lượng
- Giai đoạn: 1 ngày, 5 ngày, 10 ngày

**Đặc trưng khối lượng**
```
Volume Change = ((Vₜ - Vₜ₋₁) / Vₜ₋₁) × 100
Volume MA = MA(Volume)
Volume Ratio = Volume / Volume MA
```
- Mục đích: Đo sức mạnh giao dịch

#### 6.4.2 Tương quan đặc trưng

Bao gồm bản đồ nhiệt tương quan hiển thị mối quan hệ giữa các đặc trưng.

### 6.5 Chia dữ liệu

- **Tập huấn luyện:** 80% dữ liệu
- **Tập kiểm tra:** 20% dữ liệu
- **Tập xác thực:** 20% tập huấn luyện (cho mạng nơ-ron)
- **Phương pháp chia:** Tuần tự (dựa trên thời gian, không ngẫu nhiên)

---

## 7. Phương Pháp Luận

### 7.1 Quy trình tổng thể

**Hình: Kiến trúc hệ thống**
```
Thu thập dữ liệu → Kỹ thuật đặc trưng → Tiền xử lý → Huấn luyện mô hình → Đánh giá → Trực quan hóa
```

### 7.2 Tiền xử lý dữ liệu

#### 7.2.1 Xử lý giá trị thiếu
- Phương pháp: Forward fill (điền tiếp)
- Fallback: Backward fill (điền ngược)
- Lý do: Tính liên tục chuỗi thời gian

#### 7.2.2 Chuẩn hóa

**Công thức MinMaxScaler:**
```
x' = (x - xₘᵢₙ) / (xₘₐₓ - xₘᵢₙ)
```
- Phạm vi: [0, 1]
- Mục đích: Chuẩn hóa đặc trưng cùng phạm vi
- Tầm quan trọng: Mạng nơ-ron hội tụ nhanh hơn

#### 7.2.3 Tạo chuỗi

**Phương pháp cửa sổ trượt:**
```
X = [Pₜ₋ₙ, Pₜ₋ₙ₊₁, ..., Pₜ₋₁]  (Chuỗi đầu vào)
y = Pₜ                          (Mục tiêu)
```
- Độ dài chuỗi: 30 ngày
- Bước trượt: 1 ngày
- Đầu ra shape: (samples, 30, features)

### 7.3 Hồi quy Tuyến tính

#### 7.3.1 Công thức hóa

**Phương trình mô hình:**
```
y = w·x + b
```

**Dạng ma trận:**
```
y = Xw + b
```
trong đó:
- y: giá dự đoán (số vô hướng)
- X: vector đặc trưng (n×1)
- w: vector trọng số (n×1)
- b: độ lệch (số vô hướng)

#### 7.3.2 Mục tiêu huấn luyện

**Hàm mất mát (MSE):**
```
L(w, b) = (1/n) × Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Tối ưu hóa:**
- Phương pháp: Ordinary Least Squares (OLS)
- Giải pháp: w = (XᵀX)⁻¹Xᵀy
- Không cần tối ưu hóa lặp (giải pháp dạng đóng)

#### 7.3.3 Ưu điểm và hạn chế

**Ưu điểm:**
- Đơn giản và có thể giải thích
- Huấn luyện nhanh
- Tầm quan trọng đặc trưng từ hệ số

**Hạn chế:**
- Không thể nắm bắt mối quan hệ phi tuyến tính
- Không có bộ nhớ thời gian
- Khả năng dự đoán hạn chế

### 7.4 Recurrent Neural Network (RNN)

#### 7.4.1 Công thức hóa

**Cập nhật trạng thái ẩn:**
```
hₜ = f(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)
```

**Đầu ra:**
```
yₜ = Wᵧ·hₜ + c
```

trong đó:
- hₜ: trạng thái ẩn tại thời điểm t
- xₜ: đầu vào tại thời điểm t
- Wₓ: trọng số đầu vào-ẩn
- Wₕ: trọng số ẩn-ẩn
- b: độ lệch
- f: hàm kích hoạt (tanh)
- Wᵧ: trọng số ẩn-đầu ra
- c: độ lệch đầu ra

#### 7.4.2 Kiến trúc

```
Input (30, 6) → SimpleRNN(50) → Dropout(0.2) → Dense(1) → Output
```

**Tham số:**
- Độ dài chuỗi: 30
- Đặc trưng: 6
- Đơn vị RNN: 50
- Tỷ lệ dropout: 0.2
- Kích hoạt: tanh (ẩn), tuyến tính (đầu ra)

#### 7.4.3 Huấn luyện

**Hàm mất mát:**
```
L = (1/n) × Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Optimizer:** Adam
- Tốc độ học: 0.001
- Tốc độ học thích ứng cho từng tham số

**Cấu hình huấn luyện:**
- Epochs: 50 (với early stopping)
- Batch size: 32
- Early stopping patience: 10
- Validation split: 20%

#### 7.4.4 Ưu điểm và hạn chế

**Ưu điểm:**
- Nắm bắt phụ thuộc thời gian
- Xử lý chuỗi độ dài bất kỳ
- Linh hoạt hơn mô hình tuyến tính

**Hạn chế:**
- Vấn đề gradient biến mất
- Khó học phụ thuộc dài hạn
- Không ổn định cho chuỗi dài

### 7.5 Long Short-Term Memory (LSTM)

#### 7.5.1 Công thức hóa

**Cổng quên:**
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
```

**Cổng đầu vào:**
```
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
```

**Trạng thái ô nhớ ứng viên:**
```
C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)
```

**Cập nhật trạng thái ô nhớ:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

**Cổng đầu ra:**
```
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
```

**Trạng thái ẩn:**
```
hₜ = oₜ ⊙ tanh(Cₜ)
```

trong đó:
- σ: kích hoạt sigmoid
- ⊙: nhân từng phần tử
- Cₜ: trạng thái ô nhớ tại thời điểm t
- hₜ: trạng thái ẩn tại thời điểm t
- Wf, Wi, WC, Wo: ma trận trọng số
- bf, bi, bC, bo: vector độ lệch

#### 7.5.2 Kiến trúc

```
Input (30, 6) → LSTM(50) → Dropout(0.2) → Dense(1) → Output
```

**Tham số:**
- Độ dài chuỗi: 30
- Đặc trưng: 6
- Đơn vị LSTM: 50
- Tỷ lệ dropout: 0.2
- Kích hoạt: tanh (trạng thái ô nhớ), sigmoid (cổng), tuyến tính (đầu ra)

#### 7.5.3 Huấn luyện

Tương tự RNN:
- Loss: MSE
- Optimizer: Adam
- Tốc độ học: 0.001
- Epochs: 50 (với early stopping)
- Batch size: 32

#### 7.5.4 Ưu điểm và hạn chế

**Ưu điểm:**
- Giải quyết vấn đề gradient biến mất
- Học phụ thuộc dài hạn
- Huấn luyện ổn định hơn
- Tốt hơn cho mẫu thời gian phức tạp

**Hạn chế:**
- Tốn kém hơn về mặt tính toán
- Nhiều tham số để huấn luyện
- Thời gian huấn luyện dài hơn
- Kiến trúc phức tạp hơn

### 7.6 Khung so sánh mô hình

#### 7.6.1 Chỉ số đánh giá

**MSE (Mean Squared Error - Sai số bình phương trung bình):**
```
MSE = (1/n) × Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**RMSE (Root Mean Squared Error - Căn bậc hai sai số bình phương trung bình):**
```
RMSE = √MSE
```

**MAE (Mean Absolute Error - Sai số tuyệt đối trung bình):**
```
MAE = (1/n) × Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

**MAPE (Mean Absolute Percentage Error - Sai số phần trăm tuyệt đối trung bình):**
```
MAPE = (1/n) × Σᵢ₌₁ⁿ |(yᵢ - ŷᵢ) / yᵢ| × 100
```

**R-squared (Hệ số xác định - R²):**
```
R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
```

#### 7.6.2 Kiểm tra thống kê

- Kiểm tra t cặp để so sánh mô hình
- Kiểm tra Diebold-Mariano cho độ chính xác dự đoán
- Cross-validation cho tính mạnh mẽ

---

## 8. Thí Nghiệm

### 8.1 Thiết lập thí nghiệm

#### 8.1.1 Phần cứng
- CPU: [CPU của bạn]
- GPU: [GPU của bạn, nếu có]
- RAM: [RAM của bạn]

#### 8.1.2 Phần mềm
- Python: 3.8+
- TensorFlow: 2.13+
- scikit-learn: 1.3+
- Các thư viện khác: Xem requirements.txt

#### 8.1.3 Siêu tham số

**Hồi quy Tuyến tính:**
- Không có siêu tham số để tinh chỉnh

**RNN:**
- Độ dài chuỗi: 30
- Đơn vị RNN: 50
- Dropout: 0.2
- Tốc độ học: 0.001
- Batch size: 32
- Epochs: 50

**LSTM:**
- Độ dài chuỗi: 30
- Đơn vị LSTM: 50
- Dropout: 0.2
- Tốc độ học: 0.001
- Batch size: 32
- Epochs: 50

### 8.2 Quy trình huấn luyện

#### 8.2.1 Chuẩn bị dữ liệu
1. Tải dữ liệu từ CSV
2. Xử lý giá trị thiếu
3. Chuẩn hóa sử dụng MinMaxScaler
4. Tạo chuỗi (cửa sổ 30 ngày)
5. Chia thành train/validation/test

#### 8.2.2 Huấn luyện mô hình
1. Khởi tạo kiến trúc mô hình
2. Biên dịch với loss và optimizer
3. Huấn luyện với early stopping
4. Lưu mô hình tốt nhất
5. Ghi lại lịch sử huấn luyện

#### 8.2.3 Đánh giá
1. Tải dữ liệu kiểm tra
2. Tạo dự đoán
3. Tính toán chỉ số
4. Tạo trực quan hóa
5. So sánh kết quả

### 8.3 Nghiên cứu Ablation

#### 8.3.1 Ablation đặc trưng
- Kiểm tra với các tập đặc trưng con khác nhau
- Xác định đặc trưng quan trọng nhất
- So sánh hiệu suất mô hình

#### 8.3.2 Ablation độ dài chuỗi
- Kiểm tra các độ dài chuỗi khác nhau (7, 14, 30, 60 ngày)
- Phân tích tác động đến hiệu suất
- Tìm kích thước cửa sổ tối ưu

#### 8.3.3 Ablation kiến trúc mô hình
- Kiểm tra số lượng đơn vị khác nhau (25, 50, 100)
- Kiểm tra tỷ lệ dropout khác nhau (0.1, 0.2, 0.3)
- So sánh RNN/LSTM một lớp vs đa lớp

---

## 9. Kết Quả

### 9.1 Kết quả định lượng

#### 9.1.1 So sánh hiệu suất mô hình

**Bảng: Kết quả so sánh mô hình**

| Mô hình | MSE | RMSE | MAE | MAPE (%) | R² | Thời gian huấn luyện |
|-------|-----|------|-----|----------|-----|-------------------|
| Hồi quy Tuyến tính | ... | ... | ... | ... | ... | ... |
| RNN | ... | ... | ... | ... | ... | ... |
| LSTM | ... | ... | ... | ... | ... | ... |

*(Điền kết quả thực tế sau khi chạy thí nghiệm)*

#### 9.1.2 Ý nghĩa thống kê

**Bảng: So sánh mô hình cặp**

| Cặp mô hình | p-value | Có ý nghĩa? |
|------------|---------|-------------|
| LR vs RNN | ... | ... |
| LR vs LSTM | ... | ... |
| RNN vs LSTM | ... | ... |

### 9.2 Kết quả định tính

#### 9.2.1 Biểu đồ dự đoán

**Hình để bao gồm:**
- Thực tế vs Dự đoán (Hồi quy Tuyến tính)
- Thực tế vs Dự đoán (RNN)
- Thực tế vs Dự đoán (LSTM)
- Phóng to các giai đoạn cụ thể

#### 9.2.2 Đường cong mất mát

**Hình để bao gồm:**
- Mất mát huấn luyện vs Mất mát xác thực (RNN)
- Mất mát huấn luyện vs Mất mát xác thực (LSTM)
- MAE huấn luyện vs MAE xác thực

#### 9.2.3 Phân tích phần dư

**Hình để bao gồm:**
- Biểu đồ phần dư cho từng mô hình
- Biểu đồ phân phối phần dư
- Biểu đồ Q-Q để kiểm tra tính chuẩn

### 9.3 Tầm quan trọng đặc trưng

#### 9.3.1 Hệ số Hồi quy Tuyến tính

**Bảng: Tầm quan trọng đặc trưng (Hồi quy Tuyến tính)**

| Đặc trưng | Hệ số | Giá trị tuyệt đối | Xếp hạng |
|---------|-------------|----------------|------|
| Close | ... | ... | ... |
| MA20 | ... | ... | ... |
| MA50 | ... | ... | ... |
| RSI | ... | ... | ... |
| MACD | ... | ... | ... |
| ATR | ... | ... | ... |

#### 9.3.2 Kết quả Ablation đặc trưng

**Bảng: Hiệu suất với các tập đặc trưng khác nhau**

| Tập đặc trưng | MSE | RMSE | MAE |
|-------------|-----|------|-----|
| Tất cả đặc trưng | ... | ... | ... |
| Không RSI | ... | ... | ... |
| Không MACD | ... | ... | ... |
| Chỉ OHLCV | ... | ... | ... |

---

## 10. Thảo Luận

### 10.1 Giải thích kết quả

#### 10.1.1 Hiệu suất mô hình
- Mô hình nào hoạt động tốt nhất và tại sao
- Sự khác biệt hiệu suất giữa các mô hình
- So sánh với kỳ vọng cơ bản
- So sánh với công việc liên quan

#### 10.1.2 Tầm quan trọng đặc trưng
- Chỉ số kỹ thuật dự đoán tốt nhất
- Tại sao một số đặc trưng quan trọng hơn
- Phát hiện đáng ngạc nhiên
- Hàm ý cho lựa chọn đặc trưng

### 10.2 Điểm mạnh của phương pháp

#### 10.2.1 Điểm mạnh phương pháp luận
- So sánh mô hình toàn diện
- Khung đánh giá chặt chẽ
- Chia train/test phù hợp
- Nhiều chỉ số đánh giá

#### 10.2.2 Điểm mạnh triển khai
- Code mô-đun và có thể tái sử dụng
- Tài liệu hóa tốt
- Thân thiện với người mới bắt đầu
- Khung có thể mở rộng

### 10.3 Hạn chế

#### 10.3.1 Hạn chế dữ liệu
- Giới hạn chỉ số kỹ thuật
- Không bao gồm yếu tố bên ngoài
- Chỉ một tiền điện tử (Bitcoin)
- Khung thời gian hàng ngày (không có dữ liệu intraday)

#### 10.3.2 Hạn chế mô hình
- Chỉ dự đoán một bước
- Không có lượng hóa độ không chắc chắn
- Không tính đến thay đổi chế độ
- Khả năng overfitting tiềm ẩn

#### 10.3.3 Hạn chế đánh giá
- Không có xác thực walk-forward
- Giai đoạn kiểm tra hạn chế
- Không xem xét chi phí giao dịch
- Không có chỉ số điều chỉnh rủi ro

### 10.4 Hàm ý thực tế

#### 10.4.1 Ứng dụng giao dịch
- Cách sử dụng dự đoán (với thận trọng)
- Xem xét rủi ro
- Cần giám sát của con người
- Không thay thế cho phân tích cơ bản

#### 10.4.2 Hàm ý nghiên cứu
- Hướng nghiên cứu tương lai
- Cải thiện tiềm năng
- Khả năng chuyển sang tài sản khác
- Tích hợp với các phương pháp khác

---

## 11. Kết Luận

### 11.1 Tóm tắt công việc

- Xây dựng quy trình hoàn chỉnh từ đầu đến cuối cho dự đoán giá Bitcoin
- Triển khai và so sánh mô hình Hồi quy Tuyến tính, RNN và LSTM
- Kỹ thuật đặc trưng chỉ số kỹ thuật
- Đánh giá mô hình sử dụng nhiều chỉ số
- Tạo trực quan hóa toàn diện

### 11.2 Phát hiện chính

1. **Hiệu suất mô hình:** [Tóm tắt mô hình nào hoạt động tốt nhất]
2. **Tầm quan trọng đặc trưng:** [Tóm tắt đặc trưng quan trọng nhất]
3. **Tính khả thi thực tế:** [Đánh giá tiện ích thực tế]
4. **Hạn chế:** [Các hạn chế chính được xác định]

### 11.3 Đóng góp

1. Triển khai hoàn chỉnh, tài liệu hóa đầy đủ
2. So sánh mô hình toàn diện
3. Phân tích tầm quan trọng đặc trưng
4. Tài liệu giáo dục cho sinh viên
5. Nền tảng cho nghiên cứu tương lai

### 11.4 Lời nhận xét cuối cùng

- Machine learning có thể cung cấp thông tin về chuyển động giá
- Không mô hình nào có thể dự đoán hoàn hảo thị trường tài chính
- Kết quả nên được sử dụng thận trọng
- Công việc tương lai nên giải quyết các hạn chế đã xác định
- Dự án này đóng vai trò là nền tảng cho nghiên cứu tiếp theo

---

## 12. Hướng Phát Triển

### 12.1 Cải tiến dữ liệu

#### 12.1.1 Nguồn dữ liệu bổ sung
- Phân tích tâm lý mạng xã hội (Twitter/X, Reddit)
- Tiêu đề tin tức và phân tích sử dụng NLP/LLM
- Chỉ số on-chain (hash rate, địa chỉ hoạt động)
- Chỉ số kinh tế vĩ mô (DXY, Vàng, S&P 500)
- Chỉ số Sợ hãi và Tham lam

#### 12.1.2 Nhiều tiền điện tử
- Mở rộng sang Ethereum, Litecoin, v.v.
- Tương quan chéo tiền tệ
- Dự đoán cấp danh mục

### 12.2 Cải tiến mô hình

#### 12.2.1 Kiến trúc nâng cao
- LSTM hai chiều (BiLSTM)
- Đơn vị hồi quy có cổng (GRU)
- Cơ chế Transformer/Attention
- Mô hình lai CNN-LSTM
- Phương pháp ensemble (Random Forest, XGBoost)
- Autoencoder để phát hiện bất thường

#### 12.2.2 Phương pháp Ensemble
- Ensemble mô hình (bagging, boosting)
- Stacking nhiều mô hình
- Bộ phân loại voting
- Trung bình mô hình Bayesian

#### 12.2.3 Tối ưu hóa siêu tham số
- Tìm kiếm lưới (Grid search)
- Tìm kiếm ngẫu nhiên (Random search)
- Tối ưu hóa Bayesian
- Tìm kiếm kiến trúc nơ-ron (NAS)

### 12.3 Cải tiến đánh giá

#### 12.3.1 Xác thực nâng cao
- Xác thực walk-forward
- Cross-validation chuỗi thời gian
- Kiểm tra out-of-sample
- Nhiều giai đoạn thời gian

#### 12.3.2 Chỉ số bổ sung
- Tỷ lệ Sharpe
- Drawdown tối đa
- Hệ số lợi nhuận
- Tỷ lệ thắng
- Lợi nhuận điều chỉnh rủi ro

### 12.4 Ứng dụng thực tế

#### 12.4.1 Hệ thống giao dịch
- Quy trình dự đoán thời gian thực
- Tạo tín hiệu giao dịch
- Module quản lý rủi ro
- Khung kiểm tra ngược (backtesting)
- Giao diện giao dịch trực tiếp

#### 12.4.2 AI có thể giải thích
- Giá trị SHAP cho tầm quan trọng đặc trưng
- Trực quan hóa attention
- Phân tích đường ra quyết định
- Tính có thể giải thích mô hình

### 12.5 Hướng nghiên cứu

#### 12.5.1 Nghiên cứu lý thuyết
- Giới hạn lý thuyết về khả năng dự đoán
- Phân tích lý thuyết thông tin
- Hàm ý hiệu quả thị trường
- Suy luận nhân quả trong chuỗi thời gian tài chính

#### 12.5.2 Nghiên cứu ứng dụng
- Học chuyển giữa các thị trường
- Học ít-shot cho tài sản mới
- Học liên tục cho thay đổi chế độ thị trường
- Học đa phương thức (giá + văn bản + hình ảnh)

---

## 13. Tài Liệu Tham Khảo

### 13.1 Sách

1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

3. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

4. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.

### 13.2 Bài báo tạp chí

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

2. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.

3. McNally, S., Roche, J., & Caton, S. (2018). Predicting the price of Bitcoin using Machine Learning. *2018 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

### 13.3 Báo cáo hội nghị

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

### 13.4 Tài nguyên trực tuyến

1. TensorFlow Documentation. https://www.tensorflow.org/

2. scikit-learn Documentation. https://scikit-learn.org/

3. yfinance Documentation. https://github.com/ranaroussi/yfinance

4. Kaggle Cryptocurrency Historical Data. https://www.kaggle.com/

### 13.5 Báo cáo kỹ thuật

1. Araque, O., Corcuera-Platas, I., Sánchez-Rada, J. F., & Iglesias, C. A. (2021). Enhancing deep learning sentiment analysis with ensemble techniques in social networks. *Expert Systems with Applications*, 164, 113898.

2. Kraaijeveld, O., & De Smedt, J. (2020). The predictive power of public Twitter sentiment for forecasting cryptocurrency prices. *PLOS ONE*, 15(4), e0231605.

---

## 14. Phụ Lục

### Phụ lục A: Cấu trúc Code

#### A.1 Cấu trúc thư mục
*(Cây thư mục chi tiết)*

#### A.2 Mô tả module
*(Mô tả từng module Python)*

#### A.3 Hướng dẫn sử dụng
*(Hướng dẫn từng bước để chạy code)*

### Phụ lục B: Trực quan hóa bổ sung

#### B.1 Lịch sử huấn luyện đầy đủ
*(Đường cong mất mát hoàn chỉnh cho tất cả mô hình)*

#### B.2 Biểu đồ dự đoán mở rộng
*(Giai đoạn dài hơn hiển thị dự đoán)*

#### B.3 Bản đồ nhiệt tương quan
*(Bản đồ nhiệt tương quan hoàn chỉnh)*

### Phụ lục C: Kết quả tinh chỉnh siêu tham số

#### C.1 Kết quả tìm kiếm lưới
*(Bảng tất cả các kết hợp siêu tham số được kiểm tra)*

#### C.2 Siêu tham số tốt nhất
*(Siêu tham số cuối cùng được chọn cho từng mô hình)*

### Phụ lục D: Kiểm tra thống kê

#### D.1 Kiểm tra tính chuẩn
*(Kết quả kiểm tra Shapiro-Wilk cho phần dư)*

#### D.2 Kiểm tra tính dừng
*(Kết quả kiểm tra Augmented Dickey-Fuller)*

#### D.3 Phân tích tự tương quan
*(Biểu đồ ACF và PACF)*

### Phụ lục E: Kết quả mở rộng

#### E.1 Kết quả dự đoán nhiều bước
*(Kết quả dự đoán nhiều ngày phía trước)*

#### E.2 Kết quả khung thời gian khác nhau
*(Kết quả cho dữ liệu theo giờ, theo tuần, theo tháng)*

#### E.3 Kết quả chéo tiền tệ
*(Kết quả cho các tiền điện tử khác)*

### Phụ lục F: Liệt kê Code

#### F.1 Các hàm chính
*(Mã nguồn cho các hàm quan trọng)*

#### F.2 Kiến trúc mô hình
*(Tóm tắt mô hình chi tiết)*

#### F.3 Script huấn luyện
*(Script huấn luyện hoàn chỉnh)*

---

## Kết thúc Đề cương Báo cáo

**Lưu ý:** Đề cương này cung cấp cấu trúc toàn diện cho báo cáo học thuật. Điền kết quả, hình ảnh và bảng thực tế sau khi chạy thí nghiệm. Điều chỉnh độ dài và chiều sâu của từng phần dựa trên yêu cầu cụ thể và giới hạn trang của bạn.
