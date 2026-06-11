# Crypto Price Prediction Platform - Hướng dẫn cài đặt

Platform dự đoán giá tiền điện tử sử dụng Machine Learning và Clean Architecture với SOLID principles.

## 📋 Tổng quan

Dự án xây dựng hệ thống dự đoán giá Bitcoin và Ethereum sử dụng nhiều mô hình Machine Learning khác nhau, kết hợp với phân tích sentiment từ mạng xã hội và tin tức tài chính.

### 🎯 Tính năng chính

- **Đa mô hình dự đoán**: Linear Regression, XGBoost, LSTM, BiLSTM, Transformer, Ensemble
- **Dữ liệu real-time**: Binance (crypto), Yahoo Finance (Gold, DXY)
- **Phân tích sentiment**: Twitter/X.com, NewsAPI
- **Dashboard tương tác**: 9 sections với Streamlit
- **Backtesting**: Chiến lược giao dịch và hiệu suất
- **Clean Architecture**: SOLID principles, Dependency Injection

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- pip
- Git

### Bước 1: Clone repository

```bash
git clone https://github.com/lekimphu0209/crypto-price-prediction.git
cd crypto-price-prediction
```

### Bước 2: Tạo virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình environment variables

Tạo file `.env` trong thư mục gốc:

```env
# Binance API (không bắt buộc cho public data)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# NewsAPI (tùy chọn)
NEWSAPI_KEY=your_newsapi_key

# Twitter/X API (tùy chọn)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret

# OpenAI API (tùy chọn cho LLM sentiment)
OPENAI_API_KEY=your_openai_api_key
```

## 📖 Cách sử dụng

### Chạy training models

```bash
python train_models.py
```

Script này sẽ:
- Thu thập dữ liệu từ Binance
- Tạo features kỹ thuật
- Train tất cả models
- So sánh hiệu suất
- Lưu models đã train

### Chạy dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard sẽ mở tại http://localhost:8501 với các sections:

1. **🏠 Overview**: Giá real-time BTC/ETH, biểu đồ giá
2. **📈 Price Prediction**: Dự đoán giá với model selection
3. **🤖 Model Comparison**: So sánh hiệu suất models
4. **😊 Sentiment Analysis**: Phân tích sentiment từ Twitter/X
5. **📰 News Intelligence**: Tin tức và impact analysis
6. **🌎 External Factors**: Gold, DXY và correlation
7. **📊 Feature Importance**: Tầm quan trọng các features
8. **💰 Backtesting**: Kết quả backtesting chiến lược
9. **⚙ System Status**: Monitoring hệ thống

### Chạy prediction

```bash
python main.py
```

## 🏗️ Cấu trúc dự án

```
crypto-price-prediction/
├── src/
│   ├── core/                    # Core logic & config
│   │   ├── config.py           # Configuration management
│   │   └── container.py        # Dependency injection
│   │
│   ├── domain/                  # Domain layer
│   │   ├── entities/           # Business entities
│   │   ├── interfaces/         # Interfaces (contracts)
│   │   └── value_objects/      # Value objects
│   │
│   ├── application/            # Application layer
│   │   ├── dtos/               # Data Transfer Objects
│   │   ├── pipelines/          # Business logic pipelines
│   │   └── use_cases/         # Use cases
│   │
│   ├── infrastructure/         # Infrastructure layer
│   │   ├── data_providers/     # External data sources
│   │   ├── models/             # ML models
│   │   ├── ensemble/           # Ensemble methods
│   │   ├── external/           # External APIs
│   │   ├── backtesting/        # Backtesting logic
│   │   └── repositories/       # Data repositories
│   │
│   └── visualization/          # Visualization logic
│
├── dashboard/                  # Streamlit dashboard
│   └── streamlit_app.py       # Main dashboard file
│
├── main.py                     # Entry point for predictions
├── train_models.py            # Training script
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables
```

## 🔧 Các models được sử dụng

| Model | Mô tả | Đặc điểm |
|-------|-------|----------|
| Linear Regression | Regression cơ bản | Đơn giản, nhanh, baseline |
| XGBoost | Gradient Boosting | Hiệu suất cao, xử lý missing data tốt |
| LSTM | Long Short-Term Memory | Phù hợp time series, học pattern dài hạn |
| BiLSTM | Bidirectional LSTM | Học từ cả quá khứ và tương lai |
| Transformer | Attention mechanism | State-of-the-art cho sequence data |
| Ensemble | Weighted averaging | Kết hợp nhiều models để cải thiện độ chính xác |

## 📊 Metrics đánh giá

- **RMSE** (Root Mean Square Error): Độ lệch trung bình bình phương
- **MAE** (Mean Absolute Error): Sai số trung bình tuyệt đối
- **MAPE** (Mean Absolute Percentage Error): Sai số phần trăm trung bình
- **R²** (R-squared): Hệ số xác định, càng gần 1 càng tốt

## 🔌 Data Sources

- **Crypto prices**: Binance API (real-time)
- **Macro data**: Yahoo Finance (Gold, DXY)
- **News**: NewsAPI, CryptoPanic
- **Social sentiment**: Twitter/X.com, Reddit
- **Alternative**: Nitter scraper (không cần API), Hugging Face sentiment

## 🛠️ Công nghệ sử dụng

- **Language**: Python 3.10+
- **ML Framework**: TensorFlow/Keras, XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Streamlit
- **API Clients**: ccxt (Binance), yfinance, tweepy
- **Architecture**: Clean Architecture, SOLID principles
- **Dependency Injection**: Custom container

## 📝 License

Dự án này được tạo ra cho mục đích học tập và nghiên cứu.

## 👥 Contributing

Contributions are welcome! Vui lòng:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push đến branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📧 Liên hệ

- GitHub: [@lekimphu0209](https://github.com/lekimphu0209)
- Repository: https://github.com/lekimphu0209/crypto-price-prediction

## 🙏 Acknowledgments

- Binance API cho dữ liệu crypto real-time
- Yahoo Finance cho dữ liệu macro
- Hugging Face cho models sentiment analysis
- Streamlit cho dashboard framework
