# Thiết Kế Hệ Thống Realtime - Crypto Price Prediction

## Tổng Quan

Hệ thống realtime để dự đoán giá ETH/Bitcoin theo yêu cầu giáo viên, bao gồm:
- Thu thập dữ liệu realtime từ nhiều nguồn
- Xử lý và tạo features
- Dự đoán giá bằng các mô hình đã train
- Cập nhật kết quả liên tục

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    Realtime Prediction System                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Data Sources │    │  Data Sources │    │  Data Sources │
│   Binance API │    │  Yahoo Finance│    │  X.com API    │
│   (OHLCV)     │    │  (Gold, DXY)  │    │  (Sentiment)  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ Data Collector│
                    │   (Realtime)  │
                    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │Feature Engine │
                    │   (Streaming) │
                    └───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Linear Reg.   │    │     RNN       │    │  LSTM/Trans.  │
│   Model       │    │   Model       │    │    Model      │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ Ensemble/     │
                    │   Voting      │
                    └───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │   API     │   │ Dashboard │   │   Alerts  │
      │ Endpoint  │   │  Monitor  │   │  System   │
      └───────────┘   └───────────┘   └───────────┘
```

## Các Thành Phần

### 1. Realtime Data Collector
- Kết nối Binance WebSocket để lấy giá realtime
- Fetch dữ liệu Gold và DXY từ Yahoo Finance (cập nhật hàng ngày)
- Fetch sentiment từ X.com (cập nhật theo thời gian thực)
- Lưu dữ liệu vào buffer/sliding window

### 2. Feature Engine (Streaming)
- Tính technical indicators realtime (MA, RSI, MACD, etc.)
- Tính macro features (gold returns, DXY returns)
- Tính sentiment features
- Maintain sliding window của features

### 3. Prediction Engine
- Load các models đã train (Linear Regression, RNN, LSTM, Transformer)
- Predict trên dữ liệu mới
- Ensemble predictions từ nhiều models

### 4. API Endpoint
- REST API để lấy predictions
- WebSocket để push realtime predictions
- Endpoint để lấy historical predictions

### 5. Dashboard Monitor
- Hiển thị giá realtime
- Hiển thị predictions
- Hiển thị model performance
- Alerts khi có sự kiện quan trọng

### 6. Alert System
- Gửi alerts khi price vượt ngưỡng
- Gửi alerts khi sentiment thay đổi mạnh
- Gửi alerts khi predictions có độ tin cậy thấp

## Cấu Trúc File

```
DAMH/
├── src/
│   ├── realtime/
│   │   ├── __init__.py
│   │   ├── data_collector.py    # Realtime data collector
│   │   ├── feature_engine.py    # Streaming feature engineering
│   │   ├── predictor.py         # Realtime predictor
│   │   ├── api.py               # FastAPI endpoints
│   │   └── alerts.py            # Alert system
│   ├── models/
│   │   └── (existing models)
│   └── (existing modules)
├── realtime_predictor.py        # Main realtime script
├── api_server.py                # API server
├── dashboard/                   # Dashboard files
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── config/
│   └── realtime_config.json     # Configuration
└── requirements.txt             # Updated dependencies
```

## Công Nghệ Sử Dụng

- **Data Collection**: Binance WebSocket, yfinance, Tweepy (X.com API)
- **Processing**: pandas, numpy
- **ML/DL**: scikit-learn, TensorFlow/Keras
- **API**: FastAPI, uvicorn
- **WebSocket**: websockets
- **Dashboard**: Plotly Dash hoặc Streamlit
- **Alerts**: Telegram Bot API, Email SMTP

## Luồng Hoạt Động

1. **Initialization**
   - Load các models đã train
   - Connect đến data sources
   - Initialize sliding windows

2. **Data Collection Loop**
   - Fetch realtime data từ Binance WebSocket
   - Fetch macro data periodically
   - Fetch sentiment data periodically
   - Update data buffer

3. **Feature Engineering**
   - Calculate technical indicators
   - Calculate macro features
   - Calculate sentiment features
   - Update feature buffer

4. **Prediction**
   - Predict với Linear Regression
   - Predict với RNN/LSTM
   - Ensemble predictions
   - Calculate confidence intervals

5. **Output**
   - Push predictions via WebSocket
   - Update API endpoints
   - Update dashboard
   - Check alerts

## Cấu Hình

```json
{
  "data_sources": {
    "binance": {
      "symbol": "BTCUSDT",
      "interval": "1m",
      "websocket_url": "wss://stream.binance.com:9443/ws"
    },
    "macro": {
      "update_interval": "1h",
      "gold_symbol": "GLD",
      "dxy_symbol": "UUP"
    },
    "sentiment": {
      "update_interval": "5m",
      "keywords": ["bitcoin", "ethereum", "crypto"]
    }
  },
  "models": {
    "linear_regression": {
      "enabled": true,
      "model_path": "models/linear_regression_btc.pkl"
    },
    "rnn": {
      "enabled": true,
      "model_path": "models/rnn_btc.h5"
    },
    "lstm": {
      "enabled": false,
      "model_path": "models/lstm_btc.h5"
    }
  },
  "prediction": {
    "update_interval": "1m",
    "ensemble_method": "weighted_average",
    "confidence_threshold": 0.7
  },
  "alerts": {
    "enabled": true,
    "price_change_threshold": 0.05,
    "sentiment_change_threshold": 0.3
  }
}
```

## Các Bước Triển Khai

1. **Phase 1**: Realtime Data Collector với Binance WebSocket
2. **Phase 2**: Feature Engineering streaming
3. **Phase 3**: Realtime Prediction với existing models
4. **Phase 4**: API Endpoint
5. **Phase 5**: Dashboard Monitor
6. **Phase 6**: Alert System
7. **Phase 7**: Sentiment Analysis integration
