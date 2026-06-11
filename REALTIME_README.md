# Hệ Thống Dự Đoán Giá Crypto Realtime

## Tổng Quan

Hệ thống realtime để dự đoán giá ETH/Bitcoin theo thời gian thực, được thiết kế theo yêu cầu giáo viên.

## Cấu Trúc

```
DAMH/
├── src/realtime/
│   ├── __init__.py
│   ├── data_collector.py    # Binance WebSocket collector
│   ├── predictor.py         # Realtime predictor với trained models
│   └── api.py               # FastAPI endpoints
├── realtime_predictor.py    # Script chính chạy realtime
├── dashboard/
│   └── index.html           # Dashboard monitor
└── REALTIME_SYSTEM_DESIGN.md # Thiết kế chi tiết
```

## Cài Đặt

```bash
# Cài đặt dependencies mới cho realtime
pip install websockets fastapi uvicorn pydantic
```

## Cách Sử Dụng

### 1. Chạy Realtime Predictor (Command Line)

```bash
python realtime_predictor.py --symbol BTCUSDT --interval 1m
```

Options:
- `--symbol`: Trading pair (BTCUSDT, ETHUSDT)
- `--interval`: Interval (1m, 5m, 15m, 1h)

### 2. Chạy API Server

```bash
python -m src.realtime.api
```

Hoặc:
```bash
uvicorn src.realtime.api:app --host 0.0.0.0 --port 8000 --reload
```

API Endpoints:
- `GET /` - Health check
- `GET /symbols` - Danh sách symbols được hỗ trợ
- `POST /predict` - Dự đoán với dữ liệu tùy chỉnh
- `GET /predict/{symbol}` - Dự đoán với dữ liệu lịch sử
- `GET /history/{symbol}` - Lịch sử predictions
- `GET /latest/{symbol}` - Prediction gần nhất
- `WS /ws/{symbol}` - WebSocket realtime updates

### 3. Mở Dashboard

Mở file `dashboard/index.html` trong trình duyệt, hoặc:

```bash
# Sử dụng Python HTTP server
cd dashboard
python -m http.server 8080
```

Sau đó mở http://localhost:8080

## Tính Năng

### Realtime Data Collector
- Kết nối Binance WebSocket
- Thu thập giá realtime
- Thu thập OHLCV theo interval
- Tự động cập nhật macro data (Gold, DXY)

### Realtime Predictor
- Load trained models (Linear Regression, RNN)
- Predict trên dữ liệu mới
- Ensemble predictions từ nhiều models
- Lưu lịch sử predictions

### API Server
- REST API endpoints
- WebSocket cho realtime updates
- Support multiple symbols

### Dashboard
- Hiển thị giá realtime
- Hiển thị predictions
- Chart price history
- Activity log
- Connection status

## Kết Quả

Hệ thống sử dụng các models đã train trước:
- **Linear Regression**: R² = 0.91 (hiệu quả tốt)
- **RNN**: Cần tinh chỉnh thêm (hiệu quả kém trên test hiện tại)

## Các Bước Tiếp Theo

1. **Cải thiện RNN/LSTM**: Tinh chỉnh hyperparameters, thêm dữ liệu
2. **Sentiment Analysis**: Tích hợp X.com API để phân tích sentiment
3. **Ensemble Methods**: Kết hợp nhiều models tốt hơn
4. **Alert System**: Gửi alerts khi có sự kiện quan trọng
5. **Backtesting**: Kiểm tra hiệu quả trading
6. **Advanced Models**: Implement BiLSTM, Transformer

## Lưu Ý

- Hệ thống cần models đã train trong `models/` directory
- WebSocket kết nối đến Binance public streams (không cần API key)
- Dashboard sử dụng Chart.js cho visualization
- API server chạy trên port 8000 mặc định

## Troubleshooting

**Lỗi "Model not found"**: Chạy `python main.py` trước để train models

**Lỗi WebSocket**: Kiểm tra kết nối internet, Binance có thể bị block ở một số quốc gia

**Lỗi API port 8000 đã được sử dụng**: Thay đổi port trong uvicorn command
