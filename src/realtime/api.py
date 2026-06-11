"""
FastAPI Endpoint cho Realtime Predictions
Cung cấp REST API để lấy predictions và WebSocket để push realtime updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import asyncio
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from realtime.predictor import RealtimePredictor, MultiSymbolPredictor


app = FastAPI(title="Crypto Price Prediction API", version="1.0.0")

# Global predictors
predictors = {}
supported_symbols = ["BTCUSDT", "ETHUSDT"]

# Initialize predictors
for symbol in supported_symbols:
    try:
        predictors[symbol] = RealtimePredictor(symbol)
        print(f"Đã khởi tạo predictor cho {symbol}")
    except Exception as e:
        print(f"Lỗi khi khởi tạo predictor cho {symbol}: {e}")


# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    candles: List[Dict]


class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    current_price: float
    predictions: Dict
    data_points: int


class HealthResponse(BaseModel):
    status: str
    supported_symbols: List[str]
    timestamp: str


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        supported_symbols=supported_symbols,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/symbols")
async def get_symbols():
    """Lấy danh sách symbols được hỗ trợ"""
    return {
        "symbols": supported_symbols,
        "predictors_loaded": list(predictors.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Dự đoán giá cho một symbol
    
    Args:
        request: PredictionRequest với symbol và candles
    
    Returns:
        PredictionResponse với predictions
    """
    if request.symbol not in predictors:
        return JSONResponse(
            status_code=400,
            content={"error": f"Symbol {request.symbol} không được hỗ trợ"}
        )
    
    try:
        # Convert candles to DataFrame
        df = pd.DataFrame(request.candles)
        
        # Predict
        result = predictors[request.symbol].predict_realtime(df)
        
        if 'error' in result:
            return JSONResponse(
                status_code=400,
                content=result
            )
        
        # Convert timestamp to string
        result['timestamp'] = result['timestamp'].isoformat()
        
        return PredictionResponse(**result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str, n_candles: int = 100):
    """
    Dự đoán giá cho symbol (dùng dữ liệu lịch sử từ file)
    
    Args:
        symbol: Symbol trading pair
        n_candles: Số lượng candles để sử dụng
    
    Returns:
        Predictions
    """
    if symbol not in predictors:
        return JSONResponse(
            status_code=400,
            content={"error": f"Symbol {symbol} không được hỗ trợ"}
        )
    
    try:
        # Load processed data
        symbol_lower = symbol.replace("USDT", "").lower()
        data_path = f"data/{symbol_lower}_processed_data.csv"
        
        if not os.path.exists(data_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Không tìm thấy dữ liệu cho {symbol}"}
            )
        
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Lấy n candles gần nhất
        candles = df.tail(n_candles)
        
        # Predict
        result = predictors[symbol].predict_realtime(candles)
        
        if 'error' in result:
            return JSONResponse(
                status_code=400,
                content=result
            )
        
        # Convert timestamp to string
        result['timestamp'] = result['timestamp'].isoformat()
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/history/{symbol}")
async def get_prediction_history(symbol: str, n: int = 100):
    """
    Lấy lịch sử predictions cho symbol
    
    Args:
        symbol: Symbol trading pair
        n: Số lượng predictions gần nhất
    
    Returns:
        List of predictions
    """
    if symbol not in predictors:
        return JSONResponse(
            status_code=400,
            content={"error": f"Symbol {symbol} không được hỗ trợ"}
        )
    
    try:
        history = predictors[symbol].get_prediction_history(n)
        
        # Convert timestamps to strings
        for h in history:
            h['timestamp'] = h['timestamp'].isoformat()
        
        return {
            "symbol": symbol,
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/latest/{symbol}")
async def get_latest_prediction(symbol: str):
    """
    Lấy prediction gần nhất cho symbol
    
    Args:
        symbol: Symbol trading pair
    
    Returns:
        Latest prediction
    """
    if symbol not in predictors:
        return JSONResponse(
            status_code=400,
            content={"error": f"Symbol {symbol} không được hỗ trợ"}
        )
    
    try:
        prediction = predictors[symbol].get_latest_prediction()
        
        if prediction is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Chưa có prediction nào"}
            )
        
        # Convert timestamp to string
        prediction['timestamp'] = prediction['timestamp'].isoformat()
        
        return prediction
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint để nhận realtime predictions
    
    Args:
        websocket: WebSocket connection
        symbol: Symbol để nhận predictions
    """
    await manager.connect(websocket)
    
    try:
        if symbol not in predictors:
            await websocket.send_json({"error": f"Symbol {symbol} không được hỗ trợ"})
            await websocket.close()
            return
        
        await websocket.send_json({
            "type": "connected",
            "symbol": symbol,
            "message": f"Đã kết nối đến {symbol} prediction stream"
        })
        
        # Trong thực tế, đây sẽ được kích hoạt bởi realtime data collector
        # For demo, gửi prediction định kỳ
        while True:
            # Load latest prediction
            prediction = predictors[symbol].get_latest_prediction()
            
            if prediction:
                prediction['timestamp'] = prediction['timestamp'].isoformat()
                await websocket.send_json({
                    "type": "prediction",
                    "data": prediction
                })
            
            # Wait 5 seconds
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected from {symbol}")


if __name__ == "__main__":
    import uvicorn
    print("Đang khởi động API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
