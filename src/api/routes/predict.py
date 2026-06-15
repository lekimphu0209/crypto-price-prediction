"""
Prediction API Routes

Endpoints for making predictions using trained models.
Uses the Clean Architecture prediction pipeline (PredictPipeline), wired
together in src/training/composition.py.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.training.composition import get_predict_pipeline

router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for prediction"""
    symbol: str = "BTCUSDT"
    model_name: str = "Linear Regression"
    display_days: int = 30
    horizon: int = 7


class PredictResponse(BaseModel):
    """Response for prediction - matching dashboard format"""
    status: str
    symbol: str
    model_name: str
    current_price: Optional[float] = None
    predicted_tomorrow: Optional[float] = None
    change_pct: Optional[float] = None
    confidence: Optional[float] = None
    train_r2: Optional[float] = None
    train_mae: Optional[float] = None
    hist_dates: Optional[list] = None
    hist_values: Optional[list] = None
    forecast_dates: Optional[list] = None
    forecast_values: Optional[list] = None
    message: Optional[str] = None


@router.post("/", response_model=PredictResponse)
async def predict_price(request: PredictRequest):
    """
    Make price predictions for a given symbol

    - **symbol**: Trading symbol (e.g., BTCUSDT)
    - **model_name**: Model name (Linear Regression, RNN, LSTM, BiLSTM, Transformer)
    - **display_days**: Number of historical days to display
    - **horizon**: Forecast horizon in days
    """
    try:
        result = get_predict_pipeline().predict(
            symbol=request.symbol,
            model_name=request.model_name,
            display_days=request.display_days,
            horizon=request.horizon
        )

        if not result.get("ok"):
            return PredictResponse(
                status="error",
                symbol=request.symbol,
                model_name=request.model_name,
                message=result.get("message", "Prediction failed")
            )

        return PredictResponse(
            status="success",
            symbol=request.symbol,
            model_name=request.model_name,
            current_price=result.get("current_price"),
            predicted_tomorrow=result.get("predicted_tomorrow"),
            change_pct=result.get("change_pct"),
            confidence=result.get("confidence"),
            train_r2=result.get("train_r2"),
            train_mae=result.get("train_mae"),
            hist_dates=result.get("hist_dates"),
            hist_values=result.get("hist_values"),
            forecast_dates=result.get("forecast_dates"),
            forecast_values=result.get("forecast_values"),
            message="Prediction completed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
