"""
Prediction API Routes

Endpoints for trend-direction predictions using trained models.
Uses the Clean Architecture prediction pipeline (PredictPipeline), wired
together in src/training/composition.py.

Models are up/down trend classifiers, so the response is a direction
(UP/DOWN) + probability + the tuned decision threshold - not a price point.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.training.composition import get_predict_pipeline

router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for prediction"""
    symbol: str = "BTCUSDT"
    model_name: str = "Logistic Regression"
    display_days: int = 30
    horizon: Optional[int] = None  # None -> use the model's trained horizon


class PredictResponse(BaseModel):
    """Trend-direction prediction response."""
    status: str
    symbol: str
    model_name: str
    mode: Optional[str] = None            # classification | regression
    direction: Optional[str] = None       # UP | DOWN
    direction_label: Optional[str] = None # TANG | GIAM
    probability: Optional[float] = None   # P(UP) for classification models
    predicted_return: Optional[float] = None  # log-return for regression models
    threshold: Optional[float] = None
    horizon: Optional[int] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    confidence: Optional[float] = None
    train_metrics: Optional[Dict[str, Any]] = None
    hist_dates: Optional[list] = None
    hist_values: Optional[list] = None
    message: Optional[str] = None


@router.post("/", response_model=PredictResponse)
async def predict_trend(request: PredictRequest):
    """
    Predict the trend direction for a given symbol.

    - **symbol**: Trading symbol (e.g., BTCUSDT)
    - **model_name**: Registered model name (Logistic Regression, RNN, LSTM, BiLSTM, Transformer, ...)
    - **display_days**: Number of historical days to return for charting
    - **horizon**: Ignored for classifiers - the model's trained horizon is used
    """
    try:
        result = get_predict_pipeline().predict(
            symbol=request.symbol,
            model_name=request.model_name,
            display_days=request.display_days,
            horizon=request.horizon,
        )

        if not result.get("ok"):
            return PredictResponse(
                status="error",
                symbol=request.symbol,
                model_name=request.model_name,
                message=result.get("message", "Prediction failed"),
            )

        return PredictResponse(
            status="success",
            symbol=request.symbol,
            model_name=result.get("model_name", request.model_name),
            mode=result.get("mode"),
            direction=result.get("direction"),
            direction_label=result.get("direction_label"),
            probability=result.get("probability"),
            predicted_return=result.get("predicted_return"),
            threshold=result.get("threshold"),
            horizon=result.get("horizon"),
            current_price=result.get("current_price"),
            change_pct=result.get("change_pct"),
            confidence=result.get("confidence"),
            train_metrics=result.get("train_metrics"),
            hist_dates=result.get("hist_dates"),
            hist_values=result.get("hist_values"),
            message="Prediction completed successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
