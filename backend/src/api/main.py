"""
FastAPI Backend for Crypto Price Prediction

This API exposes endpoints for training and prediction models.
It uses the Clean Architecture from src/ (TrainModelUseCase, PredictUseCase).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import predict, models

app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for predicting crypto prices using ML models",
    version="1.0.0"
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(train.router, prefix="/api/train", tags=["Training"])  # Temporarily disabled
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])


@app.get("/")
async def root():
    return {
        "message": "Crypto Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "training": "/api/train",
            "prediction": "/api/predict",
            "models": "/api/models"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
