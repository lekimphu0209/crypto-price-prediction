"""
Models API Routes

Endpoints for listing trained models. Reads the Clean Architecture model
registry layout: models/{SYMBOL}/{model_slug}/meta.json
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json

router = APIRouter()

MODELS_DIR = "models"


class ModelInfo(BaseModel):
    """Information about a trained model"""
    name: str
    symbol: str
    type: str
    path: str
    metrics: Dict[str, Any]


class ModelsResponse(BaseModel):
    """Response for listing models"""
    status: str
    models: List[ModelInfo]


def _read_meta(symbol_dir: str, slug: str) -> Dict[str, Any]:
    meta_path = os.path.join(MODELS_DIR, symbol_dir, slug, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _scan(symbol_filter: str = None) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    if not os.path.exists(MODELS_DIR):
        return models

    for symbol_dir in os.listdir(MODELS_DIR):
        symbol_path = os.path.join(MODELS_DIR, symbol_dir)
        if not os.path.isdir(symbol_path):
            continue
        if symbol_filter and symbol_dir != symbol_filter:
            continue
        for slug in os.listdir(symbol_path):
            if not os.path.isdir(os.path.join(symbol_path, slug)):
                continue
            meta = _read_meta(symbol_dir, slug)
            if not meta:
                continue
            models.append(ModelInfo(
                name=meta.get("model_name", slug),
                symbol=symbol_dir,
                type=slug,
                path=os.path.join(MODELS_DIR, symbol_dir, slug),
                metrics=meta.get("metrics", {}),
            ))
    return models


@router.get("/", response_model=ModelsResponse)
async def list_models():
    """List all trained models in the registry."""
    try:
        return ModelsResponse(status="success", models=_scan())
    except Exception:
        return ModelsResponse(status="error", models=[])


@router.get("/{symbol}")
async def list_models_by_symbol(symbol: str):
    """List models for a specific symbol."""
    try:
        models = [m.dict() for m in _scan(symbol_filter=symbol)]
        return {"status": "success", "symbol": symbol, "models": models}
    except Exception:
        return {"status": "error", "symbol": symbol, "models": []}
