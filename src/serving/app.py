from __future__ import annotations

import os
from typing import Any
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.registry.promote import rollback_production
from src.utils.io import ensure_feature_cross

app = FastAPI(title="Manufacturing Defect Prediction Service", version="1.0")

def _load_model(stage: str = "Production"):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.model_name}/{stage}"
    return mlflow.sklearn.load_model(uri)

# Lazy-loaded global model (stateless request handling; model is immutable per process)
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
_model = None

class PredictRequest(BaseModel):
    # Accept arbitrary key/value (numeric + categorical)
    record: dict[str, Any]

class PredictResponse(BaseModel):
    defect_probability: float
    defect_pred: int

@app.get("/health")
def health():
    return {"status": "ok", "model_stage": MODEL_STAGE}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global _model
    if _model is None:
        _model = _load_model(MODEL_STAGE)

    df = pd.DataFrame([req.record])
    df = ensure_feature_cross(df)

    # Ensure model sees all expected columns: the safest approach is to pass whatever present.
    try:
        prob = float(_model.predict_proba(df)[0, 1])
        pred = int(prob >= 0.5)
        return PredictResponse(defect_probability=prob, defect_pred=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request / feature mismatch: {e}")

@app.post("/fallback/rollback")
def rollback():
    """Algorithmic fallback: rollback Production to previous known-good model in the registry."""
    rollback_production()
    return {"status": "rolled_back"}
