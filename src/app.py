import torch
import mlflow
import mlflow.pytorch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from mlflow import MlflowClient

app = FastAPI(
    title="LSTM Time Series Forecaster",
    description="Predicts next hour oil temperature from 24 hours of input. Auto-retrains on drift.",
    version="1.0"
)

# Global model state
model_state = {
    "model": None,
    "version": None
}

class PredictRequest(BaseModel):
    sequence: List[float]  # exactly 24 values, scaled 0-1

class PredictResponse(BaseModel):
    prediction: float
    model_version: str
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

def load_latest_model():
    """Load latest registered model from MLflow registry."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions("LSTMForecaster")
        if not versions:
            return None, None
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/LSTMForecaster/{latest.version}"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print(f"✅ Loaded LSTMForecaster v{latest.version} from MLflow registry")
        return model, latest.version
    except Exception as e:
        print(f"⚠️  Could not load model from registry: {e}")
        return None, None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model, version = load_latest_model()
    model_state["model"] = model
    model_state["version"] = str(version) if version else "none"

@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="running",
        model_loaded=model_state["model"] is not None,
        model_version=model_state["version"]
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model_state["model"] is not None else "no model loaded",
        model_loaded=model_state["model"] is not None,
        model_version=model_state["version"]
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run the pipeline first.")

    if len(request.sequence) != 24:
        raise HTTPException(status_code=400, detail=f"Expected 24 values, got {len(request.sequence)}")

    try:
        x = torch.tensor(request.sequence, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(-1)  # (1, 24, 1)

        with torch.no_grad():
            pred = model_state["model"](x).item()

        return PredictResponse(
            prediction=round(pred, 6),
            model_version=model_state["version"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload_model():
    """Reload latest model from MLflow registry."""
    model, version = load_latest_model()
    if model is None:
        raise HTTPException(status_code=404, detail="No model found in registry.")
    model_state["model"] = model
    model_state["version"] = str(version)
    return {"status": "reloaded", "version": str(version)}