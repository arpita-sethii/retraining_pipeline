import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import onnxruntime as ort

app = FastAPI(
    title="LSTM Time Series Forecaster",
    description="Predicts next hour oil temperature from 24 hours of input.",
    version="1.0"
)

model_state = {
    "session": None,
    "version": "1.0"
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

def load_onnx_model(path="model.onnx"):
    try:
        session = ort.InferenceSession(path)
        print(f"✅ ONNX model loaded from {path}")
        return session
    except Exception as e:
        print(f"⚠️  Could not load ONNX model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    session = load_onnx_model("model.onnx")
    model_state["session"] = session

@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="running",
        model_loaded=model_state["session"] is not None,
        model_version=model_state["version"]
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model_state["session"] is not None else "no model loaded",
        model_loaded=model_state["session"] is not None,
        model_version=model_state["version"]
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model_state["session"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(request.sequence) != 24:
        raise HTTPException(status_code=400, detail=f"Expected 24 values, got {len(request.sequence)}")

    try:
        x = np.array(request.sequence, dtype=np.float32).reshape(1, 24, 1)
        inputs = {model_state["session"].get_inputs()[0].name: x}
        pred = model_state["session"].run(None, inputs)[0][0]

        return PredictResponse(
            prediction=round(float(pred), 6),
            model_version=model_state["version"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload_model():
    session = load_onnx_model("model.onnx")
    if session is None:
        raise HTTPException(status_code=404, detail="Could not load model.")
    model_state["session"] = session
    return {"status": "reloaded", "version": model_state["version"]}