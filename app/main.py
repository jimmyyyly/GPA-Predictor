"""FastAPI service that serves the trained GPA model.

Run locally:
    uvicorn app.main:app --reload
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

model = joblib.load(MODEL_DIR / "model.joblib")
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())

app = FastAPI(title="GPA Predictor", version="1.0.0")


class StudentScores(BaseModel):
    math_sat: int = Field(..., ge=200, le=800, examples=[650])
    verbal_sat: int = Field(..., ge=200, le=800, examples=[600])


class Prediction(BaseModel):
    predicted_gpa: float
    model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict", response_model=Prediction)
def predict(scores: StudentScores):
    X = pd.DataFrame([scores.model_dump()])
    gpa = float(model.predict(X)[0])
    gpa = min(max(gpa, 0.0), 4.0)
    return Prediction(predicted_gpa=round(gpa, 2), model=metrics["model"])


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
