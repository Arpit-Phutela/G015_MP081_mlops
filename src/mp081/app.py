import os
import json
import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from mp081.cleaning import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# ---------- Logging ----------
logger = logging.getLogger("mp081_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

logger.propagate = False


# ---------- Application ----------
app = FastAPI(
    title="MP081 Energy Prediction API",
    version="1.0.0",
)


# ---------- Request schema ----------
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_start_pct: float
    session_minutes: float
    charger_kw: float = Field(le=150)
    temp_c: float = Field(ge=2.5)
    vehicle_age_yr: float
    queue_before: float
    vehicle: str
    station_type: str


# ---------- Model loading ----------
def get_model():
    """Load the champion model."""
    model_path = os.getenv(
        "MODEL_PATH",
        "models:/MP081-Energy-Model@champion",
    )

    if model_path.startswith("models:/"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    return mlflow.sklearn.load_model(model_path)


# ---------- Health endpoint ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict(request: PredictionRequest):
    model = get_model()

    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    # Apply the same category normalization used during training.
    for column in CATEGORICAL_FEATURES:
        input_data[column] = (
            input_data[column].astype("string").str.strip().str.lower()
        )

    prediction = float(
        model.predict(input_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
    )

    # One structured JSON log line per prediction.
    logger.info(json.dumps({
        "event": "prediction",
        "model": "MP081-Energy-Model",
        "prediction": prediction,
    }))

    return {
        "prediction": prediction,
        "unit": "kWh",
    }
