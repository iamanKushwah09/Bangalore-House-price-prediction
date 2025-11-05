from __future__ import annotations
from typing import Literal, Optional
from pathlib import Path
import numpy as np
import pickle

from fastapi import FastAPI, HTTPException, Query, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bangalore_model.pkl"

app = FastAPI(
    title="Bangalore House Price Prediction API",
    description="Predict house price (in lakhs) using a trained Linear Regression model.",
    version="1.0.0",
)

# --- CORS (allow Streamlit on localhost) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once ---
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please place bangalore_model.pkl next to this file.")

with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

MODEL = payload["model"]
FEATURE_ORDER = payload.get("feature_order", ["bath", "balcony", "total_sqft_int", "bhk", "price_per_sqft"])

# ------------ Pydantic Schemas with Validation --------------

class PredictRequest(BaseModel):
    bath: Annotated[int, Field(ge=1, le=12, description="Number of bathrooms (>=1)")]
    balcony: Annotated[int, Field(ge=0, le=6, description="Number of balconies (>=0)")]
    total_sqft_int: Annotated[float, Field(gt=200.0, description="Total area in sqft (>200)")]
    bhk: Annotated[int, Field(ge=1, le=12, description="Number of bedrooms (>=1)")]
    price_per_sqft: Annotated[float, Field(gt=0, description="Assumed/derived price per sqft (₹)")]
    # NOTE: price_per_sqft was part of your trained features; inference me isko input lena hoga.

    @model_validator(mode="after")
    def logical_rules(self):
        # rule-1: bath should not be absurd wrt bhk
        if self.bath > self.bhk + 2:
            raise ValueError("Invalid: bath should typically be ≤ (bhk + 2).")
        # rule-2: minimum area per BHK (approx heuristic used in your EDA)
        if (self.total_sqft_int / self.bhk) < 350:
            raise ValueError("Invalid: sqft per BHK must be ≥ 350 (data cleaning rule).")
        return self

class PredictResponse(BaseModel):
    predicted_price_lakhs: float
    feature_order: list[str]
    features_used: list[float]


# ------------ Utility ----------------

def make_feature_vector(req: PredictRequest) -> np.ndarray:
    # Keep strict order identical to training
    feature_map = {
        "bath": req.bath,
        "balcony": req.balcony,
        "total_sqft_int": req.total_sqft_int,
        "bhk": req.bhk,
        "price_per_sqft": req.price_per_sqft,
    }
    x = [float(feature_map[name]) for name in FEATURE_ORDER]
    return np.array([x], dtype=float)


# ------------ Routes ------------------

@app.get("/", summary="Health-check")
def root():
    return {"status": "ok", "message": "Bangalore Price API is running", "model_features": FEATURE_ORDER}

@app.get("/model/info", summary="Model metadata")
def model_info():
    return {"feature_order": FEATURE_ORDER, "model_type": type(MODEL).__name__}

@app.post("/predict", response_model=PredictResponse, summary="Predict via JSON body (recommended)")
def predict(req: PredictRequest):
    try:
        x = make_feature_vector(req)
        y_pred = MODEL.predict(x)[0]
        return PredictResponse(
            predicted_price_lakhs=round(float(y_pred), 2),
            feature_order=FEATURE_ORDER,
            features_used=x[0].tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# ----- Alternative style: Path + Query params (to showcase Annotated/Path/Query) -----

@app.get(
    "/predict/{bhk}",
    response_model=PredictResponse,
    summary="Predict via Path + Query parameters (demo of Annotated, Path, Query).",
)
def predict_via_query(
    bhk: Annotated[int, FPath(ge=1, le=12, description="Bedrooms in path")],
    bath: Annotated[int, Query(ge=1, le=12, description="Bathrooms (query)")] = 2,
    balcony: Annotated[int, Query(ge=0, le=6, description="Balconies (query)")] = 1,
    total_sqft_int: Annotated[float, Query(gt=200.0, description="Total area in sqft (>200)")] = 1000.0,
    price_per_sqft: Annotated[float, Query(gt=0, description="₹ per sqft (query)")] = 6000.0,
):
    req = PredictRequest(
        bath=bath,
        balcony=balcony,
        total_sqft_int=total_sqft_int,
        bhk=bhk,
        price_per_sqft=price_per_sqft,
    )
    return predict(req)
