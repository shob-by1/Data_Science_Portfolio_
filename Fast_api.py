# FastAPI - serving a trained ML model as a web API
# Same model and dataset as the Flask version, so you can compare the two
# frameworks directly on identical functionality.
#
# Needs: pip install fastapi uvicorn scikit-learn joblib pydantic
#
# Run with:  uvicorn 38_fastapi_app:app --reload
# (NOT "python 38_fastapi_app.py" - FastAPI apps are run through uvicorn, a
#  separate ASGI server, unlike Flask's built-in app.run())
#
# Once running, FastAPI gives you a free interactive docs page at:
#   http://127.0.0.1:8000/docs
# This is one of the biggest practical differences from Flask - no extra
# work needed to get a testable API playground.

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import List

MODEL_PATH = "breast_cancer_model.joblib"
SCALER_PATH = "breast_cancer_scaler.joblib"

# ============================================================
# PART 1 - Train and save the model (only runs if no saved model exists yet)
# ============================================================
if not os.path.exists(MODEL_PATH):
    print("No saved model found - training one now...")

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ============================================================
# PART 2 - Load the saved model/scaler ONCE at startup
# ============================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

cancer_meta = load_breast_cancer()
FEATURE_NAMES = cancer_meta.feature_names.tolist()
N_FEATURES = len(FEATURE_NAMES)

# ============================================================
# PART 3 - Request/response schemas (this is FastAPI's big advantage over Flask)
# ============================================================
# Pydantic models define EXACTLY what shape of data is valid - FastAPI
# validates incoming requests against this automatically, and rejects bad
# ones with a clear error BEFORE your code even runs. Flask needs you to
# write that validation by hand (as the manual checks in the Flask version did).

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Exactly {N_FEATURES} numeric feature values, in the order from /features"
    )

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_malignant: float
    probability_benign: float

class FeaturesResponse(BaseModel):
    features: List[str]
    count: int

# ============================================================
# PART 4 - The FastAPI app itself
# ============================================================
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Predicts malignant vs benign from 30 numeric tumor features",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Breast Cancer prediction API is running.",
        "endpoints": {
            "/predict": "POST - send 30 feature values, get a prediction back",
            "/features": "GET - list the feature names expected, in order",
            "/docs": "GET - interactive API documentation (try it in your browser)"
        }
    }


@app.get("/features", response_model=FeaturesResponse)
def features():
    return {"features": FEATURE_NAMES, "count": N_FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # by the time we get here, FastAPI has ALREADY confirmed:
    #   - the request has a "features" field
    #   - it's a list
    #   - it has exactly N_FEATURES numbers
    #   - every value is actually numeric
    # none of that manual checking from the Flask version is needed here

    X_input = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return PredictionResponse(
        prediction=int(prediction),
        prediction_label="benign" if prediction == 1 else "malignant",
        probability_malignant=round(float(probability[0]), 4),
        probability_benign=round(float(probability[1]), 4)
    )


# ============================================================
# HOW TO RUN AND TEST THIS
# ============================================================
#
# 1. Start the server (from a terminal, in this file's folder):
#      uvicorn 38_fastapi_app:app --reload
#
# 2. Open the auto-generated interactive docs in your browser:
#      http://127.0.0.1:8000/docs
#    This lets you fill in a form and click "Execute" - no curl needed.
#
# 3. Or test with curl, same style as the Flask version:
#
#   curl -X POST http://127.0.0.1:8000/predict ^
#        -H "Content-Type: application/json" ^
#        -d "{\"features\": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
#
# 4. Or from Python:
#
#   import requests
#   response = requests.post(
#       "http://127.0.0.1:8000/predict",
#       json={"features": [17.99, 10.38, 122.8, ...]}
#   )
#   print(response.json())