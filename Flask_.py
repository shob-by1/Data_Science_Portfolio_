# Flask - serving a trained ML model as a web API
# Dataset: Breast Cancer Wisconsin (same model-training pattern you've used before)
#
# Needs: pip install flask scikit-learn joblib
#
# This file does TWO things when you run different parts of it:
#   1. Trains and saves a model (run once)
#   2. Loads that saved model and serves predictions via a Flask API (run as the app)
#
# Run with:  python 37_flask_app.py
# Then test with curl or your browser - see the bottom of this file for examples.

import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

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

    # save BOTH the model and the scaler - you need the exact same scaling
    # at prediction time that you used during training
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ============================================================
# PART 2 - Load the saved model/scaler ONCE at startup
# ============================================================
# loading inside every request would be slow - load once, reuse for every call
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = load_breast_cancer().feature_names.tolist()

# ============================================================
# PART 3 - The Flask app itself
# ============================================================
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Breast Cancer prediction API is running.",
        "endpoints": {
            "/predict": "POST - send 30 feature values, get a prediction back",
            "/features": "GET - list the feature names expected, in order"
        }
    })


@app.route("/features", methods=["GET"])
def features():
    return jsonify({"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ---- basic input validation ----
    if data is None or "features" not in data:
        return jsonify({"error": "Request body must include a 'features' list"}), 400

    features_input = data["features"]

    if len(features_input) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features_input)}"
        }), 400

    try:
        X_input = np.array(features_input, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({"error": "All features must be numeric"}), 400

    # ---- apply the SAME scaling used during training ----
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return jsonify({
        "prediction": int(prediction),
        "prediction_label": "benign" if prediction == 1 else "malignant",
        "probability_malignant": round(float(probability[0]), 4),
        "probability_benign": round(float(probability[1]), 4)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# ============================================================
# HOW TO TEST THIS (once the app is running)
# ============================================================
#
# In a browser, visit:
#   http://127.0.0.1:5000/
#   http://127.0.0.1:5000/features
#
# To send a prediction request, use curl (in a separate terminal):
#
#   curl -X POST http://127.0.0.1:5000/predict ^
#        -H "Content-Type: application/json" ^
#        -d "{\"features\": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
#
# (that "^" line-continuation syntax is for Windows cmd - on Mac/Linux use "\" instead)
#
# Or from Python, in a separate script/notebook cell:
#
#   import requests
#   response = requests.post(
#       "http://127.0.0.1:5000/predict",
#       json={"features": [17.99, 10.38, 122.8, ...]}   # all 30 values
#   )
#   print(response.json())