from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib

app = FastAPI(title="Real-Time Fraud Detection API", version="1.0.0")

model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "fraud_detection_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded successfully! Expected features: {model.n_features_in_}")
    else:
        raise RuntimeError("Model file not found!")

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    card_type: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    amount = transaction.amount
    card_type = transaction.card_type

    # KEY FIX: Use the exact number of features the model was trained on (336).
    # Never hardcode this number again!
    n_features = model.n_features_in_
    features = np.zeros((1, n_features))

    # Feature mapping (in production, these values come from the Feast Feature Store)
    features[0][0] = amount
    features[0][1] = amount * 0.1
    features[0][2] = 1.0 if amount > 1000 else 0.0

    # Supports both card brands AND ProductCD codes (W, H, C, S, R) from Kafka
    card_encoding = {
        'visa': 0, 'mastercard': 1, 'discover': 2, 'american express': 3,
        'w': 4, 'h': 5, 'c': 6, 's': 7, 'r': 8, 'unknown': 9
    }
    features[0][3] = card_encoding.get(card_type.lower(), 0)
    features[0][4:10] = np.random.rand(6) * 100

    # ML Prediction
    ml_probability = float(model.predict_proba(features)[0][1])

    # INDUSTRY PRACTICE: Hybrid Scoring (ML + Business Rules)
    rule_score = 0.0
    if amount > 5000:
        rule_score += 0.4
    if amount > 10000:
        rule_score += 0.3
    if card_type.lower() == 'american express':
        rule_score += 0.1

    final_probability = min((ml_probability * 0.7) + (rule_score * 0.3), 1.0)

    if final_probability > 0.7:
        risk_level, is_fraud = "HIGH", True
    elif final_probability > 0.3:
        risk_level, is_fraud = "MEDIUM", False
    else:
        risk_level, is_fraud = "LOW", False

    return {
        "transaction_id": transaction.transaction_id,
        "fraud_probability": round(final_probability, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "ml_score": round(ml_probability, 4),
        "rule_score": round(rule_score, 4)
    }
# --- optional: serve the built React viewer from the same origin (free-tier deployment) ---
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_DIST = Path(os.getenv("FRONTEND_DIST", "deploy/dist"))
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="viewer-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str):
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")