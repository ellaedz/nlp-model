from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/civiclear_nlp_model.joblib")

app = FastAPI(title="CIVICLEAR NLP API")
model = joblib.load(MODEL_PATH)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "CIVICLEAR NLP API is running"}


@app.post("/predict-text")
def predict_text(request: TextRequest):
    text = request.text.strip()

    if not text:
        return {
            "detected": False,
            "violation_type": None,
            "confidence": 0,
            "message": "Empty text received",
        }

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities) * 100)

    return {
        "detected": prediction != "no_violation",
        "violation_type": prediction,
        "confidence": round(confidence, 2),
        "message": "Text prediction completed",
    }
