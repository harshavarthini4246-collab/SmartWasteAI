"""Inference helpers for the trained SmartWaste AI regression model."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "food_waste_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"


def model_ready():
    return MODEL_PATH.exists() and METRICS_PATH.exists()


def get_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_importance(limit=8):
    if not IMPORTANCE_PATH.exists():
        return []
    df = pd.read_csv(IMPORTANCE_PATH).head(limit)
    return [{"feature": str(row.feature).replace("numeric__", "").replace("categorical__", ""), "importance": round(float(row.importance), 4)} for row in df.itertuples(index=False)]


def predict_waste(form_values):
    if not model_ready():
        raise RuntimeError("ML model is not trained yet. Run python download_dataset.py, then python train_model.py.")
    saved = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{key: form_values[key] for key in saved["features"]}])
    predicted = max(0.0, float(saved["pipeline"].predict(row)[0]))
    quantity = float(form_values["Quantity of Food"])
    predicted = min(predicted, quantity)
    rate = (predicted / quantity * 100) if quantity > 0 else 0
    if rate >= 25:
        risk = "High"
        advice = "Review preparation quantity, event attendance and storage practices before the next service."
    elif rate >= 15:
        risk = "Moderate"
        advice = "Track leftovers by food type and adjust the next batch using observed demand."
    else:
        risk = "Lower"
        advice = "Continue tracking actual leftovers to confirm whether this pattern repeats."
    return {"predicted_waste": round(predicted, 2), "waste_rate": round(rate, 1), "risk": risk, "advice": advice}
