from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

app = FastAPI(
    title="SmartAgriML - Crop Failure Risk Prediction API",
    description="Predict crop failure risk score using bagging-based ML model",
    version="1.0"
)

# -----------------------------
# Input Schema
# -----------------------------
class CropInput(BaseModel):
    Area: float
    Item: str
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_risk(data: CropInput):
    try:
        logging.info("Received prediction request")

        predictor = PredictPipeline()
        risk_score = predictor.predict(data.dict())

        return {
            "crop_failure_risk_score": float(risk_score),
            "risk_level": (
                "High" if risk_score >= 70 else
                "Medium" if risk_score >= 40 else
                "Low"
            )
        }

    except Exception as e:
        return {"error": str(e)}
