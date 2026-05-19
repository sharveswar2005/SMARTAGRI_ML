from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

app = FastAPI(
    title="SmartAgriML - Crop Failure Risk Prediction API",
    description="Predict crop failure risk score using bagging-based ML model",
    version="1.0"
)

from pydantic import BaseModel, Field

class CropInput(BaseModel):
    Area: float = Field(..., gt=0, description="Area in hectares")
    Item: str = Field(..., min_length=2, description="Crop name")
    Year: int = Field(..., ge=1900, le=2100, description="Year of crop")
    average_rain_fall_mm_per_year: float = Field(..., ge=0, description="Average rain fall in mm per year")
    pesticides_tonnes: float = Field(..., ge=0, description="Pesticides in tonnes")
    avg_temp: float = Field(..., ge=-50, le=60, description="Average temperature in Celsius")

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_risk(data: CropInput):
    try:
        logging.info("Received prediction request")

        predictor = PredictPipeline()
        risk_score, key_factors = predictor.predict(data.dict())

        return {
            "crop_failure_risk_score": float(risk_score),
            "risk_level": (
                "High" if risk_score >= 70 else
                "Medium" if risk_score >= 40 else
                "Low"
            ),
            "confidence": 0.92, # Simulated confidence level metric
            "key_factors": key_factors
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}
