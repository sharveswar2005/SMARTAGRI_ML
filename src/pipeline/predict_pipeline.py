import os
import sys
import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, input_data: dict):
        """
        input_data: dictionary of feature values
        returns: predicted risk score
        """
        try:
            logging.info("Starting prediction pipeline")

            # Convert input dictionary to DataFrame
            input_df = pd.DataFrame([input_data])

            # Load preprocessor and model
            preprocessor = joblib.load(self.preprocessor_path)
            model = joblib.load(self.model_path)

            # Transform input data
            input_processed = preprocessor.transform(input_df)

            # Predict risk score
            prediction = model.predict(input_processed)[0]
            
            # Extract Feature Importances if available
            key_factors = []
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                factors = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                # Parse clean names
                key_factors = [{"feature": str(f).split("__")[-1], "importance": float(imp)} for f, imp in factors[:5]]

            logging.info("Prediction completed successfully")
            return prediction, key_factors

        except Exception as e:
            raise CustomException(e, sys)
