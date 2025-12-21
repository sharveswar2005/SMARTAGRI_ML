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

            logging.info("Prediction completed successfully")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
