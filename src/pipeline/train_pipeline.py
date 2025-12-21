import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # 1️⃣ Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # 2️⃣ Data Validation
            import pandas as pd
            train_df = pd.read_csv(train_path)

            validation = DataValidation()
            validation.validate_dataset(train_df)

            # 3️⃣ Data Transformation
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test = (
                transformation.initiate_data_transformation(
                    train_path, test_path
                )
            )

            # 4️⃣ Model Training
            trainer = ModelTrainer()
            model_name, rmse, r2 = trainer.initiate_model_training(
                X_train, X_test, y_train, y_test
            )

            logging.info(
                f"Training completed | Model: {model_name}, RMSE: {rmse}, R2: {r2}"
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
