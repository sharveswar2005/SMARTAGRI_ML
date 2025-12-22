import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw_data.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")

        try:
            df = pd.read_csv(os.path.join("data", "crop_data.csv"))
            logging.info("Dataset loaded successfully")

            logging.info(f"Dataset shape: {df.shape}")
            logging.info(f"Dataset columns: {list(df.columns)}")

            if df.empty:
                raise CustomException("Dataset is empty", sys)

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
