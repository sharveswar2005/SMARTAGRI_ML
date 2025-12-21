import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging


class DataValidation:
    def __init__(self):
        # Actual target column in your dataset
        self.target_column = "hg/ha_yield"

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Starting data validation")

            # 1️⃣ Check if dataset is empty
            if df.empty:
                raise CustomException("Dataset is empty", sys)

            # 2️⃣ Check if target column exists
            if self.target_column not in df.columns:
                raise CustomException(
                    f"Target column '{self.target_column}' not found in dataset",
                    sys
                )

            # 3️⃣ Check minimum number of columns
            if df.shape[1] < 4:
                raise CustomException(
                    "Dataset has insufficient columns for training",
                    sys
                )

            logging.info("Data validation completed successfully")
            return True

        except Exception as e:
            raise CustomException(e, sys)
