import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts", "preprocessor.pkl"
        )
        # Actual yield column in your dataset
        self.yield_column = "hg/ha_yield"

    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert yield column into Risk Score (0–100)
        """
        try:
            logging.info("Creating risk score from yield")

            yield_min = df[self.yield_column].min()
            yield_max = df[self.yield_column].max()

            df["Yield_Normalized"] = (
                (df[self.yield_column] - yield_min)
                / (yield_max - yield_min)
            ) * 100

            # Higher yield → lower risk
            df["Risk_Score"] = 100 - df["Yield_Normalized"]

            logging.info("Risk score created successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self, df: pd.DataFrame):
        """
        Create preprocessing pipeline
        """
        try:
            logging.info("Creating preprocessing pipeline")

            categorical_cols = df.select_dtypes(include=["object"]).columns

            numerical_cols = (
                df.select_dtypes(exclude=["object"])
                .drop(
                    ["Risk_Score", "Yield_Normalized", self.yield_column],
                    axis=1
                )
                .columns
            )

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )

            logging.info("Preprocessing pipeline created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Remove unwanted index column
            if "Unnamed: 0" in train_df.columns:
                train_df.drop(columns=["Unnamed: 0"], inplace=True)
                test_df.drop(columns=["Unnamed: 0"], inplace=True)

            # Create risk score
            train_df = self.create_risk_score(train_df)
            test_df = self.create_risk_score(test_df)

            # Separate features and target
            X_train = train_df.drop(
                ["Risk_Score", "Yield_Normalized", self.yield_column],
                axis=1
            )
            y_train = train_df["Risk_Score"]

            X_test = test_df.drop(
                ["Risk_Score", "Yield_Normalized", self.yield_column],
                axis=1
            )
            y_test = test_df["Risk_Score"]

            # Get preprocessor
            preprocessor = self.get_preprocessor(train_df)

            # Fit and transform
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Save preprocessor
            os.makedirs("artifacts", exist_ok=True)
            pd.to_pickle(preprocessor, self.preprocessor_path)

            logging.info("Data transformation completed successfully")

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
