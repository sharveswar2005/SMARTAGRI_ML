import os
import sys
import numpy as np

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate regression model
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return rmse, r2

    def initiate_model_training(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        try:
            logging.info("Starting model training")

            # -----------------------------
            # 1️⃣ Bagging Regressor
            # -----------------------------
            bagging_model = BaggingRegressor(
                estimator=DecisionTreeRegressor(),
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            bagging_model.fit(X_train, y_train)
            bagging_pred = bagging_model.predict(X_test)

            bagging_rmse, bagging_r2 = self.evaluate_model(
                y_test, bagging_pred
            )

            logging.info(
                f"BaggingRegressor RMSE: {bagging_rmse}, R2: {bagging_r2}"
            )

            # -----------------------------
            # 2️⃣ Random Forest Regressor
            # -----------------------------
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            rf_rmse, rf_r2 = self.evaluate_model(
                y_test, rf_pred
            )

            logging.info(
                f"RandomForest RMSE: {rf_rmse}, R2: {rf_r2}"
            )

            # -----------------------------
            # 3️⃣ Select Best Model
            # -----------------------------
            if bagging_rmse <= rf_rmse:
                best_model = bagging_model
                best_model_name = "BaggingRegressor"
                best_rmse = bagging_rmse
                best_r2 = bagging_r2
            else:
                best_model = rf_model
                best_model_name = "RandomForestRegressor"
                best_rmse = rf_rmse
                best_r2 = rf_r2

            logging.info(
                f"Best model selected: {best_model_name} "
                f"(RMSE={best_rmse}, R2={best_r2})"
            )

            # Save best model
            os.makedirs("artifacts", exist_ok=True)
            import joblib
            joblib.dump(best_model, self.model_path)

            logging.info("Model training completed successfully")

            return best_model_name, best_rmse, best_r2

        except Exception as e:
            raise CustomException(e, sys)
