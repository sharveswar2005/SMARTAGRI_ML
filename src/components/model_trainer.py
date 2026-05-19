import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.config = read_yaml("config.yaml")

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
            logging.info("Starting model training with Hyperparameter tuning")

            models = {
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(random_state=42)
            }

            model_params = self.config['model_params']
            cv_folds = self.config['ml_settings']['cv_folds']
            n_iter = self.config['ml_settings']['n_iter_search']

            best_model_name = None
            best_model_score = -float("inf")
            best_model_instance = None
            best_rmse = float("inf")

            for name, model in models.items():
                logging.info(f"Training {name}")
                params = model_params.get(name, {})
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='r2',
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)

                best_estimator = search.best_estimator_
                y_pred = best_estimator.predict(X_test)

                rmse, r2 = self.evaluate_model(y_test, y_pred)
                logging.info(f"{name} Best Params: {search.best_params_} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_rmse = rmse
                    best_model_name = name
                    best_model_instance = best_estimator

            if best_model_score < 0.5:
                logging.warning("Best model has an R2 score less than 0.5")

            logging.info(f"Overall Best Model: {best_model_name} (RMSE={best_rmse:.4f}, R2={best_model_score:.4f})")

            os.makedirs("artifacts", exist_ok=True)
            import joblib
            joblib.dump(best_model_instance, self.model_path)

            logging.info("Model training completed successfully and saved best model.")

            return best_model_name, best_rmse, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
