import logging
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.xgboost
import os

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_PATH, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("epias-forecast")

    def load_and_process_data(self) -> pd.DataFrame:
        """Fetch consumption + weather data and engineer features."""
        consumption_df = self.data_loader.get_realtime_consumption(
            start_date=pd.Timestamp("2022-01-01"),
            end_date=pd.Timestamp("2026-01-01")
        )
        
        if consumption_df.empty:
            raise ValueError("No consumption data fetched.")

        min_date = consumption_df['date'].min()
        max_date = consumption_df['date'].max()
        
        forecast_df = self.data_loader.get_weather_forecast(
            start_date=min_date,
            end_date=max_date
        )
        
        df_processed = self.feature_engineer.process_data(consumption_df, forecast_df)
        df_model = df_processed.dropna()
        
        return df_model

    def train(self):
        """Train XGBoost model, log to MLflow, and export model.json."""
        df = self.load_and_process_data()

        train = df.loc[df.index < '2026-01-01']
        
        X_train = train[FEATURE_COLUMNS]
        y_train = train[TARGET_COLUMN]
        
        with mlflow.start_run():
            params = {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": 'reg:squarederror'
            }
            mlflow.log_params(params)
            
            model = xgb.XGBRegressor(n_jobs=-1, **params)
            
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Log model to MLflow
            mlflow.xgboost.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # Export model locally for deployment
            model.save_model(MODEL_PATH)
            logger.info(f"Model exported to {MODEL_PATH}")

        logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = Trainer()
    trainer.train()
