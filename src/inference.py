import logging
import os
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.config import FEATURE_COLUMNS, MODEL_PATH

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, model_path: str = None):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
        model_path = model_path or MODEL_PATH
        
        # Try loading local model file (model.json)
        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            # Fallback: try MLflow
            try:
                import mlflow
                import mlflow.xgboost
                from src.config import MLFLOW_TRACKING_URI
                
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/epias-forecast/Production")
                self.model = mlflow.xgboost.load_model(model_uri)
                logger.info(f"Model loaded from MLflow: {model_uri}")
            except Exception as e:
                logger.error(f"Could not load model: {e}")

    def predict(self, target_date: datetime) -> pd.DataFrame:
        """Generate hourly predictions for a target date."""
        if self.model is None:
            raise RuntimeError("No model loaded. Cannot make predictions.")
        
        # Fetch 10 days of history for lag features
        history_start_date = target_date - timedelta(days=10)
        
        consumption_df = self.data_loader.get_realtime_consumption(
            start_date=history_start_date,
            end_date=target_date
        )
        
        if consumption_df.empty:
            raise ValueError("No historical consumption data found.")

        # Create target hours for prediction
        target_hours = pd.date_range(
            start=target_date.replace(hour=0, minute=0, second=0, microsecond=0),
            end=target_date.replace(hour=23, minute=0, second=0, microsecond=0),
            freq='H',
            tz='Europe/Istanbul'
        )
        
        target_df = pd.DataFrame({'date': target_hours})
        target_df['consumption'] = None 
        
        full_df = pd.concat([consumption_df, target_df], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=['date'], keep='first')
        full_df = full_df.set_index('date').sort_index()
        
        # Weather data
        forecast_df = self.data_loader.get_weather_forecast(
            start_date=history_start_date,
            end_date=target_date + timedelta(days=-1)
        )
        
        # Feature engineering
        df_processed = self.feature_engineer.process_data(full_df, forecast_df)
        
        target_rows = df_processed.loc[df_processed.index.date == target_date.date()]
        X_target = target_rows[FEATURE_COLUMNS]
        
        predictions = self.model.predict(X_target)
        
        results = pd.DataFrame({
            'date': target_rows.index,
            'prediction': predictions
        })
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = InferencePipeline()
    target_date = datetime.now() + timedelta(days=-1)
    try:
        preds = pipeline.predict(target_date)
        print(preds)
    except Exception as e:
        print(f"Inference failed: {e}")
