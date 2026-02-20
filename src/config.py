"""Shared configuration constants for the EPIAS forecast pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Feature Engineering ---
FEATURE_COLUMNS = [
    'hour', 'dayofweek', 'month', 'year', 'is_holiday',
    'forecast_temp', 'lag_48', 'lag_72', 'lag_168',
    'roll_mean_1d', 'roll_std_1d'
]
TARGET_COLUMN = 'consumption'

# --- Model ---
MODEL_PATH = os.getenv("MODEL_PATH", "model.json")

# --- Database ---
DATABASE_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///monitoring.db")

# --- API ---
EPIAS_USERNAME = os.getenv("EPIAS_USERNAME")
EPIAS_PASSWORD = os.getenv("EPIAS_PASSWORD")

# --- MLflow (optional) ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
