import os
from dotenv import load_dotenv

load_dotenv()

FEATURE_COLUMNS = [
    'hour', 'dayofweek', 'dayofyear', 'month', 'quarter', 'year',
    'is_holiday', 'is_ramadan', 'is_kurban',
    'forecast_temp', 'temp_squared',
    'lag_48', 'lag_72', 'lag_168',
    'roll_mean_1d', 'roll_std_1d', 'roll_mean_1w', 'roll_std_1w'
]
TARGET_COLUMN = 'consumption'

MODEL_PATH = os.getenv("MODEL_PATH", "model.json")
DATABASE_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///monitoring.db")
EPIAS_USERNAME = os.getenv("EPIAS_USERNAME")
EPIAS_PASSWORD = os.getenv("EPIAS_PASSWORD")
