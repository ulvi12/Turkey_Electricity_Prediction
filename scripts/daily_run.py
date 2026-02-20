"""Daily automation script: fetches yesterday's data, runs predictions, stores to DB."""

import logging
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import InferencePipeline
from src.data_loader import DataLoader
from src.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting daily run...")
    
    # 1. Setup
    db = Database()
    loader = DataLoader()
    pipeline = InferencePipeline()

    if pipeline.model is None:
        logger.error("No model loaded. Exiting.")
        return

    # 2. Target Date = Yesterday
    target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    logger.info(f"Processing data for: {target_date.date()}")
    
    target_start = target_date
    target_end = target_date + timedelta(hours=23, minutes=59)

    # 3. Fetch Data
    logger.info("Fetching actual consumption...")
    actual_df = loader.get_realtime_consumption(target_start, target_end)
    
    logger.info("Fetching EPIAS forecast...")
    epias_df = loader.get_load_estimation_plan(target_start, target_end)
    
    logger.info("Generating model predictions...")
    pred_df = pipeline.predict(target_date)

    if actual_df.empty:
        logger.warning("No actual consumption data found.")
        return

    # 4. Standardize & Merge
    actual_df = actual_df[['date', 'consumption']].rename(columns={'consumption': 'actual_consumption'})
    
    if not epias_df.empty:
        # Find the forecast value column
        if 'lep' in epias_df.columns:
            epias_df = epias_df[['date', 'lep']].rename(columns={'lep': 'epias_forecast'})
        else:
            cols = [c for c in epias_df.columns if c not in ['date', 'time']]
            if cols:
                epias_df = epias_df[['date', cols[0]]].rename(columns={cols[0]: 'epias_forecast'})
            else:
                epias_df = pd.DataFrame(columns=['date', 'epias_forecast'])
    else:
        epias_df = pd.DataFrame(columns=['date', 'epias_forecast'])

    pred_df = pred_df.rename(columns={'prediction': 'model_prediction'})

    merged_df = pd.merge(actual_df, epias_df, on='date', how='outer')
    merged_df = pd.merge(merged_df, pred_df, on='date', how='outer')

    # 5. Store in DB
    logger.info("Saving to database...")
    count = 0
    for _, row in merged_df.iterrows():
        ts = row['date']
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
            
        act = row['actual_consumption'] if pd.notnull(row.get('actual_consumption')) else None
        epi = row['epias_forecast'] if pd.notnull(row.get('epias_forecast')) else None
        mod = row['model_prediction'] if pd.notnull(row.get('model_prediction')) else None
        
        db.upsert_monitoring_data(ts, act, epi, mod)
        count += 1
    logger.info(f"Saved {count} records.")

    # 6. Performance Check
    valid_data = merged_df.dropna()
    
    if not valid_data.empty:
        mae_model = mean_absolute_error(valid_data['actual_consumption'], valid_data['model_prediction'])
        mae_epias = mean_absolute_error(valid_data['actual_consumption'], valid_data['epias_forecast'])
        
        logger.info(f"Model MAE: {mae_model:.2f} | EPIAS MAE: {mae_epias:.2f}")
        
        if mae_model > 2 * mae_epias and mae_epias > 0:
            logger.warning(f"Model performance alert! Model MAE ({mae_model:.2f}) is > 2x worse than EPIAS ({mae_epias:.2f})")
    else:
        logger.warning("Insufficient data for MAE comparison.")

    logger.info("Daily run completed.")


if __name__ == "__main__":
    main()
