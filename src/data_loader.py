import os
import logging
import requests
import pandas as pd
import calendar
from datetime import timedelta
from src.config import EPIAS_USERNAME, EPIAS_PASSWORD

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.username = EPIAS_USERNAME
        self.password = EPIAS_PASSWORD
        self.tgt = None
        self.headers = {'Content-Type': 'application/json'}

    def _get_tgt(self):
        """Authenticate with EPIAS and obtain a TGT token."""
        url = "https://giris.epias.com.tr/cas/v1/tickets"
        response = requests.post(url, data={'username': self.username, 'password': self.password}, timeout=30)
        
        if response.status_code == 201:
            self.tgt = response.headers['Location'].split('/')[-1]
            self.headers['TGT'] = self.tgt
        else:
            raise Exception(f"Failed to get TGT: {response.status_code}, {response.text}")

    def _fetch_monthly(self, url: str, start_date, end_date, label: str) -> pd.DataFrame:
        """Fetch data from an EPIAS endpoint month by month to avoid timeouts."""
        if not self.tgt:
            self._get_tgt()

        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            last_day = calendar.monthrange(year, month)[1]
            
            month_end = pd.Timestamp(year, month, last_day)
            if month_end > end_date:
                month_end = end_date
                
            start_str = f"{year}-{month:02d}-{current_date.day:02d}T00:00:00+03:00"
            end_str = f"{year}-{month:02d}-{month_end.day:02d}T23:59:59+03:00"

            logger.info(f"Fetching {label} for {year}-{month:02d}...")
            
            try:
                resp = requests.post(url, headers=self.headers, json={"startDate": start_str, "endDate": end_str})
                
                if resp.status_code == 200:
                    items = resp.json().get('items', [])
                    all_data.extend(items)
                    logger.info(f"  OK. ({len(items)} records)")
                else:
                    logger.warning(f"  Error: {resp.status_code}")
            except Exception as e:
                logger.error(f"  Exception: {e}")

            current_date = month_end + timedelta(days=1)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert("Europe/Istanbul")
        return df

    def get_realtime_consumption(self, start_date, end_date) -> pd.DataFrame:
        """Fetch hourly real-time consumption data from EPIAS."""
        url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"
        return self._fetch_monthly(url, start_date, end_date, label="consumption")

    def get_load_estimation_plan(self, start_date, end_date) -> pd.DataFrame:
        """Fetch EPIAS load estimation plan (their official forecast)."""
        url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/load-estimation-plan"
        return self._fetch_monthly(url, start_date, end_date, label="load estimation plan")

    def get_weather_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Fetch historical weather forecast data from Open-Meteo."""
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching weather data from {start_date_str} to {end_date_str}...")

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 39.0,
            "longitude": 35.0,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m",
            "models": "gfs_seamless",
            "timezone": "Europe/Istanbul"
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            hourly_data = data.get("hourly", {})
            
            forecast_df = pd.DataFrame({
                "date": hourly_data.get("time", []),
                "forecast_temp": hourly_data.get("temperature_2m", [])
            })
            
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            
            if forecast_df["date"].dt.tz is None:
                 forecast_df["date"] = forecast_df["date"].dt.tz_localize("Europe/Istanbul", ambiguous='NaT', nonexistent='shift_forward')
            else:
                 forecast_df["date"] = forecast_df["date"].dt.tz_convert("Europe/Istanbul")

            return forecast_df
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    print("DataLoader initialized.")
