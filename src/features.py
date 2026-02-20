import pandas as pd
import holidays

RAMADAN_DATES = [
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-29"),
    ("2026-02-18", "2026-03-19"),
    ("2027-02-08", "2027-03-08"),
]

KURBAN_DATES = [
    ("2022-07-09", "2022-07-12"),
    ("2023-06-28", "2023-07-01"),
    ("2024-06-17", "2024-06-20"),
    ("2025-06-06", "2025-06-09"),
    ("2026-05-26", "2026-05-29"),
    ("2027-05-16", "2027-05-19"),
]

ramadan_set = set()
for start, end in RAMADAN_DATES:
    for d in pd.date_range(start, end):
        ramadan_set.add(d.date())

kurban_set = set()
for start, end in KURBAN_DATES:
    for d in pd.date_range(start, end):
        kurban_set.add(d.date())


class FeatureEngineer:
    def __init__(self):
        pass

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'date' in df.columns:
            df = df.set_index('date')

        df['hour'] = df.index.hour
        df['dayofyear'] = df.index.dayofyear
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        return df

    def add_holiday_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        unique_dates = pd.to_datetime(df.index.date).unique()
        
        years = df.index.year.unique()
        tr_holidays = holidays.Turkey(years=years)
        holiday_dates = set(tr_holidays.keys())
        
        df['is_holiday'] = df.index.map(lambda x: x.date() in holiday_dates).astype(int)
        return df

    def add_islamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dates = df.index.date
        df['is_ramadan'] = [1 if d in ramadan_set else 0 for d in dates]
        df['is_kurban'] = [1 if d in kurban_set else 0 for d in dates]
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['lag_48'] = df['consumption'].shift(48)
        df['lag_72'] = df['consumption'].shift(72)
        df['lag_168'] = df['consumption'].shift(168)
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['roll_mean_1d'] = df['lag_48'].rolling(window=24).mean()
        df['roll_std_1d'] = df['lag_48'].rolling(window=24).std()
        df['roll_mean_1w'] = df['lag_48'].rolling(window=168).mean()
        df['roll_std_1w'] = df['lag_48'].rolling(window=168).std()
        return df

    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'forecast_temp' in df.columns:
            df['temp_squared'] = df['forecast_temp'] ** 2
        return df

    def merge_weather(self, df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # If date is index, reset it so we can merge on 'date' column
        if df.index.name == 'date':
            df = df.reset_index()
        merged_df = pd.merge(df, forecast_df, on='date', how='left')
        
        # Restore index
        merged_df = merged_df.set_index('date')
        
        return merged_df

    def process_data(self, df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = self.add_temporal_features(df)
        df = self.add_holiday_feature(df)
        df = self.add_islamic_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.merge_weather(df, forecast_df)
        df = self.add_weather_features(df)
        return df
