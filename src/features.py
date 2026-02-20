import pandas as pd
import holidays

class FeatureEngineer:
    def __init__(self):
        pass

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'date' in df.columns:
             df = df.set_index('date')

        df['hour'] = df.index.hour
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

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['lag_48'] = df['consumption'].shift(48)
        df['lag_72'] = df['consumption'].shift(72)
        df['lag_168'] = df['consumption'].shift(168)
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Window 24 hours of two days prior
        df['roll_mean_1d'] = df['lag_48'].rolling(window=24).mean()
        df['roll_std_1d'] = df['lag_48'].rolling(window=24).std()
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
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.merge_weather(df, forecast_df)
        
        return df
