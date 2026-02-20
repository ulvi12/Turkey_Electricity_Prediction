import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys, os

# project root to path so Streamlit can find the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import Database

st.set_page_config(page_title="Türkiye's Electricity Consumption Prediction", layout="wide")

st.title("Türkiye's Electricity Consumption Prediction")
st.caption("Comparing our XGBoost model predictions vs. EPIAS official forecast against actual consumption.")


@st.cache_resource
def get_db():
    return Database()


db = get_db()
data = db.get_monitoring_data()

if not data:
    st.warning("No data found in monitoring database.")
else:
    records = []
    for r in data:
        records.append({
            'date': r.date,
            'Actual': r.actual_consumption,
            'EPIAS Forecast': r.epias_forecast,
            'Model Prediction': r.model_prediction
        })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    tab1, tab2 = st.tabs(["Daily View", "Cumulative View"])

    with tab1:
        st.header("Daily Performance")
        
        available_dates = df['date'].dt.date.unique()
        if len(available_dates) > 0:
            default_date = available_dates[-2] if len(available_dates) > 1 else available_dates[-1]
        else:
            default_date = datetime.now().date() - timedelta(days=1)
            
        min_allowed = datetime(2026, 2, 15).date()
        selected_date = st.date_input("Select Date", max(default_date, min_allowed), min_value=min_allowed)
        
        daily_mask = (df['date'].dt.date == selected_date)
        daily_df = df.loc[daily_mask]
        
        if not daily_df.empty:
            d_col1, d_col2, d_col3 = st.columns(3)
            
            valid_daily = daily_df.dropna()
            
            if not valid_daily.empty:
                d_mae_model = (valid_daily['Actual'] - valid_daily['Model Prediction']).abs().mean()
                d_mae_epias = (valid_daily['Actual'] - valid_daily['EPIAS Forecast']).abs().mean()
                
                d_mape_model = ((valid_daily['Actual'] - valid_daily['Model Prediction']).abs() / valid_daily['Actual']).mean() * 100
                d_mape_epias = ((valid_daily['Actual'] - valid_daily['EPIAS Forecast']).abs() / valid_daily['Actual']).mean() * 100
                
                d_rmse_model = ((valid_daily['Actual'] - valid_daily['Model Prediction']) ** 2).mean() ** 0.5
                d_rmse_epias = ((valid_daily['Actual'] - valid_daily['EPIAS Forecast']) ** 2).mean() ** 0.5

                d_col1.metric("Model MAE", f"{d_mae_model:.2f}", delta=f"{(d_mae_model-d_mae_epias):.2f} vs EPIAS Forecast", delta_color="inverse")
                d_col2.metric("Model MAPE", f"{d_mape_model:.2f}%", delta=f"{(d_mape_model-d_mape_epias):.2f}% vs EPIAS Forecast", delta_color="inverse")
                d_col3.metric("Model RMSE", f"{d_rmse_model:.2f}", delta=f"{(d_rmse_model-d_rmse_epias):.2f} vs EPIAS Forecast", delta_color="inverse")
                
                d_col1.info(f"EPIAS Forecast MAE: {d_mae_epias:.2f}")
                d_col2.info(f"EPIAS Forecast MAPE: {d_mape_epias:.2f}%")
                d_col3.info(f"EPIAS Forecast RMSE: {d_rmse_epias:.2f}")

            else:
                st.warning("Incomplete data for this date.")

            # Daily time series plot
            st.subheader("Hourly Comparison")
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['Actual'], name='Actual', line=dict(color='#2ecc71', width=2)))
            fig_daily.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['EPIAS Forecast'], name='EPIAS Forecast', line=dict(color='#3498db', dash='dash')))
            fig_daily.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['Model Prediction'], name='Model Prediction', line=dict(color='#e74c3c')))
            fig_daily.update_layout(xaxis_title="Hour", yaxis_title="MWh", template="plotly_white")
            st.plotly_chart(fig_daily, use_container_width=True)
            
            with st.expander("Show Raw Data"):
                st.dataframe(daily_df, use_container_width=True)
        else:
            st.info(f"No data available for {selected_date}")

    with tab2:
        st.header("Cumulative Performance")
        
        min_allowed = datetime(2026, 2, 15).date()
        min_date = max(df['date'].min().date(), min_allowed)
        max_date = df['date'].max().date()
        
        date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_allowed, key='cum_range')
        
        if len(date_range) == 2:
            start_d, end_d = date_range
            mask = (df['date'].dt.date >= start_d) & (df['date'].dt.date <= end_d)
            filtered_df = df.loc[mask]
        else:
            filtered_df = df
            
        valid_cum = filtered_df.dropna()
        
        if not valid_cum.empty:
            c_col1, c_col2, c_col3 = st.columns(3)
            
            c_mae_model = (valid_cum['Actual'] - valid_cum['Model Prediction']).abs().mean()
            c_mae_epias = (valid_cum['Actual'] - valid_cum['EPIAS Forecast']).abs().mean()
            c_mape_epias = ((valid_cum['Actual'] - valid_cum['EPIAS Forecast']).abs() / valid_cum['Actual']).mean() * 100
            c_mape_model = ((valid_cum['Actual'] - valid_cum['Model Prediction']).abs() / valid_cum['Actual']).mean() * 100
            c_rmse_epias = ((valid_cum['Actual'] - valid_cum['EPIAS Forecast']) ** 2).mean() ** 0.5
            c_rmse_model = ((valid_cum['Actual'] - valid_cum['Model Prediction']) ** 2).mean() ** 0.5
            
            c_col1.metric("Avg MAE", f"{c_mae_model:.2f}", delta=f"{(c_mae_model-c_mae_epias):.2f} vs EPIAS Forecast", delta_color="inverse")
            c_col2.metric("Avg MAPE", f"{c_mape_model:.2f}%", delta=f"{(c_mape_model-c_mape_epias):.2f}% vs EPIAS Forecast", delta_color="inverse")
            c_col3.metric("Avg RMSE", f"{c_rmse_model:.2f}", delta=f"{(c_rmse_model-c_rmse_epias):.2f} vs EPIAS Forecast", delta_color="inverse")
            
            c_col1.info(f"EPIAS Forecast MAE: {c_mae_epias:.2f}")
            c_col2.info(f"EPIAS Forecast MAPE: {c_mape_epias:.2f}%")
            c_col3.info(f"EPIAS Forecast RMSE: {c_rmse_epias:.2f}")
            
            # Overall time series
            st.subheader("Time Series Overview")
            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(x=valid_cum['date'], y=valid_cum['Actual'], name='Actual', line=dict(color='#2ecc71', width=1)))
            fig_all.add_trace(go.Scatter(x=valid_cum['date'], y=valid_cum['Model Prediction'], name='Model Prediction', line=dict(color='#e74c3c', width=1)))
            fig_all.add_trace(go.Scatter(x=valid_cum['date'], y=valid_cum['EPIAS Forecast'], name='EPIAS Forecast', line=dict(color='#3498db', dash='dash', width=1)))
            fig_all.update_layout(xaxis_title="Date", yaxis_title="MWh", template="plotly_white")
            st.plotly_chart(fig_all, use_container_width=True)
        else:
            st.info("No valid data for selected range.")
