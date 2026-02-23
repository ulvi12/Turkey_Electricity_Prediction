# Türkiye's Electricity Consumption Forecasting

An end-to-end ML pipeline that forecasts Turkey's hourly electricity consumption one day ahead, updated automatically every day.

**[Live Dashboard](https://turkeyelectricityprediction.streamlit.app/)**

---

## What it does

Turkey's energy market [EPIAS](https://www.epias.com.tr/en/) publishes hourly electricity consumption data for Türkiye. They also publish their official forecasts for the hourly consumption. This project builds an XGBoost model that competes with that official forecast, using the same publicly available inputs: historical consumption and weather data.

> The day-ahead market requires participants to submit their consumption/generation plans by 12:00 the previous day. To create the same realistic setting, our forecasts are made one full day prior to the actual realizations. See [EPIAS Day-Ahead Market](https://www.epias.com.tr/en/day-ahead-market/processes/) for details.

Each day, a GitHub Actions job:
1. Fetches actual consumption from EPIAS API
2. Fetches the official EPIAS forecast
3. Runs the XGBoost model
4. Writes all three to Supabase
5. The Streamlit dashboard shows the comparison in real time

---

## Stack

| Layer | Tech |
|-------|------|
| ML | XGBoost, Scikit-learn, Pandas |
| Data Sources | EPIAS (consumption + forecast), Open-Meteo (weather) |
| Database | Supabase (PostgreSQL) via SQLAlchemy |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Docker |
| Automation | GitHub Actions (daily cron) |

---

## Local setup - Windows

```bash
git clone https://github.com/ulvi12/Turkey_Electricity_Prediction.git
cd Turkey_Electricity_Prediction
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # fill in EPIAS credentials + Supabase URL

streamlit run dashboard/app.py
```

**FastAPI (Docker):**
```bash
docker-compose up --build
# http://localhost:8000/docs
```

**Get predictions for a specific date:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-02-15"}'
```

> Omit the `date` field to predict for yesterday. Dates in the future, starting from today, are rejected because of unavailable data.

---

## Deployment

- **Dashboard:** Streamlit Cloud — set `SUPABASE_DB_URL` as a secret
- **Daily job:** GitHub Actions — set `EPIAS_USERNAME`, `EPIAS_PASSWORD`, `SUPABASE_DB_URL` as repo secrets
