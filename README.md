# 🌫️ Karachi AQI Forecasting System

> **Real-time 3-day Air Quality Index forecasting for Karachi, Pakistan**
> Built with OpenWeather API · Supabase · Scikit-learn · TensorFlow · Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=flat-square&logo=supabase)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Pipeline Details](#-pipeline-details)
- [ML Models](#-ml-models)
- [Dashboard Pages](#-dashboard-pages)
- [EDA Notebook](#-eda-notebook)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

The **Karachi AQI Forecasting System** is an end-to-end machine learning pipeline that:

1. **Collects** hourly air quality and weather data from OpenWeather API
2. **Stores** all features in a Supabase (PostgreSQL) database
3. **Trains** three ML models (Ridge Regression, Random Forest, LSTM) and picks the best
4. **Forecasts** AQI for the next 72 hours using recursive prediction
5. **Visualizes** everything in a beautiful dark-themed Streamlit dashboard

> Karachi consistently ranks among the world's most polluted cities. This project aims to provide actionable, data-driven air quality forecasts to help residents make informed health decisions.

---

## 🚀 Live Demo

```
streamlit run app/streamlit_app.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│                                                             │
│   OpenWeather Air Pollution API  +  Weather API            │
│        (Real-time & Historical)                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE PIPELINES                          │
│                                                             │
│   backfill_pipeline.py   →   30-90 days historical data    │
│   feature_pipeline.py    →   Hourly live data (scheduled)  │
│                                                             │
│   Features engineered:                                      │
│   • EPA AQI from PM2.5  • Dew point (Magnus formula)       │
│   • Lag features (1h, 2h, 3h)  • Rolling avg (3h, 7h)     │
│   • Time features (hour, day, rush hour, weekend)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               SUPABASE (PostgreSQL)                         │
│                                                             │
│   Table: aqi_features  (~720 rows per 30 days)             │
│   29 columns · Upsert on timestamp conflict                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               TRAINING PIPELINE                             │
│                                                             │
│   Ridge Regression  →  Baseline linear model               │
│   Random Forest     →  Tree ensemble (max_depth=10)        │
│   LSTM              →  Sequential neural network           │
│                                                             │
│   Best model selected by RMSE → saved to models/           │
│   SHAP explainability computed for Random Forest            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               INFERENCE PIPELINE                            │
│                                                             │
│   Recursive 72-hour forecast                               │
│   • Real lag features from OpenWeather history API         │
│   • 10% per-hour drift cap to prevent runaway predictions  │
│   • AQI capped at 0–500                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               STREAMLIT DASHBOARD                           │
│                                                             │
│   📊 Live Dashboard  │  📈 EDA & Trends                    │
│   🔍 Model Insights  │  ⚠️  AQI Health Guide               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
aqi-forecasting/
│
├── pipelines/
│   ├── utils.py                # Shared utilities
│   │   ├── pm25_to_aqi()       # EPA AQI formula
│   │   ├── calc_dew_point()    # Magnus formula
│   │   ├── categorize_aqi_*()  # Category helpers
│   │   └── fetch_recent_aqi_history()
│   │
│   ├── backfill_pipeline.py    # Historical data collection
│   ├── feature_pipeline.py     # Hourly live data ingestion
│   ├── training_pipeline.py    # Model training & selection
│   └── inference_pipeline.py   # 72-hour forecast generation
│
├── app/
│   └── streamlit_app.py        # Main dashboard
│
├── models/                     # Auto-generated after training
│   ├── best_model.pkl          # Best sklearn model (or .keras)
│   ├── best_model_name.json    # Which model won
│   ├── scaler.pkl              # StandardScaler
│   ├── feature_names.json      # Feature order
│   ├── metrics.json            # RMSE, MAE, R²
│   └── shap_importance.csv     # Feature importance
│
├── notebooks/
│   └── karachi_aqi_eda.ipynb   # Google Colab EDA notebook
│
├── data/
│   └── aqi_features_rows.csv   # Optional CSV fallback
│
├── .env                        # API keys (not committed)
├── .env.example                # Template
├── requirements.txt
├── run.bat                     # Windows one-click runner
└── README.md
```

---

## ✨ Features

### Data Pipeline
- ✅ **EPA-standard AQI** calculation from PM2.5 using official breakpoints
- ✅ **Real dew point** using Magnus formula (not feels_like)
- ✅ **Real lag features** from OpenWeather history API (not fake duplicates)
- ✅ **Upsert logic** — safe to re-run, no duplicate rows
- ✅ **Outlier capping** — PM10 capped at 400 µg/m³

### ML Models
- ✅ **Three models trained** and compared automatically
- ✅ **Best model auto-selected** by RMSE
- ✅ **SHAP explainability** for feature importance
- ✅ **Drift prevention** — max 10% AQI change per forecast hour

### Dashboard
- ✅ **Live AQI** with color-coded health alerts
- ✅ **3-day forecast cards** with min/max range
- ✅ **72-hour hourly chart** with AQI zone bands
- ✅ **EDA page** with 10+ interactive charts
- ✅ **SHAP feature importance** visualization
- ✅ **AQI health guide** with recommendations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Source | OpenWeather Air Pollution API + Weather API |
| Database | Supabase (PostgreSQL) via REST API |
| Feature Engineering | Pandas, NumPy |
| ML Models | Scikit-learn (Ridge, Random Forest), TensorFlow (LSTM) |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| EDA | Jupyter / Google Colab, Plotly, Seaborn, SciPy |
| Environment | Python 3.10+, python-dotenv |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/aqi-forecasting.git
cd aqi-forecasting
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Create Supabase table
Run this SQL in your Supabase SQL Editor:
```sql
CREATE TABLE aqi_features (
    timestamp       TIMESTAMPTZ PRIMARY KEY,
    aqi             INTEGER,
    pm25            FLOAT,  pm10     FLOAT,
    o3              FLOAT,  no2      FLOAT,
    so2             FLOAT,  co       FLOAT,
    temp            FLOAT,  humidity FLOAT,
    pressure        FLOAT,  wind     FLOAT,
    dew             FLOAT,  visibility FLOAT,
    cloud_cover     FLOAT,  weather_main TEXT,
    hour            INTEGER, day     INTEGER,
    month           INTEGER, day_of_week INTEGER,
    is_weekend      INTEGER, is_rush_hour INTEGER,
    aqi_category    INTEGER,
    aqi_lag_1       FLOAT,  aqi_lag_2    FLOAT,
    aqi_lag_3       FLOAT,  aqi_rolling_3 FLOAT,
    aqi_rolling_7   FLOAT,  aqi_change_rate FLOAT
);
```

### 6. Run the full pipeline
```bash
# Collect 30 days of historical data
python pipelines/backfill_pipeline.py

# Train models
python pipelines/training_pipeline.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 🔧 Pipeline Details

### `backfill_pipeline.py`
Collects historical air quality and weather data.

```bash
# Default: 30 days
python pipelines/backfill_pipeline.py

# Custom: change at bottom of file
run_backfill(days_back=90)  # 3 months for better training
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `days_back` | 30 | How many days of history to fetch |
| Batch size | 100 rows | Rows per Supabase upsert |
| Rate | ~1 req/sec | OpenWeather API calls |

### `feature_pipeline.py`
Runs hourly to keep the database current. Schedule with cron or GitHub Actions:

```bash
# Run manually
python pipelines/feature_pipeline.py

# Cron (every hour)
0 * * * * cd /path/to/project && python pipelines/feature_pipeline.py
```

### `training_pipeline.py`
Trains all three models and saves the best one.

```bash
python pipelines/training_pipeline.py
```

Output:
```
📊 Train: 580 | Test: 145
🔵 Ridge Regression  →  RMSE: 18.4 | MAE: 12.1 | R²: 0.8821
🌲 Random Forest     →  RMSE: 11.2 | MAE:  7.3 | R²: 0.9312
🧠 LSTM              →  RMSE: 14.6 | MAE:  9.8 | R²: 0.9071
🏆 Best model: Random Forest (RMSE: 11.2)
```

### `inference_pipeline.py`
Generates 72-hour recursive forecast.

```bash
python pipelines/inference_pipeline.py
```

---

## 🤖 ML Models

### Features Used (24 total)

| Category | Features |
|----------|----------|
| Pollutants | pm25, pm10, o3, no2, so2, co |
| Weather | temp, humidity, pressure, wind, dew, cloud_cover |
| Time | hour, day, month, day_of_week, is_weekend, is_rush_hour |
| Lag | aqi_lag_1, aqi_lag_2, aqi_lag_3 |
| Rolling | aqi_rolling_3, aqi_rolling_7, aqi_change_rate |

### AQI Calculation

AQI is computed from PM2.5 using the **US EPA formula**:

```
AQI = ((I_hi - I_lo) / (C_hi - C_lo)) × (C - C_lo) + I_lo
```

| PM2.5 (µg/m³) | AQI Range | Category |
|----------------|-----------|----------|
| 0.0 – 12.0 | 0 – 50 | Good |
| 12.1 – 35.4 | 51 – 100 | Moderate |
| 35.5 – 55.4 | 101 – 150 | Unhealthy for Sensitive Groups |
| 55.5 – 150.4 | 151 – 200 | Unhealthy |
| 150.5 – 250.4 | 201 – 300 | Very Unhealthy |
| 250.5+ | 301 – 500 | Hazardous |

### LSTM Architecture

```
Input (1, 24 features)
    └── LSTM(64, return_sequences=True)
    └── Dropout(0.2)
    └── LSTM(32)
    └── Dropout(0.2)
    └── Dense(16, relu)
    └── Dense(1)  →  Predicted AQI
```

---

## 📊 Dashboard Pages

### 📊 Live Dashboard
- Real-time AQI with color-coded health alert banner
- Current pollutant readings (PM2.5, PM10, O₃, NO₂)
- Current weather (temp, humidity, wind, pressure)
- 3-day forecast cards with avg/min/max
- 72-hour interactive forecast chart with AQI zone bands
- Model performance metrics

### 📈 EDA & Trends
- AQI time series with 24h & 7-day rolling averages
- AQI distribution histogram + KDE
- Monthly average bar chart
- Hourly pattern analysis
- Day of week heatmap
- Pollutant correlation chart
- AQI category breakdown (donut + bar)

### 🔍 Model Insights
- Best model RMSE, MAE, R² metrics
- SHAP feature importance bar chart
- Feature importance table
- How the model works (explained simply)

### ⚠️ AQI Health Guide
- Full AQI scale explanation
- Health recommendations per category
- Who is most at risk (children, elderly, asthma patients etc.)

---

## 📓 EDA Notebook

A full Google Colab notebook is included at `notebooks/karachi_aqi_eda.ipynb`.

**Open in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Sections:**
1. Setup & Install
2. Load data from Supabase
3. Dataset overview & statistics
4. AQI time series with rolling averages
5. AQI distribution (histogram + KDE + box plots)
6. Temporal patterns (hourly heatmap, day of week)
7. Pollutant analysis & correlation heatmap
8. Weather vs AQI (including 3D scatter)
9. AQI category breakdown
10. Daily summary with min/max range
11. Lag feature validation & autocorrelation
12. Final summary report

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# OpenWeather API
OPENWEATHER_API_KEY=your_openweather_api_key

# Supabase
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_service_role_key
```

### Getting API Keys

| Service | URL | Notes |
|---------|-----|-------|
| OpenWeather | [openweathermap.org/api](https://openweathermap.org/api) | Free tier: 60 calls/min |
| Supabase | [supabase.com](https://supabase.com) | Free tier: 500MB database |

---

## 📦 Requirements

```txt
requests
pandas
numpy
scikit-learn
tensorflow
shap
streamlit
plotly
supabase
python-dotenv
scipy
seaborn
joblib
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [ ] GitHub Actions for hourly automated data collection
- [ ] Multi-city support (Lahore, Islamabad, Peshawar)
- [ ] SMS/email alerts when AQI exceeds threshold
- [ ] Prophet model for seasonal forecasting
- [ ] Docker containerization
- [ ] Deploy dashboard to Streamlit Cloud

---

## 📍 Location

| Parameter | Value |
|-----------|-------|
| City | Karachi, Pakistan |
| Latitude | 24.8607° N |
| Longitude | 67.0011° E |
| Timezone | PKT (UTC+5) |

---

## 👨‍💻 Author

Built by **Talha** — Computer Science Student, Karachi

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <b>🌫️ Breathe safe. Stay informed.</b><br>
  <sub>Data refreshed every hour · Forecasts updated on demand</sub>
</div>
