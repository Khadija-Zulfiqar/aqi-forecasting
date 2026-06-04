# pipelines/feature_pipeline.py
# Fetches AQI + weather data, engineers features, stores in Hopsworks

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
AQICN_TOKEN        = os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY    = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")
CITY               = "karachi"
LAT, LON           = 24.8607, 67.0011

# ─────────────────────────────────────────
# 1. FETCH AQI DATA FROM AQICN
# ─────────────────────────────────────────
def fetch_aqi_data():
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        raise ValueError(f"AQICN API error: {data}")

    d = data["data"]
    iaqi = d.get("iaqi", {})

    return {
        "aqi":       d["aqi"],
        "pm25":      iaqi.get("pm25", {}).get("v", np.nan),
        "pm10":      iaqi.get("pm10", {}).get("v", np.nan),
        "o3":        iaqi.get("o3",   {}).get("v", np.nan),
        "no2":       iaqi.get("no2",  {}).get("v", np.nan),
        "so2":       iaqi.get("so2",  {}).get("v", np.nan),
        "co":        iaqi.get("co",   {}).get("v", np.nan),
        "humidity":  iaqi.get("h",    {}).get("v", np.nan),
        "temp":      iaqi.get("t",    {}).get("v", np.nan),
        "pressure":  iaqi.get("p",    {}).get("v", np.nan),
        "wind":      iaqi.get("w",    {}).get("v", np.nan),
        "dew":       iaqi.get("dew",  {}).get("v", np.nan),
    }

# ─────────────────────────────────────────
# 2. FETCH WEATHER DATA FROM OPENWEATHER
# ─────────────────────────────────────────
def fetch_weather_data():
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print(f"⚠️  OpenWeather error: {data.get('message')} — using NaN fallback")
        return {
            "visibility": np.nan,
            "cloud_cover": np.nan,
            "weather_main": "Unknown",
        }

    return {
        "visibility":   data.get("visibility", np.nan),
        "cloud_cover":  data["clouds"]["all"],
        "weather_main": data["weather"][0]["main"],
    }

# ─────────────────────────────────────────
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────
def engineer_features(aqi_data, weather_data):
    now = datetime.now(timezone.utc)

    features = {
        # Raw AQI + pollutants
        **aqi_data,
        # Weather
        "visibility":   float(weather_data.get("visibility", 0.0)),
        "cloud_cover":  float(weather_data.get("cloud_cover", 0.0)),
        "weather_main": weather_data.get("weather_main", "Unknown"),
        # Time-based features
        "timestamp":    pd.Timestamp(now.strftime("%Y-%m-%d %H:%M:%S")),
        "hour":         now.hour,
        "day":          now.day,
        "month":        now.month,
        "day_of_week":  now.weekday(),
        "is_weekend":   int(now.weekday() >= 5),
        "is_rush_hour": int(now.hour in [7, 8, 9, 17, 18, 19]),
        # AQI category
        "aqi_category": categorize_aqi(aqi_data["aqi"]),
        # Lag features — for live pipeline we use current AQI as approximation
        "aqi_lag_1":       float(aqi_data["aqi"]),
        "aqi_lag_2":       float(aqi_data["aqi"]),
        "aqi_lag_3":       float(aqi_data["aqi"]),
        "aqi_rolling_3":   float(aqi_data["aqi"]),
        "aqi_rolling_7":   float(aqi_data["aqi"]),
        "aqi_change_rate": 0.0,
    }

    return pd.DataFrame([features])

def categorize_aqi(aqi):
    if aqi <= 50:   return 1   # Good
    elif aqi <= 100: return 2  # Moderate
    elif aqi <= 150: return 3  # Unhealthy for sensitive
    elif aqi <= 200: return 4  # Unhealthy
    elif aqi <= 300: return 5  # Very Unhealthy
    else:            return 6  # Hazardous

# ─────────────────────────────────────────
# 4. STORE FEATURES IN HOPSWORKS
# ─────────────────────────────────────────
def store_features(df):
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Hourly AQI and weather features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
    )

    # Fix types to match schema
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["visibility"]  = df["visibility"].astype(float)
    df["cloud_cover"] = df["cloud_cover"].astype(float)

    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Features stored successfully! Shape: {df.shape}")
# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_feature_pipeline():
    print(f"🚀 Running feature pipeline at {datetime.now()}")

    print("📡 Fetching AQI data...")
    aqi_data = fetch_aqi_data()
    print(f"   AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}")

    print("🌤️  Fetching weather data...")
    weather_data = fetch_weather_data()

    print("⚙️  Engineering features...")
    df = engineer_features(aqi_data, weather_data)

    print("💾 Storing in Hopsworks Feature Store...")
    store_features(df)

    print("✅ Feature pipeline complete!")

if __name__ == "__main__":
    run_feature_pipeline()

