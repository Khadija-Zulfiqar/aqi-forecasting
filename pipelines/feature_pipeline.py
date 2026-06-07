# pipelines/feature_pipeline.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hopsworks
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")
LAT, LON          = 24.8607, 67.0011

# ─────────────────────────────────────────
# 1. FETCH AQI FROM OPENWEATHER
# ─────────────────────────────────────────
def fetch_aqi_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    r = requests.get(url)
    data = r.json()

    if r.status_code != 200:
        raise ValueError(f"OpenWeather AQI error: {data}")

    components = data["list"][0]["components"]
    ow_aqi     = data["list"][0]["main"]["aqi"]

    # Convert OpenWeather 1-5 index to standard AQI
    ow_to_aqi = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
    aqi = ow_to_aqi.get(ow_aqi, 100)

    return {
        "aqi":   aqi,
        "pm25":  components.get("pm2_5", 0),
        "pm10":  components.get("pm10",  0),
        "o3":    components.get("o3",    0),
        "no2":   components.get("no2",   0),
        "so2":   components.get("so2",   0),
        "co":    components.get("co",    0),
    }

# ─────────────────────────────────────────
# 2. FETCH WEATHER FROM OPENWEATHER
# ─────────────────────────────────────────
def fetch_weather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    r = requests.get(url)
    data = r.json()

    if r.status_code != 200:
        print(f"⚠️ Weather error: {data.get('message')} — using defaults")
        return {
            "temp": 30, "humidity": 60, "pressure": 1013,
            "wind": 5, "dew": 15, "visibility": 8000,
            "cloud_cover": 30, "weather_main": "Clear"
        }

    return {
        "temp":         data["main"]["temp"],
        "humidity":     data["main"]["humidity"],
        "pressure":     data["main"]["pressure"],
        "wind":         data["wind"]["speed"],
        "dew":          data["main"].get("feels_like", 25),
        "visibility":   float(data.get("visibility", 8000)),
        "cloud_cover":  float(data["clouds"]["all"]),
        "weather_main": data["weather"][0]["main"],
    }

# ─────────────────────────────────────────
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────
def categorize_aqi(aqi):
    if aqi <= 50:    return 1
    elif aqi <= 100: return 2
    elif aqi <= 150: return 3
    elif aqi <= 200: return 4
    elif aqi <= 300: return 5
    else:            return 6

def engineer_features(aqi_data, weather_data):
    now = datetime.now(timezone.utc)

    features = {
        **aqi_data,
        **{k: v for k, v in weather_data.items() if k != "weather_main"},
        "weather_main": weather_data["weather_main"],
        "timestamp":    pd.Timestamp(now.strftime("%Y-%m-%d %H:%M:%S")),
        "hour":         now.hour,
        "day":          now.day,
        "month":        now.month,
        "day_of_week":  now.weekday(),
        "is_weekend":   int(now.weekday() >= 5),
        "is_rush_hour": int(now.hour in [7, 8, 9, 17, 18, 19]),
        "aqi_category": categorize_aqi(aqi_data["aqi"]),
        "aqi_lag_1":       float(aqi_data["aqi"]),
        "aqi_lag_2":       float(aqi_data["aqi"]),
        "aqi_lag_3":       float(aqi_data["aqi"]),
        "aqi_rolling_3":   float(aqi_data["aqi"]),
        "aqi_rolling_7":   float(aqi_data["aqi"]),
        "aqi_change_rate": 0.0,
    }

    return pd.DataFrame([features])

# ─────────────────────────────────────────
# 4. STORE IN HOPSWORKS
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
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["visibility"]  = df["visibility"].astype(float)
    df["cloud_cover"] = df["cloud_cover"].astype(float)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Features stored! Shape: {df.shape}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_feature_pipeline():
    print(f"🚀 Feature pipeline at {datetime.now()}")

    print("🌍 Fetching AQI data from OpenWeather...")
    aqi_data = fetch_aqi_data()
    print(f"   AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']:.1f}, PM10: {aqi_data['pm10']:.1f}")

    print("🌤️  Fetching weather data...")
    weather_data = fetch_weather_data()
    print(f"   Temp: {weather_data['temp']}°C, Humidity: {weather_data['humidity']}%")

    print("⚙️  Engineering features...")
    df = engineer_features(aqi_data, weather_data)

    print("💾 Storing in Hopsworks...")
    store_features(df)
    print("✅ Feature pipeline complete!")

if __name__ == "__main__":
    run_feature_pipeline()