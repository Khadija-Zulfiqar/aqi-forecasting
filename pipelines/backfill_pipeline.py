# pipelines/backfill_pipeline.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")
LAT, LON          = 24.8607, 67.0011

# ─────────────────────────────────────────
# 1. FETCH HISTORICAL AQI FROM OPENWEATHER
# ─────────────────────────────────────────
def fetch_historical_data(dt: datetime):
    """Fetch air pollution + weather for a specific past datetime"""
    unix_ts = int(dt.timestamp())

    # Air pollution history
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={unix_ts}&end={unix_ts+3600}&appid={OPENWEATHER_KEY}"
    aqi_r = requests.get(aqi_url).json()

    if not aqi_r.get("list"):
        return None

    components = aqi_r["list"][0]["components"]
    ow_aqi     = aqi_r["list"][0]["main"]["aqi"]
    aqi = pm25_to_aqi(components["pm2_5"])

    return {
        "aqi":  aqi,
        "pm25": components.get("pm2_5", 0),
        "pm10": components.get("pm10",  0),
        "o3":   components.get("o3",    0),
        "no2":  components.get("no2",   0),
        "so2":  components.get("so2",   0),
        "co":   components.get("co",    0),
    }

def fetch_historical_weather(dt: datetime):
    """Fetch historical weather using One Call API"""
    unix_ts = int(dt.timestamp())
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={LAT}&lon={LON}&dt={unix_ts}&appid={OPENWEATHER_KEY}&units=metric"
    r = requests.get(url)

    if r.status_code != 200:
        return {
            "temp": 30, "humidity": 60, "pressure": 1013,
            "wind": 5, "dew": 15, "visibility": 8000.0,
            "cloud_cover": 30.0, "weather_main": "Clear"
        }

    data = r.json()
    current = data.get("current", data.get("hourly", [{}])[0])
    return {
        "temp":         current.get("temp", 30),
        "humidity":     current.get("humidity", 60),
        "pressure":     current.get("pressure", 1013),
        "wind":         current.get("wind_speed", 5),
        "dew":          current.get("dew_point", 15),
        "visibility":   float(current.get("visibility", 8000)),
        "cloud_cover":  float(current.get("clouds", 30)),
        "weather_main": current.get("weather", [{"main": "Clear"}])[0]["main"],
    }

# ─────────────────────────────────────────
# 2. ENGINEER FEATURES
# ─────────────────────────────────────────
def categorize_aqi(aqi):
    if aqi <= 50:    return 1
    elif aqi <= 100: return 2
    elif aqi <= 150: return 3
    elif aqi <= 200: return 4
    elif aqi <= 300: return 5
    else:            return 6

def engineer_features(aqi_data, weather_data, dt):
    return {
        **aqi_data,
        **{k: v for k, v in weather_data.items() if k != "weather_main"},
        "weather_main": weather_data["weather_main"],
        "timestamp":    dt.strftime("%Y-%m-%d %H:%M:%S"),
        "hour":         dt.hour,
        "day":          dt.day,
        "month":        dt.month,
        "day_of_week":  dt.weekday(),
        "is_weekend":   int(dt.weekday() >= 5),
        "is_rush_hour": int(dt.hour in [7, 8, 9, 17, 18, 19]),
        "aqi_category": categorize_aqi(aqi_data["aqi"]),
    }

# ─────────────────────────────────────────
# 3. STORE IN HOPSWORKS
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
    df["aqi"]         = df["aqi"].astype("int64")
    df["pm25"]        = df["pm25"].astype("int64")
    df["temp"]        = df["temp"].astype("int64")
    df["dew"]         = df["dew"].astype("int64")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["is_weekend"]  = df["is_weekend"].astype("int64")
    df["is_rush_hour"]= df["is_rush_hour"].astype("int64")

    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Stored {len(df)} rows!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_backfill(days_back=30):
    print(f"🚀 Backfill for last {days_back} days using OpenWeather...")
    records = []
    now = datetime.utcnow()

    total = days_back * 24
    for i in range(total, 0, -1):
        dt = now - timedelta(hours=i)
        print(f"📡 {dt.strftime('%Y-%m-%d %H:00')} ({total-i+1}/{total})...")

        aqi_data = fetch_historical_data(dt)
        if aqi_data is None:
            print("   ⚠️  Skipping — no data")
            continue

        weather_data = fetch_historical_weather(dt)
        features = engineer_features(aqi_data, weather_data, dt)
        records.append(features)

    if not records:
        print("❌ No data collected!")
        return

    df = pd.DataFrame(records)
    df["aqi_lag_1"]       = df["aqi"].shift(1)
    df["aqi_lag_2"]       = df["aqi"].shift(2)
    df["aqi_lag_3"]       = df["aqi"].shift(3)
    df["aqi_rolling_3"]   = df["aqi"].rolling(3).mean()
    df["aqi_rolling_7"]   = df["aqi"].rolling(7).mean()
    df["aqi_change_rate"] = df["aqi"].pct_change()
    df = df.dropna(subset=["aqi_lag_3"]).fillna(0)

    print(f"\n📊 Summary: {len(df)} rows | AQI: {df['aqi'].min()}-{df['aqi'].max()} | Avg: {df['aqi'].mean():.1f}")
    store_features(df)
    print("✅ Backfill complete!")

if __name__ == "__main__":
    run_backfill(days_back=30)