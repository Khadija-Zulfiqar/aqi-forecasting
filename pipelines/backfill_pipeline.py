import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
from utils import pm25_to_aqi, calc_dew_point, categorize_aqi_int

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
LAT, LON        = 24.8607, 67.0011


# ─────────────────────────────────────────
# 1. FETCH HISTORICAL AQI
# ─────────────────────────────────────────
def fetch_historical_data(dt: datetime):
    unix_ts = int(dt.timestamp())
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={unix_ts}&end={unix_ts + 3600}&appid={OPENWEATHER_KEY}"
    )
    try:
        r = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"   ⚠️  Request failed: {e}")
        return None

    if not r.get("list"):
        return None

    components = r["list"][0]["components"]
    pm25       = components.get("pm2_5", 0)

    return {
        "aqi":  pm25_to_aqi(pm25),
        "pm25": float(pm25),
        "pm10": float(min(components.get("pm10", 0), 400)),
        "o3":   float(components.get("o3",   0)),
        "no2":  float(components.get("no2",  0)),
        "so2":  float(components.get("so2",  0)),
        "co":   float(components.get("co",   0)),
    }


# ─────────────────────────────────────────
# 2. FETCH HISTORICAL WEATHER
# ─────────────────────────────────────────
def fetch_historical_weather(dt: datetime):
    unix_ts = int(dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={LAT}&lon={LON}&dt={unix_ts}&appid={OPENWEATHER_KEY}&units=metric"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise ValueError(r.json().get("message", "Unknown error"))
        data    = r.json()
        current = data.get("current", data.get("hourly", [{}])[0])
        temp     = float(current.get("temp", 30))
        humidity = float(current.get("humidity", 60))
        return {
            "temp":         temp,
            "humidity":     humidity,
            "pressure":     float(current.get("pressure", 1013)),
            "wind":         float(current.get("wind_speed", 5)),
            "dew":          calc_dew_point(temp, humidity),
            "visibility":   float(current.get("visibility", 8000)),
            "cloud_cover":  float(current.get("clouds", 30)),
            "weather_main": current.get("weather", [{"main": "Clear"}])[0]["main"],
        }
    except Exception as e:
        print(f"   ⚠️  Weather fallback ({e})")
        return {
            "temp": 30.0, "humidity": 60.0, "pressure": 1013.0,
            "wind": 5.0,  "dew": 15.0,     "visibility": 8000.0,
            "cloud_cover": 30.0, "weather_main": "Clear",
        }


# ─────────────────────────────────────────
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────
def engineer_features(aqi_data, weather_data, dt):
    return {
        **aqi_data,
        **{k: v for k, v in weather_data.items() if k != "weather_main"},
        "weather_main": weather_data["weather_main"],
        "timestamp":    dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "hour":         int(dt.hour),
        "day":          int(dt.day),
        "month":        int(dt.month),
        "day_of_week":  int(dt.weekday()),
        "is_weekend":   int(dt.weekday() >= 5),
        "is_rush_hour": int(dt.hour in [7, 8, 9, 17, 18, 19]),
        "aqi_category": int(categorize_aqi_int(aqi_data["aqi"])),
    }


# ─────────────────────────────────────────
# 4. STORE IN SUPABASE
# ─────────────────────────────────────────
def store_features(df):
    print("🔗 Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    records  = df.to_dict(orient="records")
    batch_size = 100
    total_stored = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        # upsert — safe to re-run, won't create duplicates
        supabase.table("aqi_features").upsert(batch, on_conflict="timestamp").execute()
        total_stored += len(batch)
        print(f"   📦 Stored {total_stored}/{len(records)} rows...")

    print(f"✅ All {total_stored} rows stored in Supabase!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_backfill(days_back=30):
    print(f"🚀 Backfill for last {days_back} days...")
    records = []
    now     = datetime.utcnow()
    total   = days_back * 24

    for i in range(total, 0, -1):
        dt = now - timedelta(hours=i)
        print(f"📡 {dt.strftime('%Y-%m-%d %H:00')} ({total - i + 1}/{total})...", end=" ")

        aqi_data = fetch_historical_data(dt)
        if aqi_data is None:
            print("⚠️  Skipping — no data")
            continue

        weather_data = fetch_historical_weather(dt)
        features     = engineer_features(aqi_data, weather_data, dt)
        records.append(features)
        print(f"AQI={aqi_data['aqi']}, PM2.5={aqi_data['pm25']:.1f}")

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

    print(f"\n📊 {len(df)} rows | AQI range: {df['aqi'].min()}-{df['aqi'].max()} | Avg: {df['aqi'].mean():.1f}")
    store_features(df)
    print("✅ Backfill complete!")


if __name__ == "__main__":
    run_backfill(days_back=90)
