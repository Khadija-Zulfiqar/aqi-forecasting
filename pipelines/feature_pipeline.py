import os
import requests
import numpy as np
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv
from utils import pm25_to_aqi, calc_dew_point, categorize_aqi_int, fetch_recent_aqi_history

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
LAT, LON        = 24.8607, 67.0011


# ─────────────────────────────────────────
# 1. FETCH AQI
# ─────────────────────────────────────────
def fetch_aqi_data():
    url  = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    r    = requests.get(url, timeout=10)
    data = r.json()

    if r.status_code != 200:
        raise ValueError(f"OpenWeather AQI error: {data}")

    components = data["list"][0]["components"]
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
# 2. FETCH WEATHER
# ─────────────────────────────────────────
def fetch_weather_data():
    url  = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    r    = requests.get(url, timeout=10)
    data = r.json()

    if r.status_code != 200:
        print(f"⚠️  Weather error — using defaults")
        return {
            "temp": 30.0, "humidity": 60.0, "pressure": 1013.0,
            "wind": 5.0,  "dew": 15.0,     "visibility": 8000.0,
            "cloud_cover": 30.0, "weather_main": "Clear",
        }

    temp     = float(data["main"]["temp"])
    humidity = float(data["main"]["humidity"])
    return {
        "temp":         temp,
        "humidity":     humidity,
        "pressure":     float(data["main"]["pressure"]),
        "wind":         float(data["wind"]["speed"]),
        "dew":          calc_dew_point(temp, humidity),
        "visibility":   float(data.get("visibility", 8000)),
        "cloud_cover":  float(data["clouds"]["all"]),
        "weather_main": data["weather"][0]["main"],
    }


# ─────────────────────────────────────────
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────
def engineer_features(aqi_data, weather_data):
    now = datetime.now(timezone.utc)

    print("   📡 Fetching AQI history for lag features...")
    history = fetch_recent_aqi_history(n=7, api_key=OPENWEATHER_KEY)
    print(f"   History (last 7h): {history}")

    return {
        **aqi_data,
        **{k: v for k, v in weather_data.items() if k != "weather_main"},
        "weather_main":    weather_data["weather_main"],
        "timestamp":       now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "hour":            int(now.hour),
        "day":             int(now.day),
        "month":           int(now.month),
        "day_of_week":     int(now.weekday()),
        "is_weekend":      int(now.weekday() >= 5),
        "is_rush_hour":    int(now.hour in [7, 8, 9, 17, 18, 19]),
        "aqi_category":    int(categorize_aqi_int(aqi_data["aqi"])),
        "aqi_lag_1":       float(history[-1]),
        "aqi_lag_2":       float(history[-2]),
        "aqi_lag_3":       float(history[-3]),
        "aqi_rolling_3":   float(np.mean(history[-3:])),
        "aqi_rolling_7":   float(np.mean(history[-7:])),
        "aqi_change_rate": float((history[-1] - history[-2]) / (history[-2] + 1e-9)),
    }


# ─────────────────────────────────────────
# 4. STORE IN SUPABASE
# ─────────────────────────────────────────
def store_features(row: dict):
    print("🔗 Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.table("aqi_features").upsert(row, on_conflict="timestamp").execute()
    print("✅ Feature row stored in Supabase!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_feature_pipeline():
    print(f"🚀 Feature pipeline at {datetime.now()}")

    print("🌍 Fetching AQI data...")
    aqi_data = fetch_aqi_data()
    print(f"   AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']:.1f}, PM10: {aqi_data['pm10']:.1f}")

    print("🌤️  Fetching weather data...")
    weather_data = fetch_weather_data()
    print(f"   Temp: {weather_data['temp']}°C, Humidity: {weather_data['humidity']}%, Dew: {weather_data['dew']}°C")

    print("⚙️  Engineering features...")
    row = engineer_features(aqi_data, weather_data)

    print("💾 Storing in Supabase...")
    store_features(row)
    print("✅ Feature pipeline complete!")


if __name__ == "__main__":
    run_feature_pipeline()
