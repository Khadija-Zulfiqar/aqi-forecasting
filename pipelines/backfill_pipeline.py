# pipelines/backfill_pipeline.py
# Fetches historical AQI data to build training dataset

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN       = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")
CITY              = "karachi"

# ─────────────────────────────────────────
# 1. FETCH HISTORICAL DATA FROM AQICN
# ─────────────────────────────────────────
def fetch_historical_aqi(date_str):
    """Fetch AQI data for a specific date (YYYY-MM-DD)"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        return None

    d = data["data"]
    iaqi = d.get("iaqi", {})

    return {
        "aqi":          d["aqi"],
        "pm25":         iaqi.get("pm25", {}).get("v", np.nan),
        "pm10":         iaqi.get("pm10", {}).get("v", np.nan),
        "o3":           iaqi.get("o3",   {}).get("v", np.nan),
        "no2":          iaqi.get("no2",  {}).get("v", np.nan),
        "so2":          iaqi.get("so2",  {}).get("v", np.nan),
        "co":           iaqi.get("co",   {}).get("v", np.nan),
        "humidity":     iaqi.get("h",    {}).get("v", np.nan),
        "temp":         iaqi.get("t",    {}).get("v", np.nan),
        "pressure":     iaqi.get("p",    {}).get("v", np.nan),
        "wind":         iaqi.get("w",    {}).get("v", np.nan),
        "dew":          iaqi.get("dew",  {}).get("v", np.nan),
        "visibility":   np.nan,
        "cloud_cover":  np.nan,
        "weather_main": "Unknown",
    }

# ─────────────────────────────────────────
# 2. ENGINEER FEATURES FOR HISTORICAL DATE
# ─────────────────────────────────────────
def engineer_features(raw, date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    return {
        **raw,
        "timestamp": pd.Timestamp(date_str + " 12:00:00"),
        "hour":         12,
        "day":          dt.day,
        "month":        dt.month,
        "day_of_week":  dt.weekday(),
        "is_weekend":   int(dt.weekday() >= 5),
        "is_rush_hour": 0,
        "aqi_category": categorize_aqi(raw["aqi"]),
    }

def categorize_aqi(aqi):
    if aqi <= 50:    return 1
    elif aqi <= 100: return 2
    elif aqi <= 150: return 3
    elif aqi <= 200: return 4
    elif aqi <= 300: return 5
    else:            return 6

# ─────────────────────────────────────────
# 3. GENERATE DATE RANGE
# ─────────────────────────────────────────
def generate_date_range(days_back=180):
    """Generate list of dates from today going back N days"""
    dates = []
    today = datetime.now()
    for i in range(days_back, 0, -1):
        date = today - timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    return dates

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
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Stored {len(df)} rows successfully!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_backfill(days_back=180):
    print(f"🚀 Starting backfill for last {days_back} days...")

    dates = generate_date_range(days_back)
    records = []

    for i, date_str in enumerate(dates):
        print(f"📡 Fetching {date_str} ({i+1}/{len(dates)})...")
        raw = fetch_historical_aqi(date_str)

        if raw is None:
            print(f"   ⚠️  Skipping {date_str} — no data")
            continue

        features = engineer_features(raw, date_str)
        records.append(features)

    if not records:
        print("❌ No data collected!")
        return

    df = pd.DataFrame(records)

    # Add lag features (previous day AQI)
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_2"] = df["aqi"].shift(2)
    df["aqi_lag_3"] = df["aqi"].shift(3)

    # Rolling average (3-day and 7-day)
    df["aqi_rolling_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_rolling_7"] = df["aqi"].rolling(window=7).mean()

    # AQI change rate
    df["aqi_change_rate"] = df["aqi"].pct_change()

    # Drop rows with NaN from lag features
    df = df.dropna(subset=["aqi_lag_3"])
    df = df.fillna(0)

    print(f"\n📊 Dataset Summary:")
    print(f"   Total records : {len(df)}")
    print(f"   Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Avg AQI       : {df['aqi'].mean():.1f}")
    print(f"   Max AQI       : {df['aqi'].max()}")
    print(f"   Min AQI       : {df['aqi'].min()}")
    print(f"   Columns       : {list(df.columns)}")

    store_features(df)
    print("\n✅ Backfill complete!")

if __name__ == "__main__":
    run_backfill(days_back=180)

