# pipelines/upload_friend_data.py
# Uploads friend's historical data into our Hopsworks feature store

import pandas as pd
import numpy as np
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

print("📥 Loading friend's data...")
df = pd.read_csv('data/aqi_features_rows.csv')
print(f"   Loaded {len(df)} rows")

# ── Rename columns to match our schema ──
df = df.rename(columns={'temperature': 'temp'})

# ── Drop columns we don't need ──
df = df.drop(columns=['id', 'city', 'created_at'], errors='ignore')

# ── Fix timestamp ──
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
df = df.sort_values('timestamp').reset_index(drop=True)

# ── Add missing columns ──
df['dew']          = -10.0
df['visibility']   = 8000.0
df['cloud_cover']  = 30.0
df['weather_main'] = 'Clear'
df['day_of_week']  = df['timestamp'].dt.weekday
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)

# ── AQI category ──
def categorize_aqi(aqi):
    if aqi <= 50:    return 1
    elif aqi <= 100: return 2
    elif aqi <= 150: return 3
    elif aqi <= 200: return 4
    elif aqi <= 300: return 5
    else:            return 6

df['aqi_category'] = df['aqi'].apply(categorize_aqi)

# ── Lag features ──
df['aqi_lag_1']       = df['aqi'].shift(1)
df['aqi_lag_2']       = df['aqi'].shift(2)
df['aqi_lag_3']       = df['aqi'].shift(3)
df['aqi_rolling_3']   = df['aqi'].rolling(3).mean()
df['aqi_rolling_7']   = df['aqi'].rolling(7).mean()
df['aqi_change_rate'] = df['aqi'].pct_change()

# ── Clean up ──
df = df.dropna(subset=['aqi_lag_3']).fillna(0)
df['visibility']  = df['visibility'].astype(float)
df['cloud_cover'] = df['cloud_cover'].astype(float)

print(f"\n✅ Prepared {len(df)} rows")
print(f"   Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"   Avg AQI    : {df['aqi'].mean():.1f}")
print(f"   Columns    : {list(df.columns)}")

# ── Push to Hopsworks ──
print("\n🔗 Connecting to Hopsworks...")
project = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT", "aqiforecasting"),
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

print(f"💾 Uploading {len(df)} rows to Hopsworks...")
fg.insert(df, write_options={"wait_for_job": True})
print("✅ Done! Data is now in Hopsworks!")