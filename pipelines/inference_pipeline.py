# pipelines/inference_pipeline.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY")
LAT, LON          = 24.8607, 67.0011

def categorize_aqi(aqi):
    if aqi <= 50:    return "Good",                          "#00e400"
    elif aqi <= 100: return "Moderate",                      "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups","#ff7e00"
    elif aqi <= 200: return "Unhealthy",                     "#ff0000"
    elif aqi <= 300: return "Very Unhealthy",                "#8f3f97"
    else:            return "Hazardous",                     "#7e0023"

def fetch_current_data():
    """Fetch current AQI + weather from OpenWeather"""
    # Air pollution
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    aqi_r   = requests.get(aqi_url).json()
    components = aqi_r["list"][0]["components"]
    ow_aqi     = aqi_r["list"][0]["main"]["aqi"]
    ow_to_aqi  = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
    aqi        = ow_to_aqi.get(ow_aqi, 100)

    # Weather
    w_url  = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    w_data = requests.get(w_url).json()

    return {
        "aqi":         aqi,
        "pm25":        components.get("pm2_5", 0),
        "pm10":        components.get("pm10",  0),
        "o3":          components.get("o3",    0),
        "no2":         components.get("no2",   0),
        "so2":         components.get("so2",   0),
        "co":          components.get("co",    0),
        "temp":        w_data["main"]["temp"],
        "humidity":    w_data["main"]["humidity"],
        "pressure":    w_data["main"]["pressure"],
        "wind":        w_data["wind"]["speed"],
        "dew":         w_data["main"].get("feels_like", 25),
        "visibility":  float(w_data.get("visibility", 8000)),
        "cloud_cover": float(w_data["clouds"]["all"]),
    }

def load_model():
    model   = joblib.load("models/best_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

def generate_forecast():
    print("🔮 Generating 3-day AQI forecast for Karachi...")

    current       = fetch_current_data()
    model, scaler, feature_names = load_model()
    current_aqi   = current["aqi"]
    forecasts     = []
    now           = datetime.now()
    prev_aqi      = [current_aqi] * 7

    print(f"   Current AQI: {current_aqi}, PM2.5: {current['pm25']:.1f}, Temp: {current['temp']}°C")

    for h in range(1, 73):
        future = now + timedelta(hours=h)

        row = {
            "pm25":          current["pm25"],
            "pm10":          current["pm10"],
            "o3":            current["o3"],
            "no2":           current["no2"],
            "so2":           current["so2"],
            "co":            current["co"],
            "humidity":      current["humidity"],
            "temp":          current["temp"],
            "pressure":      current["pressure"],
            "wind":          current["wind"],
            "dew":           current["dew"],
            "hour":          future.hour,
            "day":           future.day,
            "month":         future.month,
            "day_of_week":   future.weekday(),
            "is_weekend":    int(future.weekday() >= 5),
            "is_rush_hour":  int(future.hour in [7,8,9,17,18,19]),
            "aqi_lag_1":     prev_aqi[-1],
            "aqi_lag_2":     prev_aqi[-2],
            "aqi_lag_3":     prev_aqi[-3],
            "aqi_rolling_3": np.mean(prev_aqi[-3:]),
            "aqi_rolling_7": np.mean(prev_aqi[-7:]),
            "aqi_change_rate": (prev_aqi[-1] - prev_aqi[-2]) / (prev_aqi[-2] + 1e-9),
            "cloud_cover":   current["cloud_cover"],
        }

        X = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
        X_scaled = scaler.transform(X)
        predicted = float(model.predict(X_scaled)[0])

        # Prevent runaway drift — cap change at 10% per hour
        max_change = prev_aqi[-1] * 0.10
        predicted  = float(np.clip(predicted,
                                   prev_aqi[-1] - max_change,
                                   prev_aqi[-1] + max_change))
        predicted  = max(0, min(500, round(predicted, 1)))

        category, color = categorize_aqi(predicted)
        forecasts.append({
            "datetime": future.strftime("%Y-%m-%d %H:%M"),
            "date":     future.strftime("%b %d"),
            "hour":     future.hour,
            "aqi":      predicted,
            "category": category,
            "color":    color,
        })

        prev_aqi.append(predicted)
        prev_aqi = prev_aqi[-7:]

    # Daily summary
    df    = pd.DataFrame(forecasts)
    daily = df.groupby("date").agg(
        avg_aqi  =("aqi", "mean"),
        max_aqi  =("aqi", "max"),
        min_aqi  =("aqi", "min"),
        category =("category", lambda x: x.mode()[0]),
        color    =("color",    lambda x: x.mode()[0]),
    ).reset_index()

    result = {
        "current_aqi":   current_aqi,
        "current_cat":   categorize_aqi(current_aqi)[0],
        "current_color": categorize_aqi(current_aqi)[1],
        "current_pm25":  current["pm25"],
        "current_temp":  current["temp"],
        "current_humidity": current["humidity"],
        "hourly":        forecasts,
        "daily":         daily.to_dict(orient="records"),
        "generated_at":  now.strftime("%Y-%m-%d %H:%M:%S"),
        "city":          "Karachi, Pakistan",
    }

    print(f"✅ Forecast generated!")
    for d in result["daily"]:
        print(f"   {d['date']:8s} → Avg: {d['avg_aqi']:.1f}, Max: {d['max_aqi']:.1f} ({d['category']})")

    return result

if __name__ == "__main__":
    generate_forecast()