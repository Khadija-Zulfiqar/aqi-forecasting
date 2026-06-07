# pipelines/utils.py
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone

LAT, LON = 24.8607, 67.0011


# ─────────────────────────────────────────
# AQI CALCULATION (EPA PM2.5 Formula)
# ─────────────────────────────────────────
def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration (µg/m³) to EPA AQI."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm25 = max(0.0, float(pm25))
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round((i_hi - i_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + i_lo)
    return 500


# ─────────────────────────────────────────
# DEW POINT (Magnus Formula)
# ─────────────────────────────────────────
def calc_dew_point(temp, humidity):
    """Calculate dew point from temperature (°C) and relative humidity (%)."""
    humidity = max(1, float(humidity))
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
    return round((b * alpha) / (a - alpha), 1)


# ─────────────────────────────────────────
# AQI CATEGORY
# ─────────────────────────────────────────
def categorize_aqi_int(aqi):
    """Return integer category (1-6) for storage/training."""
    if aqi <= 50:    return 1
    elif aqi <= 100: return 2
    elif aqi <= 150: return 3
    elif aqi <= 200: return 4
    elif aqi <= 300: return 5
    else:            return 6


def categorize_aqi_label(aqi):
    """Return (label, color) for display."""
    if aqi <= 50:    return "Good",                           "#00e400"
    elif aqi <= 100: return "Moderate",                       "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200: return "Unhealthy",                      "#ff0000"
    elif aqi <= 300: return "Very Unhealthy",                 "#8f3f97"
    else:            return "Hazardous",                      "#7e0023"


# ─────────────────────────────────────────
# FETCH RECENT AQI HISTORY (for lag features)
# ─────────────────────────────────────────
def fetch_recent_aqi_history(n=7, api_key=None):
    """
    Fetch last n hours of AQI from OpenWeather history API.
    Returns a list of AQI values (oldest → newest), length n.
    Falls back to repeated current value if API fails.
    """
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    now     = datetime.now(timezone.utc)
    history = []

    for i in range(n, 0, -1):
        dt      = now - timedelta(hours=i)
        unix_ts = int(dt.timestamp())
        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={LAT}&lon={LON}&start={unix_ts}&end={unix_ts + 3600}&appid={api_key}"
        )
        try:
            r    = requests.get(url, timeout=10).json()
            items = r.get("list", [])
            if items:
                pm25 = items[0]["components"]["pm2_5"]
                history.append(pm25_to_aqi(pm25))
            else:
                history.append(None)
        except Exception:
            history.append(None)

    # Fill gaps with interpolation / forward-fill
    valid = [x for x in history if x is not None]
    if not valid:
        return [100] * n  # fallback default

    # Replace None with nearest valid value
    filled = []
    last   = valid[0]
    for v in history:
        if v is not None:
            last = v
        filled.append(last)

    return filled  # list of n values, oldest first