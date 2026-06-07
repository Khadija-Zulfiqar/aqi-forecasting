import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import shap
from supabase import create_client
from dotenv import load_dotenv
from utils import pm25_to_aqi, calc_dew_point, categorize_aqi_int

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "humidity", "temp", "pressure", "wind", "dew",
    "hour", "day", "month", "day_of_week",
    "is_weekend", "is_rush_hour",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_rolling_3", "aqi_rolling_7",
    "aqi_change_rate", "cloud_cover",
]
TARGET_COL = "aqi"


# ─────────────────────────────────────────
# 1. FETCH FROM SUPABASE
# ─────────────────────────────────────────
def fetch_features():
    print("🔗 Connecting to Supabase...")
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Fetch all rows in batches (Supabase default limit is 1000)
        all_rows = []
        offset   = 0
        batch    = 1000

        while True:
            result = (
                supabase.table("aqi_features")
                .select("*")
                .order("timestamp")
                .range(offset, offset + batch - 1)
                .execute()
            )
            rows = result.data
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch
            if len(rows) < batch:
                break

        if not all_rows:
            raise ValueError("No rows returned from Supabase")

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"✅ Fetched {len(df)} rows from Supabase")
        print(f"   AQI — min: {df['aqi'].min()}, max: {df['aqi'].max()}, mean: {df['aqi'].mean():.1f}")
        return df

    except Exception as e:
        print(f"⚠️  Supabase failed: {e}")
        print("📥 Falling back to local CSV...")

        csv_path = "data/aqi_features_rows.csv"
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"temperature": "temp"})
        df = df.drop(columns=["id", "city", "created_at"], errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

        if "pm25" in df.columns:
            df["aqi"] = df["pm25"].apply(pm25_to_aqi)
        if "temp" in df.columns and "humidity" in df.columns:
            df["dew"] = df.apply(lambda r: calc_dew_point(r["temp"], r["humidity"]), axis=1)
        else:
            df["dew"] = 15.0

        df["visibility"]   = 8000.0
        df["cloud_cover"]  = 30.0
        df["weather_main"] = "Clear"
        df["day_of_week"]  = df["timestamp"].dt.weekday
        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)
        df["aqi_category"] = df["aqi"].apply(categorize_aqi_int)
        df["aqi_lag_1"]       = df["aqi"].shift(1)
        df["aqi_lag_2"]       = df["aqi"].shift(2)
        df["aqi_lag_3"]       = df["aqi"].shift(3)
        df["aqi_rolling_3"]   = df["aqi"].rolling(3).mean()
        df["aqi_rolling_7"]   = df["aqi"].rolling(7).mean()
        df["aqi_change_rate"] = df["aqi"].pct_change()
        df = df.dropna(subset=["aqi_lag_3"]).fillna(0)

        print(f"✅ Loaded {len(df)} rows from CSV")
        return df


# ─────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────
def prepare_data(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  Missing features (skipped): {missing}")

    df = df[available + [TARGET_COL]].dropna()
    X  = df[available]
    y  = df[TARGET_COL]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, available


# ─────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"   RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────
def train_ridge(X_train, X_test, y_train, y_test):
    print("\n🔵 Training Ridge Regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model, evaluate("Ridge Regression", y_test, model.predict(X_test))


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n🌲 Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, evaluate("Random Forest", y_test, model.predict(X_test))


def train_lstm(X_train, X_test, y_train, y_test):
    print("\n🧠 Training LSTM...")
    X_tr = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_te = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))
    model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_tr, y_train,
        epochs=100, batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1,
    )
    return model, evaluate("LSTM", y_test, model.predict(X_te).flatten())


# ─────────────────────────────────────────
# 5. SHAP
# ─────────────────────────────────────────
def compute_shap(model, X_train, feature_names):
    print("\n🔍 Computing SHAP values...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:100])
        mean_shap   = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": mean_shap,
        }).sort_values("importance", ascending=False)
        print("\n🏆 Top 10 Features:")
        print(shap_df.head(10).to_string(index=False))
        return shap_df
    except Exception as e:
        print(f"⚠️  SHAP failed: {e}")
        return None


# ─────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────
def save_best_model(models_metrics, models_dict, scaler, feature_names, shap_df):
    best       = min(models_metrics, key=lambda x: x["rmse"])
    best_name  = best["model"]
    best_model = models_dict[best_name]
    print(f"\n🏆 Best model: {best_name} (RMSE: {best['rmse']:.2f})")

    os.makedirs("models", exist_ok=True)

    if best_name == "LSTM":
        best_model.save("models/best_model.keras")
    else:
        joblib.dump(best_model, "models/best_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open("models/metrics.json", "w") as f:
        json.dump(best, f, indent=2)
    with open("models/best_model_name.json", "w") as f:
        json.dump({"name": best_name}, f)

    if shap_df is not None:
        shap_df.to_csv("models/shap_importance.csv", index=False)

    print("✅ Model saved to models/")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_training_pipeline():
    print(f"🚀 Training pipeline started at {datetime.now()}")

    df = fetch_features()
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    ridge_model, ridge_metrics = train_ridge(X_train, X_test, y_train, y_test)
    rf_model,    rf_metrics    = train_random_forest(X_train, X_test, y_train, y_test)
    lstm_model,  lstm_metrics  = train_lstm(X_train, X_test, y_train, y_test)

    all_metrics = [ridge_metrics, rf_metrics, lstm_metrics]
    all_models  = {
        "Ridge Regression": ridge_model,
        "Random Forest":    rf_model,
        "LSTM":             lstm_model,
    }

    shap_df = compute_shap(rf_model, X_train, feature_names)
    save_best_model(all_metrics, all_models, scaler, feature_names, shap_df)

    print(f"\n✅ Training complete at {datetime.now()}")
    print("\n📊 All Results:")
    for m in all_metrics:
        print(f"   {m['model']:20s} → RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} | R²: {m['r2']:.4f}")


if __name__ == "__main__":
    run_training_pipeline()