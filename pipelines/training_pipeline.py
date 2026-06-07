# pipelines/training_pipeline.py
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
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")

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
# 1. FETCH FEATURES
# ─────────────────────────────────────────
def fetch_features():
    try:
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features", version=1)
        print("📥 Fetching features from Hopsworks...")
        df = fg.read()

        if len(df) < 100:
            raise ValueError(f"Only {len(df)} rows in Hopsworks — using CSV")

        print(f"✅ Fetched {len(df)} rows from Hopsworks")
        return df, project

    except Exception as e:
        print(f"⚠️  {e}")
        print("📥 Loading from friend's CSV instead...")

        csv_path = "data/aqi_features_rows.csv"
        df = pd.read_csv(csv_path)
        df = df.rename(columns={'temperature': 'temp'})
        df = df.drop(columns=['id', 'city', 'created_at'], errors='ignore')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['dew']          = -10.0
        df['visibility']   = 8000.0
        df['cloud_cover']  = 30.0
        df['weather_main'] = 'Clear'
        df['day_of_week']  = df['timestamp'].dt.weekday
        df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
        df['aqi_category'] = df['aqi'].apply(lambda x: 1 if x<=50 else 2 if x<=100 else 3 if x<=150 else 4 if x<=200 else 5 if x<=300 else 6)
        df['aqi_lag_1']       = df['aqi'].shift(1)
        df['aqi_lag_2']       = df['aqi'].shift(2)
        df['aqi_lag_3']       = df['aqi'].shift(3)
        df['aqi_rolling_3']   = df['aqi'].rolling(3).mean()
        df['aqi_rolling_7']   = df['aqi'].rolling(7).mean()
        df['aqi_change_rate'] = df['aqi'].pct_change()
        df = df.dropna(subset=['aqi_lag_3']).fillna(0)
        print(f"✅ Loaded {len(df)} rows from CSV")
        return df, None

# ─────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────
def prepare_data(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available_features + [TARGET_COL]].dropna()
    X = df[available_features]
    y = df[TARGET_COL]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, available_features

# ─────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📈 {name} Results:")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   MAE  : {mae:.2f}")
    print(f"   R²   : {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}

# ─────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────
def train_ridge(X_train, X_test, y_train, y_test):
    print("\n🔵 Training Ridge Regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate("Ridge Regression", y_test, y_pred)

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n🌲 Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate("Random Forest", y_test, y_pred)

def train_lstm(X_train, X_test, y_train, y_test):
    print("\n🧠 Training LSTM...")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))
    model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train_lstm, y_train,
        epochs=100, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    y_pred = model.predict(X_test_lstm).flatten()
    return model, evaluate("LSTM", y_test, y_pred)

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
            "importance": mean_shap
        }).sort_values("importance", ascending=False)
        print("\n🏆 Top 10 Most Important Features:")
        print(shap_df.head(10).to_string(index=False))
        return shap_df
    except Exception as e:
        print(f"⚠️  SHAP failed: {e}")
        return None

# ─────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────
def save_best_model(project, models_metrics, models_dict, scaler, feature_names, shap_df):
    best      = min(models_metrics, key=lambda x: x["rmse"])
    best_name = best["model"]
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
    if shap_df is not None:
        shap_df.to_csv("models/shap_importance.csv", index=False)
    print("✅ Model saved locally to models/ folder!")

    if project:
        try:
            print("📤 Pushing to Hopsworks Model Registry...")
            mr = project.get_model_registry()
            model_meta = mr.sklearn.create_model(
                name="aqi_forecaster",
                version=1,
                metrics=best,
                description=f"Best AQI forecasting model: {best_name}",
            )
            model_meta.save("models/")
            print("✅ Model saved to Hopsworks Model Registry!")
        except Exception as e:
            print(f"⚠️  Could not save to Hopsworks registry: {e}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_training_pipeline():
    print(f"🚀 Training pipeline started at {datetime.now()}")

    df, project = fetch_features()
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
    save_best_model(project, all_metrics, all_models, scaler, feature_names, shap_df)

    print(f"\n✅ Training pipeline complete at {datetime.now()}")
    print("\n📊 All Model Results:")
    for m in all_metrics:
        print(f"   {m['model']:20s} → RMSE: {m['rmse']:.2f}, MAE: {m['mae']:.2f}, R²: {m['r2']:.4f}")

if __name__ == "__main__":
    run_training_pipeline()