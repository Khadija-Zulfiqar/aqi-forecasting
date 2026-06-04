# pipelines/training_pipeline.py
# Fetches features from Hopsworks, trains multiple ML models,
# evaluates them and stores the best one in Model Registry

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Explainability
import shap

# Hopsworks
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqiforecasting")

# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────
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
# 1. FETCH FEATURES FROM HOPSWORKS
# ─────────────────────────────────────────
def fetch_features():
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)

    print("📥 Fetching features...")
    df = fg.read()
    print(f"✅ Fetched {len(df)} rows, {len(df.columns)} columns")
    return df, project

# ─────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────
def prepare_data(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep only needed columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available_features + [TARGET_COL]].dropna()

    X = df[available_features]
    y = df[TARGET_COL]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, available_features

# ─────────────────────────────────────────
# 3. EVALUATE MODEL
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
    metrics = evaluate("Ridge Regression", y_test, y_pred)
    return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n🌲 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate("Random Forest", y_test, y_pred)
    return model, metrics

def train_lstm(X_train, X_test, y_train, y_test):
    print("\n🧠 Training LSTM...")

    # Reshape for LSTM: (samples, timesteps, features)
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

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test_lstm).flatten()
    metrics = evaluate("LSTM", y_test, y_pred)
    return model, metrics

# ─────────────────────────────────────────
# 5. SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────
def compute_shap(model, X_train, feature_names):
    print("\n🔍 Computing SHAP values...")
    try:
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:100])
        mean_shap  = np.abs(shap_values).mean(axis=0)

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
# 6. SAVE BEST MODEL TO HOPSWORKS
# ─────────────────────────────────────────
def save_best_model(project, models_metrics, models_dict, scaler, feature_names, shap_df):
    # Pick best model by lowest RMSE
    best = min(models_metrics, key=lambda x: x["rmse"])
    best_name = best["model"]
    best_model = models_dict[best_name]

    print(f"\n🏆 Best model: {best_name} (RMSE: {best['rmse']:.2f})")

    # Save locally first
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

    # Push to Hopsworks Model Registry
    print("\n📤 Pushing to Hopsworks Model Registry...")
    mr = project.get_model_registry()

    model_meta = mr.sklearn.create_model(
        name="aqi_forecaster",
        version=1,
        metrics=best,
        description=f"Best AQI forecasting model: {best_name}",
        input_example=pd.DataFrame(
            [[0] * len(feature_names)], columns=feature_names
        ),
    )

    model_meta.save("models/")
    print("✅ Model saved to Hopsworks Model Registry!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_training_pipeline():
    print(f"🚀 Training pipeline started at {datetime.now()}")

    # Fetch data
    df, project = fetch_features()

    # Prepare
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    # Train all models
    ridge_model,  ridge_metrics  = train_ridge(X_train, X_test, y_train, y_test)
    rf_model,     rf_metrics     = train_random_forest(X_train, X_test, y_train, y_test)
    lstm_model,   lstm_metrics   = train_lstm(X_train, X_test, y_train, y_test)

    all_metrics = [ridge_metrics, rf_metrics, lstm_metrics]
    all_models  = {
        "Ridge Regression": ridge_model,
        "Random Forest":    rf_model,
        "LSTM":             lstm_model,
    }

    # SHAP on Random Forest (tree-based, works best)
    shap_df = compute_shap(rf_model, X_train, feature_names)

    # Save best model
    save_best_model(
        project, all_metrics, all_models,
        scaler, feature_names, shap_df
    )

    print(f"\n✅ Training pipeline complete at {datetime.now()}")
    print("\n📊 All Model Results:")
    for m in all_metrics:
        print(f"   {m['model']:20s} → RMSE: {m['rmse']:.2f}, MAE: {m['mae']:.2f}, R²: {m['r2']:.4f}")

if __name__ == "__main__":
    run_training_pipeline()