# app/flask_api.py
# Production Flask API for AQI Forecasting

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# LAZY IMPORT — avoids slow startup
# ─────────────────────────────────────────
def get_forecast():
    from pipelines.inference_pipeline import generate_forecast
    return generate_forecast()

def get_all_model_forecasts():
    from pipelines.inference_pipeline import generate_all_model_forecasts
    return generate_all_model_forecasts()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Karachi AQI Forecasting API",
        "endpoints": ["/forecast", "/forecast/all-models", "/health"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/forecast")
def forecast():
    try:
        result = get_forecast()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast/all-models")
def forecast_all_models():
    try:
        result = get_all_model_forecasts()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
    