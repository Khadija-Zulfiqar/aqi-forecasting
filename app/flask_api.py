# app/flask_api.py
from flask import Flask, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.inference_pipeline import generate_forecast

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "AQI Forecasting API is running!"})

@app.route("/forecast")
def forecast():
    try:
        result = generate_forecast()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)