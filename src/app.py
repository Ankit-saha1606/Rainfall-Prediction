"""
app.py
------
Flask REST API + UI for the Vidarbha Rainfall Prediction service.

Endpoints:
  GET  /                -> HTML form for interactive predictions
  POST /predict          -> HTML form submission handler
  POST /api/predict      -> JSON REST endpoint
  GET  /health           -> health check for k8s liveness/readiness probes
  GET  /metadata          -> model metadata (metrics, feature columns, version)
"""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from prometheus_flask_exporter import PrometheusMetrics

# ---------------------------------------------------------------------------
# Logging setup: structured, timestamped, writes to both stdout and a file.
# stdout is what kubectl logs / container log drivers pick up in production.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.environ.get("LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log"),
    ],
)
logger = logging.getLogger("vidarbha_rainfall.api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models")))
MODEL_PATH = MODEL_DIR / "rainfall_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SEASONS = {
    "WINTER": ["JAN", "FEB"],
    "SUMMER": ["MAR", "APR", "MAY"],
    "MONSOON": ["JUN", "JUL", "AUG", "SEP"],
    "POST_MONSOON": ["OCT", "NOV", "DEC"],
}

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent.parent / "static"),
)

# Exposes /metrics in Prometheus text format: request counts, latency
# histograms per endpoint, and in-progress request gauges. This is what a
# Prometheus server (or a k8s ServiceMonitor) scrapes for the monitoring
# stack referenced in the deployment.
metrics = PrometheusMetrics(app)
metrics.info("vidarbha_rainfall_app_info", "Vidarbha Rainfall Prediction API", version="1.0.0")

prediction_counter = metrics.counter(
    "rainfall_predictions_total", "Total number of rainfall predictions served",
    labels={"status": lambda r: "success" if r.status_code == 200 else "error"},
)

# ---------------------------------------------------------------------------
# Model loading with robust error handling. The app should still start
# (health checks pass) even if the model artifact is temporarily missing,
# but /predict will report a clear 503 rather than crashing the process.
# ---------------------------------------------------------------------------
model = None
metadata = None
model_load_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    else:
        model_load_error = f"Model artifact not found at {MODEL_PATH}"
        logger.warning(model_load_error)

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        logger.info("Loaded metadata: best_model=%s", metadata.get("best_model"))
except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
    model_load_error = f"Failed to load model: {exc}"
    logger.exception(model_load_error)


def build_features(monthly_values: dict, year: int, lag1: float, rolling5: float) -> pd.DataFrame:
    """Reconstruct the exact feature schema the model was trained on."""
    row = {m: float(monthly_values.get(m, 0.0)) for m in MONTHS}
    for season, months in SEASONS.items():
        row[f"{season}_TOTAL"] = sum(row[m] for m in months)
    row["ANNUAL_LAG1"] = float(lag1)
    row["ANNUAL_ROLLING5"] = float(rolling5)
    row["YEAR"] = int(year)

    if metadata and "feature_columns" in metadata:
        cols = metadata["feature_columns"]
    else:
        cols = MONTHS + [f"{s}_TOTAL" for s in SEASONS] + ["ANNUAL_LAG1", "ANNUAL_ROLLING5", "YEAR"]

    return pd.DataFrame([row])[cols]


def validate_payload(payload: dict):
    """Raise ValueError with a clear message on bad input."""
    if "year" not in payload:
        raise ValueError("Missing required field: 'year'")
    if "months" not in payload or not isinstance(payload["months"], dict):
        raise ValueError("Missing or invalid required field: 'months' (expected object)")

    missing_months = [m for m in MONTHS if m not in payload["months"]]
    if missing_months:
        raise ValueError(f"Missing monthly values for: {', '.join(missing_months)}")

    for m in MONTHS:
        val = payload["months"][m]
        if not isinstance(val, (int, float)):
            raise ValueError(f"Monthly value for {m} must be numeric, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"Monthly value for {m} cannot be negative")


@app.route("/health", methods=["GET"])
def health():
    """Liveness/readiness probe target for Kubernetes."""
    status = "ok" if model is not None else "degraded"
    code = 200 if model is not None else 503
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "error": model_load_error,
    }), code


@app.route("/metadata", methods=["GET"])
def get_metadata():
    if metadata is None:
        return jsonify({"error": "No model metadata available"}), 503
    return jsonify(metadata), 200


@app.route("/api/predict", methods=["POST"])
@prediction_counter
def api_predict():
    """JSON REST endpoint.

    Expected body:
    {
      "year": 2024,
      "months": {"JAN": 8.1, "FEB": 9.4, ..., "DEC": 6.0},
      "annual_lag1": 950.2,
      "annual_rolling5": 910.5
    }
    """
    request_start = time.time()

    if model is None:
        logger.error("Prediction requested but model is not loaded")
        return jsonify({"error": "Model unavailable", "detail": model_load_error}), 503

    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            raise ValueError("Request body must be valid JSON")

        validate_payload(payload)

        year = int(payload["year"])
        lag1 = float(payload.get("annual_lag1", 0.0))
        rolling5 = float(payload.get("annual_rolling5", 0.0))

        X = build_features(payload["months"], year, lag1, rolling5)
        prediction = float(model.predict(X)[0])

        elapsed_ms = round((time.time() - request_start) * 1000, 2)
        logger.info(
            "Prediction served: year=%s prediction=%.2fmm latency=%.2fms",
            year, prediction, elapsed_ms,
        )
        return jsonify({
            "year": year,
            "predicted_annual_rainfall_mm": round(prediction, 2),
            "model": metadata.get("best_model") if metadata else "unknown",
            "latency_ms": elapsed_ms,
        }), 200

    except ValueError as ve:
        logger.warning("Bad request: %s", ve)
        return jsonify({"error": "Invalid input", "detail": str(ve)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["POST"])
def form_predict():
    """Handles the HTML form submission for interactive, human-friendly use."""
    try:
        year = int(request.form.get("year", 0))
        months = {m: float(request.form.get(m, 0) or 0) for m in MONTHS}
        lag1 = float(request.form.get("annual_lag1", 0) or 0)
        rolling5 = float(request.form.get("annual_rolling5", 0) or 0)

        payload = {"year": year, "months": months, "annual_lag1": lag1, "annual_rolling5": rolling5}
        validate_payload(payload)

        if model is None:
            raise RuntimeError(model_load_error or "Model unavailable")

        X = build_features(months, year, lag1, rolling5)
        prediction = float(model.predict(X)[0])
        logger.info("Form prediction served: year=%s prediction=%.2fmm", year, prediction)

        return render_template(
            "index.html",
            metadata=metadata,
            prediction=round(prediction, 2),
            year=year,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Form submission error: %s", exc)
        return render_template("index.html", metadata=metadata, error=str(exc)), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Vidarbha Rainfall API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
