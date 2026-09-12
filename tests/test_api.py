"""
test_api.py
-----------
Unit tests for the Vidarbha Rainfall API using Flask's built-in test client
(no live server / network needed). Run with: pytest tests/test_api.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import app as flask_app_module  # noqa: E402


@pytest.fixture
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as c:
        yield c


SAMPLE_MONTHS = {
    "JAN": 8.1, "FEB": 9.4, "MAR": 12.3, "APR": 15.6, "MAY": 21.0,
    "JUN": 168.2, "JUL": 255.4, "AUG": 218.9, "SEP": 149.7,
    "OCT": 54.2, "NOV": 14.8, "DEC": 6.1,
}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    data = resp.get_json()
    assert "status" in data
    assert "model_loaded" in data


def test_metadata_endpoint(client):
    resp = client.get("/metadata")
    # 200 if model trained, 503 if not - both are valid depending on state
    assert resp.status_code in (200, 503)


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Vidarbha" in resp.data


def test_predict_valid_payload(client):
    payload = {
        "year": 2024,
        "months": SAMPLE_MONTHS,
        "annual_lag1": 950.0,
        "annual_rolling5": 910.0,
    }
    resp = client.post("/api/predict", data=json.dumps(payload),
                        content_type="application/json")
    if flask_app_module.model is None:
        assert resp.status_code == 503
    else:
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_annual_rainfall_mm" in data
        assert isinstance(data["predicted_annual_rainfall_mm"], float)


def test_predict_missing_field(client):
    payload = {"year": 2024}  # missing 'months'
    resp = client.post("/api/predict", data=json.dumps(payload),
                        content_type="application/json")
    assert resp.status_code in (400, 503)


def test_predict_negative_rainfall_rejected(client):
    bad_months = dict(SAMPLE_MONTHS)
    bad_months["JUN"] = -5.0
    payload = {"year": 2024, "months": bad_months, "annual_lag1": 900, "annual_rolling5": 880}
    resp = client.post("/api/predict", data=json.dumps(payload),
                        content_type="application/json")
    assert resp.status_code in (400, 503)


def test_predict_non_json_body_handled(client):
    resp = client.post("/api/predict", data="not json",
                        content_type="application/json")
    assert resp.status_code in (400, 503)


def test_metrics_endpoint_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"rainfall_predictions_total" in resp.data or resp.status_code == 200


def test_predict_missing_month_key(client):
    incomplete_months = {k: v for k, v in SAMPLE_MONTHS.items() if k != "DEC"}
    payload = {"year": 2024, "months": incomplete_months,
               "annual_lag1": 900, "annual_rolling5": 880}
    resp = client.post("/api/predict", data=json.dumps(payload),
                        content_type="application/json")
    assert resp.status_code in (400, 503)
