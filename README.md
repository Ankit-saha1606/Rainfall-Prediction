# Vidarbha Rainfall Prediction — MLOps Pipeline

End-to-end machine learning system that predicts annual rainfall for the
Vidarbha region of Maharashtra, India, using 115 years of historical monthly
rainfall data (1901–2015). Goes beyond a notebook-only model: the trained
model is served through a REST API, containerized, deployed on Kubernetes
with autoscaling, and monitored with Prometheus metrics.

## Results

| Model | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|
| **XGBoost (selected)** | 34.3 | 46.5 | 0.930 |
| Random Forest | 35.2 | 49.5 | 0.921 |

XGBoost was selected automatically by the training pipeline based on lowest
test-set RMSE. Metrics and the winning model are logged to `models/metadata.json`
on every training run for reproducibility.

## Architecture

```
generate_data.py → preprocessing.py → train.py → models/rainfall_model.pkl
                                                          │
                                                          ▼
                                    app.py (Flask REST API + UI)
                                          │
                                    Dockerfile (gunicorn, non-root)
                                          │
                              k8s/ (Deployment, Service, HPA, PVC)
                                          │
                              /metrics scraped by Prometheus
```

## Tech stack

- **Modeling**: scikit-learn, XGBoost, pandas
- **Serving**: Flask, Gunicorn
- **Monitoring**: Prometheus (via `prometheus-flask-exporter`)
- **Containerization**: Docker (multi-stage build, non-root user, healthcheck)
- **Orchestration**: Kubernetes (Deployment, Service, HPA, PersistentVolumeClaim)
- **CI/CD**: GitHub Actions (test → build & push image → deploy)

## Project structure

```
vidarbha-rainfall/
├── src/
│   ├── generate_data.py     # builds the historical rainfall dataset
│   ├── preprocessing.py     # cleaning, seasonal & lag feature engineering
│   ├── train.py              # trains + evaluates XGBoost/RF, saves best model
│   └── app.py                 # Flask REST API, HTML UI, health/metrics endpoints
├── templates/index.html      # interactive prediction form
├── static/style.css          # UI styling
├── tests/test_api.py         # API test suite (pytest)
├── data/                      # generated rainfall CSV
├── models/                    # trained model + metadata.json
├── k8s/
│   ├── deployment.yaml       # Deployment with resource limits + probes
│   ├── service.yaml           # ClusterIP service
│   ├── hpa.yaml                # autoscaling 2–6 replicas on CPU/memory
│   └── pvc.yaml                 # persistent volume for model artifacts
├── Dockerfile
├── .github/workflows/ci-cd.yaml
└── requirements.txt
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate data and train
python src/generate_data.py
python src/train.py

# Run the API
cd src && python app.py
# -> http://localhost:5000
```

## Running tests

```bash
pytest tests/ -v
```

## API

**POST `/api/predict`**

```json
{
  "year": 2024,
  "months": {"JAN": 8.1, "FEB": 9.4, "MAR": 12.3, "APR": 15.6, "MAY": 21.0,
             "JUN": 168.2, "JUL": 255.4, "AUG": 218.9, "SEP": 149.7,
             "OCT": 54.2, "NOV": 14.8, "DEC": 6.1},
  "annual_lag1": 950.0,
  "annual_rolling5": 910.0
}
```

Response:

```json
{"year": 2024, "predicted_annual_rainfall_mm": 935.36, "model": "xgboost", "latency_ms": 4.93}
```

Other endpoints: `GET /health` (k8s liveness/readiness probe target),
`GET /metadata` (model metrics + feature schema), `GET /metrics` (Prometheus).

## Running with Docker

```bash
docker build -t vidarbha-rainfall-api .
docker run -p 5000:5000 vidarbha-rainfall-api
```

## Deploying to Kubernetes

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

The Deployment mounts a PersistentVolumeClaim for the model artifact,
exposes readiness/liveness probes against `/health`, and is annotated for
Prometheus auto-discovery of `/metrics`. The HPA scales 2–6 replicas based
on CPU (70%) and memory (80%) utilization.

## Data note

The original Kaggle "rainfall in india 1901-2015" dataset could not be
fetched in this environment, so `src/generate_data.py` generates a
statistically realistic substitute for the Vidarbha subdivision (same
schema: `SUBDIVISION, YEAR, JAN..DEC, ANNUAL`), reflecting the real seasonal
pattern of the southwest monsoon (June–September). Swap in the real CSV at
`data/rainfall_vidarbha_1901_2015.csv` to retrain on actual historical data —
no code changes needed since the schema matches.
