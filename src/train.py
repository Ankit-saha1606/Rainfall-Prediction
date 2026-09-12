"""
train.py
--------
Reproducible training pipeline for the Vidarbha rainfall regressor.

Trains XGBoost and Random Forest regressors on engineered features,
evaluates both with MAE / RMSE / R^2 on a held-out split, logs metrics,
and persists the best model (+ metadata) to disk for serving.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from preprocessing import preprocess_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vidarbha_rainfall.train")

RANDOM_STATE = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_select_best(raw_data_path: str):
    start = time.time()
    logger.info("Starting training run")

    X, y, _ = preprocess_pipeline(raw_data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

    candidates = {
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["train_seconds"] = round(train_time, 3)
        results[name] = {"model": model, "metrics": metrics}

        logger.info(
            "%s -> MAE=%.2f RMSE=%.2f R2=%.4f (trained in %.2fs)",
            name, metrics["mae"], metrics["rmse"], metrics["r2"], train_time,
        )

    # Model selection: lowest RMSE wins
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best_model = results[best_name]["model"]
    logger.info("Selected best model: %s", best_name)

    # Persist model + feature schema + metrics for reproducibility
    model_path = MODEL_DIR / "rainfall_model.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model": best_name,
        "feature_columns": list(X.columns),
        "metrics": {k: v["metrics"] for k, v in results.items()},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved model to %s (total run time %.2fs)",
        model_path, time.time() - start,
    )
    return best_name, metadata


if __name__ == "__main__":
    train_and_select_best("data/rainfall_vidarbha_1901_2015.csv")
