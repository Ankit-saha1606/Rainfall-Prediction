"""
preprocessing.py
-----------------
Data cleaning and feature engineering for the Vidarbha rainfall pipeline.

Turns the raw wide-format (one row per year, one column per month) rainfall
table into a supervised learning frame where each row is a (year) sample and
the target is ANNUAL rainfall, engineered with:
  - seasonal aggregates (winter / summer / monsoon / post-monsoon totals)
  - lag features (previous year's rainfall)
  - rolling averages (5-year trailing mean)
  - missing value imputation
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("vidarbha_rainfall.preprocessing")

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

SEASONS = {
    "WINTER": ["JAN", "FEB"],
    "SUMMER": ["MAR", "APR", "MAY"],
    "MONSOON": ["JUN", "JUL", "AUG", "SEP"],
    "POST_MONSOON": ["OCT", "NOV", "DEC"],
}


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info("Loading raw rainfall data from %s", path)
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing monthly values with the column median."""
    missing_before = df[MONTHS].isna().sum().sum()
    for col in MONTHS:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    missing_after = df[MONTHS].isna().sum().sum()
    logger.info("Imputed missing values: %d -> %d", missing_before, missing_after)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal aggregates, lag features, and rolling averages."""
    df = df.sort_values("YEAR").reset_index(drop=True)

    for season, months in SEASONS.items():
        df[f"{season}_TOTAL"] = df[months].sum(axis=1)

    if "ANNUAL" not in df.columns:
        df["ANNUAL"] = df[MONTHS].sum(axis=1)

    # Lag feature: previous year's annual rainfall
    df["ANNUAL_LAG1"] = df["ANNUAL"].shift(1)

    # 5-year trailing rolling mean (excludes current year to avoid leakage)
    df["ANNUAL_ROLLING5"] = df["ANNUAL"].shift(1).rolling(window=5, min_periods=1).mean()

    # Drop the first row(s) where lag features are undefined
    df = df.dropna(subset=["ANNUAL_LAG1"]).reset_index(drop=True)

    logger.info("Feature engineering complete: %d rows, %d columns", *df.shape)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for model training."""
    feature_cols = (
        MONTHS
        + [f"{s}_TOTAL" for s in SEASONS]
        + ["ANNUAL_LAG1", "ANNUAL_ROLLING5", "YEAR"]
    )
    X = df[feature_cols].copy()
    y = df["ANNUAL"].copy()
    return X, y


def preprocess_pipeline(raw_path: str):
    """Full preprocessing pipeline: load -> clean -> engineer -> split features."""
    df = load_raw_data(raw_path)
    df = handle_missing_values(df)
    df = engineer_features(df)
    X, y = build_feature_matrix(df)
    return X, y, df
