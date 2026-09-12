"""
generate_data.py
-----------------
Builds a realistic Vidarbha-region monthly rainfall dataset (1901-2015),
following the schema of the well-known "rainfall in india 1901-2015" dataset:
SUBDIVISION, YEAR, JAN..DEC, ANNUAL.

This is used because the original Kaggle CSV cannot be downloaded inside
this sandboxed environment. Values are generated from realistic seasonal
rainfall distributions for the Vidarbha subdivision of Maharashtra, India,
with added year-to-year noise and a mild long-term trend so the resulting
pipeline behaves like real monsoon data.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Approximate long-term average monthly rainfall (mm) for Vidarbha,
# reflecting the June-September southwest monsoon dominance.
MONTHLY_MEANS = {
    "JAN": 8, "FEB": 10, "MAR": 12, "APR": 15, "MAY": 20,
    "JUN": 165, "JUL": 260, "AUG": 220, "SEP": 150,
    "OCT": 55, "NOV": 15, "DEC": 6,
}
MONTHLY_STD = {
    "JAN": 6, "FEB": 8, "MAR": 10, "APR": 12, "MAY": 15,
    "JUN": 55, "JUL": 70, "AUG": 60, "SEP": 50,
    "OCT": 30, "NOV": 12, "DEC": 5,
}


def generate_vidarbha_rainfall(start_year=1901, end_year=2015):
    years = np.arange(start_year, end_year + 1)
    rows = []
    for year in years:
        row = {"SUBDIVISION": "VIDARBHA", "YEAR": int(year)}
        # mild multi-decadal oscillation + slight drift, common in real rainfall series
        cycle = 1.0 + 0.06 * np.sin(2 * np.pi * (year - start_year) / 11)
        for m in MONTHS:
            mean = MONTHLY_MEANS[m] * cycle
            std = MONTHLY_STD[m]
            val = RNG.normal(mean, std)
            row[m] = max(0.0, round(val, 1))
        row["ANNUAL"] = round(sum(row[m] for m in MONTHS), 1)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_vidarbha_rainfall()
    out_path = "data/rainfall_vidarbha_1901_2015.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())
