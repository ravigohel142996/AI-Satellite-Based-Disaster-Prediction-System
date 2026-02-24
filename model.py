from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_generator import generate_satellite_data

FEATURE_COLS = [
    "temperature",
    "rainfall",
    "soil_moisture",
    "seismic_activity",
    "water_level",
]
TARGET_COL = "disaster_risk_score"


def train_model(
    n_samples: int = 1000, seed: int = 42
) -> tuple[RandomForestRegressor, dict, pd.DataFrame]:
    """Train a RandomForestRegressor and return (model, metrics, feature_importance_df)."""
    df = generate_satellite_data(n_samples=n_samples, seed=seed)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2_score": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2_score_pct": float(r2_score(y_test, y_pred) * 100),
    }

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, importance_df


def predict_risk(model: RandomForestRegressor, inputs: dict) -> float:
    """Predict disaster risk score (0–100) from a dict of sensor values."""
    X = pd.DataFrame([inputs])[FEATURE_COLS]
    score = float(model.predict(X)[0])
    return float(np.clip(score, 0, 100))


def risk_level(score: float) -> str:
    """Return a human-readable disaster level label."""
    if score < 35:
        return "Safe"
    if score < 65:
        return "Warning"
    return "Critical"
