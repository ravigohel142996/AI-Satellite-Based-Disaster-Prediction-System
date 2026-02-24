import numpy as np
import pandas as pd


def generate_satellite_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic satellite sensor data for disaster risk modeling."""
    rng = np.random.default_rng(seed)

    temperature = rng.uniform(10, 55, n_samples)
    rainfall = rng.uniform(0, 300, n_samples)
    soil_moisture = rng.uniform(5, 95, n_samples)
    seismic_activity = rng.uniform(0, 9, n_samples)
    water_level = rng.uniform(0, 15, n_samples)

    # Compute a risk score (0–100) as a weighted combination with noise
    risk_score = (
        0.25 * np.clip((temperature - 10) / 45 * 100, 0, 100)
        + 0.25 * np.clip(rainfall / 300 * 100, 0, 100)
        + 0.15 * np.clip(soil_moisture / 95 * 100, 0, 100)
        + 0.20 * np.clip(seismic_activity / 9 * 100, 0, 100)
        + 0.15 * np.clip(water_level / 15 * 100, 0, 100)
        + rng.normal(0, 3, n_samples)
    )
    risk_score = np.clip(risk_score, 0, 100)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "rainfall": rainfall,
            "soil_moisture": soil_moisture,
            "seismic_activity": seismic_activity,
            "water_level": water_level,
            "disaster_risk_score": risk_score,
        }
    )


def get_live_reading(seed: int | None = None) -> dict:
    """Return a single simulated live sensor reading."""
    rng = np.random.default_rng(seed)
    return {
        "temperature": float(rng.uniform(10, 55)),
        "rainfall": float(rng.uniform(0, 300)),
        "soil_moisture": float(rng.uniform(5, 95)),
        "seismic_activity": float(rng.uniform(0, 9)),
        "water_level": float(rng.uniform(0, 15)),
    }
