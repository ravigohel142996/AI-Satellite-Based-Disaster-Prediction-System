# AI Satellite-Based Disaster Prediction System

A professional multi-page dashboard that simulates satellite sensor data and uses Machine Learning to predict disaster risk in real time.

## Features

- **Overview Dashboard** — live risk score, disaster level, sensor readings, and a risk gauge
- **Live Satellite Monitoring** — time-series charts for all five sensor channels with a configurable history window
- **Disaster Prediction Engine** — interactive sliders to adjust environmental parameters; the AI model instantly returns a risk score and sub-risk breakdown (Flood, Heatwave, Earthquake, Infrastructure)
- **Analytics** — risk score distribution, disaster level pie chart, correlation heatmap, scatter plots, and a simulated risk trend over time
- **Model Insights** — algorithm name, R² score, MAE, feature importance chart, model configuration table, and input feature reference

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Dashboard | Streamlit |
| ML Model | scikit-learn (RandomForestRegressor) |
| Charts | Plotly |
| Data | pandas / NumPy (synthetic satellite data) |

## Project Structure

```
.
├── app.py               # Multi-page Streamlit dashboard
├── model.py             # ML model training, prediction, and risk-level helpers
├── data_generator.py    # Synthetic satellite sensor data generation
├── requirements.txt     # Python dependencies
└── README.md
```

## ML Model

**Input features**

| Feature | Unit | Range |
|---|---|---|
| temperature | °C | 10 – 55 |
| rainfall | mm | 0 – 300 |
| soil_moisture | % | 5 – 95 |
| seismic_activity | Richter-like | 0 – 9 |
| water_level | m | 0 – 15 |

**Output**: `disaster_risk_score` (0 – 100)

**Risk levels**

| Score | Level |
|---|---|
| < 35 | Safe |
| 35 – 65 | Warning |
| > 65 | Critical |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
