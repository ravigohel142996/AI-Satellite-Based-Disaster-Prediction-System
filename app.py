"""
AI Satellite-Based Disaster Prediction System
Multi-page Streamlit dashboard
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_satellite_data, get_live_reading
from model import FEATURE_COLS, predict_risk, risk_level, train_model

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Satellite Disaster Prediction",
    page_icon=None,  # no icon per design spec (clean professional look)
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    /* ── Global background + typography ── */
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(88, 166, 255, 0.14), transparent 35%),
            radial-gradient(circle at 80% 20%, rgba(248, 81, 73, 0.10), transparent 30%),
            linear-gradient(165deg, #060b14 0%, #0b1220 45%, #0e1629 100%);
        color: #f2f7ff;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        animation: fadeInUp 0.55s ease-out;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1322 0%, #111b2f 100%);
        border-right: 1px solid rgba(88, 166, 255, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #f2f7ff !important;
        font-size: 1.08rem !important;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(160deg, rgba(20, 31, 52, 0.92), rgba(10, 18, 30, 0.92));
        border: 1px solid rgba(88, 166, 255, 0.30);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(10, 132, 255, 0.22);
        border-color: rgba(88, 166, 255, 0.65);
    }

    div[data-testid="metric-container"] label {
        color: #9cc8ff !important;
        font-weight: 700 !important;
        font-size: 0.96rem !important;
        letter-spacing: 0.01em;
    }

    div[data-testid="metric-container"] div {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
    }

    /* ── Headers ── */
    h1, h2, h3 {
        color: #86beff !important;
        letter-spacing: 0.01em;
        text-shadow: 0 0 24px rgba(88, 166, 255, 0.25);
    }

    p, li, label, span {
        color: #e7f1ff !important;
    }

    /* ── Sidebar radio ── */
    div[role="radiogroup"] > label {
        background: rgba(88, 166, 255, 0.10);
        border: 1px solid rgba(88, 166, 255, 0.30);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 8px;
        transition: all 0.25s ease;
    }

    div[role="radiogroup"] > label:hover {
        border-color: rgba(88, 166, 255, 0.72);
        background: rgba(88, 166, 255, 0.18);
        transform: translateX(2px);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(120deg, #238636, #2ea043);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 700;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(46, 160, 67, 0.35);
    }

    /* ── Sliders ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #7ec8ff);
    }

    /* ── Dataframes ── */
    .stDataFrame {
        background: rgba(15, 24, 40, 0.92);
        border: 1px solid rgba(88, 166, 255, 0.28);
        border-radius: 12px;
    }

    /* ── Divider ── */
    hr {border-color: rgba(88, 166, 255, 0.22);}

    /* ── Small motion effects ── */
    .stPlotlyChart {
        animation: softPulse 3s ease-in-out infinite;
    }

    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @keyframes softPulse {
        0%, 100% {filter: drop-shadow(0 0 0 rgba(88, 166, 255, 0.00));}
        50% {filter: drop-shadow(0 0 8px rgba(88, 166, 255, 0.16));}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state — train model once per session
# ---------------------------------------------------------------------------
if "model" not in st.session_state:
    with st.spinner("Training AI model on satellite data..."):
        model, metrics, importance_df = train_model()
        st.session_state["model"] = model
        st.session_state["metrics"] = metrics
        st.session_state["importance_df"] = importance_df
        st.session_state["training_data"] = generate_satellite_data()

model = st.session_state["model"]
metrics = st.session_state["metrics"]
importance_df = st.session_state["importance_df"]
training_data: pd.DataFrame = st.session_state["training_data"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select Page",
        [
            "Overview",
            "Live Satellite Monitoring",
            "Disaster Prediction",
            "Analytics",
            "Model Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown("Satellite Link: `ONLINE`")
    st.markdown("Model Status: `ACTIVE`")
    st.markdown("Data Feed: `LIVE`")

# ===========================================================================
# PAGE 1 — Overview
# ===========================================================================
if page == "Overview":
    st.title("Overview Dashboard")
    st.markdown("Real-time satellite monitoring and disaster risk assessment.")
    st.markdown("---")

    # Live reading for overview metrics
    live = get_live_reading()
    score = predict_risk(model, live)
    level = risk_level(score)

    level_color = {"Safe": "#3fb950", "Warning": "#d29922", "Critical": "#f85149"}[level]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Disaster Risk Score", f"{score:.1f} / 100")
    col2.metric("Disaster Level", level)
    col3.metric("Active Satellite Feeds", "6")
    col4.metric("Active Alerts", "2" if level == "Critical" else "1" if level == "Warning" else "0")

    st.markdown("---")

    # Risk gauge
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "Current Risk Score", "font": {"color": "#c9d1d9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": level_color},
                "steps": [
                    {"range": [0, 35], "color": "#0d2818"},
                    {"range": [35, 65], "color": "#2d2008"},
                    {"range": [65, 100], "color": "#300a05"},
                ],
                "threshold": {
                    "line": {"color": "#f85149", "width": 3},
                    "thickness": 0.75,
                    "value": 65,
                },
            },
            number={"font": {"color": "#c9d1d9"}},
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
        height=320,
    )

    col_g, col_s = st.columns([1, 1])
    with col_g:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_s:
        st.markdown("### Current Sensor Readings")
        sensor_df = pd.DataFrame(
            {
                "Sensor": [
                    "Temperature (°C)",
                    "Rainfall (mm)",
                    "Soil Moisture (%)",
                    "Seismic Activity",
                    "Water Level (m)",
                ],
                "Value": [
                    f"{live['temperature']:.2f}",
                    f"{live['rainfall']:.2f}",
                    f"{live['soil_moisture']:.2f}",
                    f"{live['seismic_activity']:.2f}",
                    f"{live['water_level']:.2f}",
                ],
            }
        )
        st.dataframe(sensor_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"<h3 style='color:{level_color};text-align:center;margin-top:16px;'>"
            f"Status: {level.upper()}</h3>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# PAGE 2 — Live Satellite Monitoring
# ===========================================================================
elif page == "Live Satellite Monitoring":
    st.title("Live Satellite Feed")
    st.markdown("Simulated real-time sensor telemetry from orbital satellite array.")
    st.markdown("---")

    n_points = st.slider("History window (readings)", 20, 100, 40)

    rng = np.random.default_rng(int(time.time()) % 86400)  # cycle within a day
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq="1min")

    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": rng.uniform(10, 55, n_points),
            "rainfall": rng.uniform(0, 300, n_points),
            "soil_moisture": rng.uniform(5, 95, n_points),
            "seismic_activity": rng.uniform(0, 9, n_points),
            "water_level": rng.uniform(0, 15, n_points),
        }
    )

    PLOT_CFG = dict(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#c9d1d9")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            history,
            x="timestamp",
            y="temperature",
            title="Temperature Over Time (°C)",
            color_discrete_sequence=["#f85149"],
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            history,
            x="timestamp",
            y="soil_moisture",
            title="Soil Moisture Over Time (%)",
            color_discrete_sequence=["#3fb950"],
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            history,
            x="timestamp",
            y="rainfall",
            title="Rainfall Over Time (mm)",
            color_discrete_sequence=["#58a6ff"],
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            history,
            x="timestamp",
            y="seismic_activity",
            title="Seismic Activity Over Time",
            color_discrete_sequence=["#d29922"],
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Water Level Monitor (m)")
    fig = px.area(
        history,
        x="timestamp",
        y="water_level",
        title="Water Level Over Time (m)",
        color_discrete_sequence=["#58a6ff"],
    )
    fig.update_layout(**PLOT_CFG)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE 3 — Disaster Prediction
# ===========================================================================
elif page == "Disaster Prediction":
    st.title("Disaster Prediction Engine")
    st.markdown(
        "Adjust environmental parameters below. "
        "The AI model computes a real-time disaster risk score."
    )
    st.markdown("---")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### Input Parameters")
        temp = st.slider("Temperature (°C)", 10.0, 55.0, 30.0, 0.1)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0, 1.0)
        moisture = st.slider("Soil Moisture (%)", 5.0, 95.0, 50.0, 0.5)
        seismic = st.slider("Seismic Activity", 0.0, 9.0, 3.0, 0.1)
        water = st.slider("Water Level (m)", 0.0, 15.0, 5.0, 0.1)

    inputs = {
        "temperature": temp,
        "rainfall": rain,
        "soil_moisture": moisture,
        "seismic_activity": seismic,
        "water_level": water,
    }
    score = predict_risk(model, inputs)
    level = risk_level(score)
    level_color = {"Safe": "#3fb950", "Warning": "#d29922", "Critical": "#f85149"}[level]

    with col_result:
        st.markdown("### Prediction Result")

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Disaster Risk Score", "font": {"color": "#c9d1d9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": level_color},
                    "steps": [
                        {"range": [0, 35], "color": "#0d2818"},
                        {"range": [35, 65], "color": "#2d2008"},
                        {"range": [65, 100], "color": "#300a05"},
                    ],
                },
                number={"font": {"color": "#c9d1d9"}},
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"<h2 style='color:{level_color};text-align:center;'>{level.upper()}</h2>",
            unsafe_allow_html=True,
        )

        # Sub-risk coefficients (scaled so max inputs produce ~100)
        # flood: rainfall 0.33/mm + water level 6.67/m → max ~100
        # heatwave: each °C above 35 adds 5 points → max 100 at 55 °C
        # earthquake: 0–9 scale × 11.1 → max ~100
        # infrastructure: weighted combination of moisture, seismic, water level
        _FLOOD_RAIN, _FLOOD_WATER = 0.3, 5.0
        _HEAT_THRESH, _HEAT_SCALE = 35.0, 5.0
        _QUAKE_SCALE = 11.0
        _INFRA_MOIST, _INFRA_SEIS, _INFRA_WATER = 0.4, 6.0, 3.0

        flood_risk = min(100, rain * _FLOOD_RAIN + water * _FLOOD_WATER)
        heat_risk = min(100, max(0, (temp - _HEAT_THRESH) * _HEAT_SCALE))
        quake_risk = min(100, seismic * _QUAKE_SCALE)
        infra_risk = min(100, moisture * _INFRA_MOIST + seismic * _INFRA_SEIS + water * _INFRA_WATER)

        st.markdown("### Risk Breakdown")
        breakdown = pd.DataFrame(
            {
                "Risk Type": ["Flood", "Heatwave", "Earthquake", "Infrastructure Damage"],
                "Score": [flood_risk, heat_risk, quake_risk, infra_risk],
            }
        )
        fig_bar = px.bar(
            breakdown,
            x="Risk Type",
            y="Score",
            color="Score",
            color_continuous_scale=["#3fb950", "#d29922", "#f85149"],
            range_color=[0, 100],
            title="Disaster Sub-Risk Scores",
        )
        fig_bar.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font_color="#c9d1d9",
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ===========================================================================
# PAGE 4 — Analytics
# ===========================================================================
elif page == "Analytics":
    st.title("Analytics")
    st.markdown("Statistical analysis of satellite training data and risk trends.")
    st.markdown("---")

    PLOT_CFG = dict(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#c9d1d9")

    # Risk score distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            training_data,
            x="disaster_risk_score",
            nbins=40,
            title="Risk Score Distribution",
            color_discrete_sequence=["#58a6ff"],
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Disaster level pie
        training_data["level"] = training_data["disaster_risk_score"].apply(risk_level)
        level_counts = training_data["level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        fig = px.pie(
            level_counts,
            names="Level",
            values="Count",
            title="Disaster Level Distribution",
            color="Level",
            color_discrete_map={
                "Safe": "#3fb950",
                "Warning": "#d29922",
                "Critical": "#f85149",
            },
        )
        fig.update_layout(paper_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    corr = training_data[FEATURE_COLS + ["disaster_risk_score"]].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    fig_heat.update_layout(**PLOT_CFG, height=480)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Scatter — temperature vs risk
    st.markdown("### Risk Trends")
    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(
            training_data,
            x="temperature",
            y="disaster_risk_score",
            color="disaster_risk_score",
            color_continuous_scale=["#3fb950", "#d29922", "#f85149"],
            title="Temperature vs Risk Score",
            opacity=0.6,
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            training_data,
            x="seismic_activity",
            y="disaster_risk_score",
            color="disaster_risk_score",
            color_continuous_scale=["#3fb950", "#d29922", "#f85149"],
            title="Seismic Activity vs Risk Score",
            opacity=0.6,
        )
        fig.update_layout(**PLOT_CFG)
        st.plotly_chart(fig, use_container_width=True)

    # Disaster probability over simulated time series
    st.markdown("### Simulated Risk Trend Over Time")
    n_ts = 200
    ts_rng = np.random.default_rng(99)
    ts_df = pd.DataFrame(
        {
            "time": range(n_ts),
            "risk": np.clip(
                50 + ts_rng.normal(0, 15, n_ts).cumsum() * 0.05
                + ts_rng.normal(0, 8, n_ts),
                0,
                100,
            ),
        }
    )
    fig_ts = px.line(
        ts_df,
        x="time",
        y="risk",
        title="Disaster Risk Score Over Time (Simulated)",
        color_discrete_sequence=["#58a6ff"],
    )
    fig_ts.add_hline(y=35, line_dash="dash", line_color="#3fb950", annotation_text="Safe Threshold")
    fig_ts.add_hline(y=65, line_dash="dash", line_color="#f85149", annotation_text="Critical Threshold")
    fig_ts.update_layout(**PLOT_CFG, height=360)
    st.plotly_chart(fig_ts, use_container_width=True)

# ===========================================================================
# PAGE 5 — Model Insights
# ===========================================================================
elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown("Transparency report for the deployed AI prediction model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "RandomForestRegressor")
    col2.metric("R² Score", f"{metrics['r2_score']:.4f}")
    col3.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")

    st.markdown("---")

    PLOT_CFG = dict(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#c9d1d9")

    # Feature importance
    col_fi, col_info = st.columns([1, 1])
    with col_fi:
        st.markdown("### Feature Importance")
        fig_fi = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#161b22", "#58a6ff"],
            title="Feature Importance (RandomForest)",
        )
        fig_fi.update_layout(**PLOT_CFG, height=360, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_info:
        st.markdown("### Model Configuration")
        config = {
            "Estimators": 100,
            "Max Features": "auto (sqrt)",
            "Bootstrap": True,
            "Train / Test Split": "80 / 20",
            "Training Samples": 800,
            "Test Samples": 200,
            "Random State": 42,
        }
        config_df = pd.DataFrame(
            {"Parameter": config.keys(), "Value": config.values()}
        )
        st.dataframe(config_df, use_container_width=True, hide_index=True)

        st.markdown("### Input Features")
        features_info = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Unit": ["°C", "mm", "%", "Richter-like scale", "m"],
                "Range": ["10 – 55", "0 – 300", "5 – 95", "0 – 9", "0 – 15"],
            }
        )
        st.dataframe(features_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### How the Model Works")
    st.markdown(
        """
        The **RandomForestRegressor** is an ensemble learning method that builds multiple
        decision trees during training and outputs the average prediction of the individual trees.

        - **Input**: Five satellite sensor readings (temperature, rainfall, soil moisture,
          seismic activity, water level).
        - **Output**: A continuous disaster risk score in the range 0–100.
        - **Risk Levels**: Safe (< 35), Warning (35–65), Critical (> 65).
        - **Training data**: 1,000 synthetic samples generated with a physics-inspired
          weighting formula plus Gaussian noise, ensuring the model generalises to
          unseen sensor combinations.
        """
    )
