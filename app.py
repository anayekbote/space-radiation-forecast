import pickle
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.inference import SpaceRadiationForecaster

st.set_page_config(
    page_title="Space Radiation Forecaster",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #38bdf8;
    }
    .hazard-safe { color: #4ade80; font-weight: bold; }
    .hazard-elevated { color: #facc15; font-weight: bold; }
    .hazard-severe { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_engine():
    return SpaceRadiationForecaster()

@st.cache_data
def load_datasets():
    with open(PROCESSED_DATA_DIR / "train_test_split.pkl", "rb") as f:
        split_data = pickle.load(f)
    df_test = split_data['df_test']
    
    isro_path = PROCESSED_DATA_DIR / "isro_gsat19_grasp_2018_5min.parquet"
    df_isro = pd.read_parquet(isro_path) if isro_path.exists() else None
    
    return df_test, df_isro

forecaster = load_engine()
df_test, df_isro = load_datasets()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🛰️ Mission Control")
st.sidebar.markdown("**Multi-Mission Space Radiation Forecaster**")
st.sidebar.caption("NASA WIND • NOAA GOES-15 • ISRO GSAT-19")

mode = st.sidebar.radio("Navigation", ["Forecast & Time Series", "Mission Cross-Validation", "Model Driver Attribution"])

# ---------------------------------------------------------
# Tab 1: Forecast & Time Series
# ---------------------------------------------------------
if mode == "Forecast & Time Series":
    st.title("⚡ Operational Relativistic Electron Flux Forecast")
    
    col1, col2, col3 = st.columns(3)
    latest_ts = df_test.index[-1]
    latest_sample = df_test.iloc[[-1]]
    latest_preds = forecaster.predict(latest_sample)
    
    flux_1h = latest_preds['flux_lead_1h'].iloc[0]
    flux_6h = latest_preds['flux_lead_6h'].iloc[0]
    flux_24h = latest_preds['flux_lead_24h'].iloc[0]
    
    def get_status(val):
        if val < 100:
            return "🟢 SAFE", "hazard-safe"
        elif val < 1000:
            return "🟡 ELEVATED", "hazard-elevated"
        else:
            return "🔴 SEVERE STORM", "hazard-severe"

    with col1:
        status_text, cls = get_status(flux_1h)
        st.metric("+1 Hour Lead", f"{flux_1h:.2f} cm⁻²s⁻¹sr⁻¹", f"Status: {status_text}")
    with col2:
        status_text, cls = get_status(flux_6h)
        st.metric("+6 Hours Lead", f"{flux_6h:.2f} cm⁻²s⁻¹sr⁻¹", f"Status: {status_text}")
    with col3:
        status_text, cls = get_status(flux_24h)
        st.metric("+24 Hours Lead", f"{flux_24h:.2f} cm⁻²s⁻¹sr⁻¹", f"Status: {status_text}")

    st.markdown("---")
    st.subheader("Interactive Multi-Horizon Evaluation Window (2016 Out-of-Sample Test Set)")
    
    window_days = st.slider("Select Forecast Window (Days)", min_value=3, max_value=30, value=7)
    steps = window_days * 288 # 288 five-minute steps per day
    df_window = df_test.iloc[1000:1000 + steps]
    window_preds = forecaster.predict(df_window)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_window.index, y=df_window['target_log_flux'],
        mode='lines', name='GOES-15 Observed (log₁₀ Flux)',
        line=dict(color='#94a3b8', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_window.index, y=window_preds['log_flux_lead_1h'],
        mode='lines', name='Forecast (+1h)',
        line=dict(color='#38bdf8', width=2, dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=df_window.index, y=window_preds['log_flux_lead_24h'],
        mode='lines', name='Forecast (+24h)',
        line=dict(color='#f87171', width=2, dash='dot')
    ))
    
    fig.add_hline(y=3.0, line_dash="dash", line_color="#ef4444", annotation_text="Hazard Threshold (10³ cm⁻²s⁻¹sr⁻¹)")
    fig.update_layout(
        template="plotly_dark",
        height=480,
        xaxis_title="Time (UTC)",
        yaxis_title="log₁₀ [Flux (cm⁻² s⁻¹ sr⁻¹)]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Mission Cross-Validation
# ---------------------------------------------------------
elif mode == "Mission Cross-Validation":
    st.title("🛰️ Multi-Mission Cross-Calibration & Transfer")
    st.markdown("Comparison between **NOAA GOES-15 ($135^\\circ\\text{ W}$)** and **ISRO GSAT-19 GRASP ($48^\\circ\\text{ E}$)**.")
    
    if df_isro is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total ISRO Telemetry Rows", f"{len(df_isro):,}")
        with col_b:
            st.metric("Max Recorded ISRO Flux", f"{df_isro['isro_electron_flux'].max():.2f} counts/s")
            
        fig_isro = go.Figure()
        sample_isro = df_isro.iloc[:2016] # 1 week sample
        fig_isro.add_trace(go.Scatter(
            x=sample_isro.index, y=sample_isro['log_isro_electron_flux'],
            mode='lines', name='ISRO GSAT-19 GRASP L2 (log₁₀ Flux)',
            line=dict(color='#f59e0b', width=1.5)
        ))
        fig_isro.update_layout(
            template="plotly_dark",
            height=400,
            title="ISRO GSAT-19 GRASP Electron Observation Profile (April 2018)",
            xaxis_title="Time (UTC)",
            yaxis_title="log₁₀ [Flux]"
        )
        st.plotly_chart(fig_isro, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Model Driver Attribution
# ---------------------------------------------------------
elif mode == "Model Driver Attribution":
    st.title("🧠 Physical Driver Attribution & Feature Importance")
    st.markdown("Derived feature influence for relativistic electron acceleration models.")
    
    h_selected = st.selectbox("Select Horizon", ["1h", "6h", "24h"], index=2)
    feat_df = pd.read_csv(MODELS_DIR / f"feature_importance_{h_selected}.csv").head(12)
    
    fig_imp = go.Figure(go.Bar(
        x=feat_df['importance'][::-1],
        y=feat_df['feature'][::-1],
        orientation='h',
        marker=dict(color='#38bdf8')
    ))
    fig_imp.update_layout(
        template="plotly_dark",
        height=450,
        title=f"Top Physical Drivers for +{h_selected} Lead Horizon",
        xaxis_title="Normalized Gini Importance",
        yaxis_title="Feature Name"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
