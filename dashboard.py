"""
dashboard.py
============
EY – EV Market Demand Forecasting
----------------------------------
Streamlit interactive dashboard.

Run:  streamlit run dashboard.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title = "EY | EV Demand Forecasting",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #C9D1D9;
}

/* Header strip */
.ey-header {
    background: linear-gradient(135deg, #FFE600 0%, #F5C800 100%);
    padding: 18px 32px;
    border-radius: 10px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.ey-header h1 { color: #0D1117; font-size: 1.6rem; margin: 0; font-weight: 700; }
.ey-header p  { color: #1A1A1A; margin: 0; font-size: 0.9rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #161B22, #21262D);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.metric-label { font-size: 0.8rem; color: #8B949E; letter-spacing: 0.05em; text-transform: uppercase; }
.metric-value { font-size: 2rem; font-weight: 700; color: #FFE600; margin-top: 4px; }
.metric-unit  { font-size: 0.75rem; color: #6E7681; margin-top: 2px; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E6EDF3;
    border-left: 3px solid #FFE600;
    padding-left: 10px;
    margin: 28px 0 12px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
ART_DIR   = BASE / "artefacts"


# ── Helper: load artefacts ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    svr = joblib.load(ART_DIR / "svr_model.pkl")
    xgb = joblib.load(ART_DIR / "xgb_model.pkl")
    ens = joblib.load(ART_DIR / "ensemble_model.pkl")
    return svr, xgb, ens

@st.cache_data(show_spinner=False)
def load_data():
    raw   = pd.read_csv(ART_DIR / "raw_dataset.csv",    parse_dates=["date"])
    feat  = pd.read_csv(ART_DIR / "feature_dataset.csv", parse_dates=["date"])
    preds = pd.read_csv(ART_DIR / "test_predictions.csv", parse_dates=["date"])
    comp  = pd.read_csv(ART_DIR / "model_comparison.csv")
    return raw, feat, preds, comp

# ── Check artefacts exist ─────────────────────────────────────────────────
artefacts_ready = all([
    (ART_DIR / f).exists()
    for f in ["svr_model.pkl", "xgb_model.pkl", "ensemble_model.pkl",
              "raw_dataset.csv", "feature_dataset.csv",
              "test_predictions.csv", "model_comparison.csv"]
])

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ey-header">
  <div>
    <h1>⚡ EV Market Demand Forecasting</h1>
    <p>Ernst &amp; Young &nbsp;|&nbsp; XGBoost + SVR Ensemble &nbsp;|&nbsp; 
       Data: U.S. Energy Information Administration (EIA)</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/EY_logo_2019.svg/320px-EY_logo_2019.svg.png",
             width=120)
    st.markdown("---")
    st.markdown("### ⚙️ Dashboard Controls")

    selected_model = st.selectbox(
        "Primary Forecast Model",
        ["Ensemble", "XGBoost", "SVR"],
        index=0,
    )

    date_range = st.slider(
        "Date Range (days from end)",
        min_value=30, max_value=730, value=365, step=30,
    )

    show_all_models = st.checkbox("Overlay All Models", value=True)
    show_raw_data   = st.checkbox("Show Raw Data Table", value=False)

    st.markdown("---")
    st.markdown("### 📚 References")
    st.caption("• EIA Weekly Gasoline Prices")
    st.caption("• EIA Retail Electricity Prices")
    st.caption("• SVR: Smola & Schölkopf (2004)")
    st.caption("• XGBoost: Chen & Guestrin (2016)")
    st.caption("• Ensemble EV Forecasting: Almaghrebi et al. (2022)")

    st.markdown("---")
    st.caption("EY Internship | Advanced Data Science | 2024-25")

# ── Body ───────────────────────────────────────────────────────────────────
if not artefacts_ready:
    st.warning(
        "⚠️  Model artefacts not found. "
        "Please run `python main.py` first to train the models, "
        "then refresh this page."
    )
    st.code("python main.py --fast    # quick run (no tuning)\npython main.py       # full run with hyper-parameter tuning")
    st.stop()

# Load data
with st.spinner("Loading models & data …"):
    svr_model, xgb_model, ens_model = load_models()
    raw, feat, preds, comp = load_data()

# Slice by date range
preds_slice = preds.tail(date_range).copy()

# ── KPI Row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Performance Metrics — Out-of-Sample Test Set</div>',
            unsafe_allow_html=True)

best_row = comp[comp["Model"] == "Ensemble"].squeeze()
if isinstance(best_row, pd.Series) and len(best_row) > 0:
    mae_val  = best_row.get("MAE",  0)
    rmse_val = best_row.get("RMSE", 0)
    mape_val = best_row.get("MAPE", 0)
    r2_val   = best_row.get("R2",   0)
else:
    mae_val = rmse_val = mape_val = r2_val = 0

c1, c2, c3, c4 = st.columns(4)
for col, label, value, unit in [
    (c1, "MAE",  f"{mae_val:.1f}",  "sessions / day"),
    (c2, "RMSE", f"{rmse_val:.1f}", "sessions / day"),
    (c3, "MAPE", f"{mape_val:.2f}%", "error"),
    (c4, "R²",   f"{r2_val:.4f}",   "coefficient"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Forecast Chart ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 EV Charging Demand Forecast</div>',
            unsafe_allow_html=True)

COLOUR = dict(
    actual="#4CAF50", svr="#2196F3", xgb="#FF9800", ensemble="#E91E63",
    bg="#0D1117", grid="#21262D", text="#C9D1D9",
)
LAYOUT = dict(
    paper_bgcolor=COLOUR["bg"], plot_bgcolor=COLOUR["bg"],
    font=dict(color=COLOUR["text"], family="Inter, sans-serif", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=preds_slice["date"], y=preds_slice["actual"],
    name="Actual", line=dict(color=COLOUR["actual"], width=2), opacity=0.85,
))

if show_all_models or selected_model == "SVR":
    fig_fc.add_trace(go.Scatter(
        x=preds_slice["date"], y=preds_slice["svr_pred"],
        name="SVR", line=dict(color=COLOUR["svr"], width=1.8, dash="dot"),
    ))
if show_all_models or selected_model == "XGBoost":
    fig_fc.add_trace(go.Scatter(
        x=preds_slice["date"], y=preds_slice["xgb_pred"],
        name="XGBoost", line=dict(color=COLOUR["xgb"], width=1.8, dash="dot"),
    ))
if show_all_models or selected_model == "Ensemble":
    fig_fc.add_trace(go.Scatter(
        x=preds_slice["date"], y=preds_slice["ensemble_pred"],
        name="Ensemble", line=dict(color=COLOUR["ensemble"], width=2.5),
    ))

fig_fc.update_layout(
    **LAYOUT,
    xaxis=dict(title="Date",            gridcolor=COLOUR["grid"]),
    yaxis=dict(title="Sessions / Day",  gridcolor=COLOUR["grid"]),
    height=450,
)
st.plotly_chart(fig_fc, use_container_width=True)

# ── Feature Importance + Model Comparison ─────────────────────────────────
st.markdown('<div class="section-header">🔍 Explainability & Model Comparison</div>',
            unsafe_allow_html=True)

col_fi, col_mc = st.columns([2, 1])

with col_fi:
    imp      = xgb_model.feature_importances().head(15).sort_values()
    fig_imp  = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker=dict(color=imp.values, colorscale="Plasma", showscale=False),
    ))
    fig_imp.update_layout(
        **LAYOUT,
        title=dict(text="Top-15 Feature Importances (XGBoost – Gain)",
                   x=0.02, font=dict(size=14)),
        xaxis=dict(gridcolor=COLOUR["grid"]),
        height=420,
        margin=dict(l=160),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col_mc:
    fig_mc = go.Figure()
    for metric, col_hex in [("MAE", COLOUR["svr"]), ("RMSE", COLOUR["xgb"])]:
        fig_mc.add_trace(go.Bar(
            name=metric,
            x=comp["Model"],
            y=comp[metric],
            marker_color=col_hex,
            text=comp[metric].round(1),
            textposition="outside",
        ))
    fig_mc.update_layout(
        **LAYOUT,
        barmode="group",
        title=dict(text="MAE & RMSE by Model", x=0.02, font=dict(size=14)),
        yaxis=dict(title="Sessions / Day", gridcolor=COLOUR["grid"]),
        height=420,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# ── Gasoline Price vs EV Demand ────────────────────────────────────────────
st.markdown('<div class="section-header">⛽ Gasoline Price vs EV Charging Demand</div>',
            unsafe_allow_html=True)

from plotly.subplots import make_subplots
fig_gd = make_subplots(specs=[[{"secondary_y": True}]])
fig_gd.add_trace(go.Scatter(
    x=raw["date"], y=raw["gas_price_usd_gal"],
    name="Gasoline Price ($/gal)", line=dict(color=COLOUR["xgb"], width=1.5)
), secondary_y=False)
fig_gd.add_trace(go.Scatter(
    x=raw["date"], y=raw["ev_demand_sessions"],
    name="EV Demand (sessions/day)", line=dict(color=COLOUR["actual"], width=1.5)
), secondary_y=True)
fig_gd.update_layout(**LAYOUT, height=380)
fig_gd.update_yaxes(title_text="USD / Gallon",    gridcolor=COLOUR["grid"], secondary_y=False)
fig_gd.update_yaxes(title_text="Sessions / Day",  gridcolor=COLOUR["grid"], secondary_y=True)
fig_gd.update_xaxes(gridcolor=COLOUR["grid"])
st.plotly_chart(fig_gd, use_container_width=True)

# ── Residual Analysis ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">📉 Residual Analysis</div>',
            unsafe_allow_html=True)

residuals = preds["actual"].values - preds["ensemble_pred"].values
col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_res = go.Figure(go.Scatter(
        x=preds["date"], y=residuals,
        mode="lines",
        line=dict(color="#9C27B0", width=1),
        name="Residual",
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#E91E63")
    fig_res.update_layout(**LAYOUT,
                           title="Residuals Over Time",
                           yaxis_title="Error (sessions/day)",
                           xaxis=dict(gridcolor=COLOUR["grid"]),
                           yaxis=dict(gridcolor=COLOUR["grid"]),
                           height=340)
    st.plotly_chart(fig_res, use_container_width=True)

with col_r2:
    fig_hist = go.Figure(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color="#9C27B0", opacity=0.8,
    ))
    fig_hist.update_layout(**LAYOUT,
                            title="Residual Distribution",
                            xaxis_title="Residual (sessions/day)",
                            yaxis_title="Count",
                            xaxis=dict(gridcolor=COLOUR["grid"]),
                            yaxis=dict(gridcolor=COLOUR["grid"]),
                            height=340)
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Raw Data Table ─────────────────────────────────────────────────────────
if show_raw_data:
    st.markdown('<div class="section-header">📋 Raw Dataset</div>', unsafe_allow_html=True)
    st.dataframe(raw.tail(200), use_container_width=True, height=350)
    csv_bytes = raw.to_csv(index=False).encode()
    st.download_button(
        label     = "⬇️  Download Full Dataset (CSV)",
        data      = csv_bytes,
        file_name = "ey_ev_demand_dataset.csv",
        mime      = "text/csv",
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>EY Internship · EV Market Demand ML Forecasting · "
    "XGBoost + SVR Ensemble · Data Source: U.S. EIA</small></center>",
    unsafe_allow_html=True,
)
