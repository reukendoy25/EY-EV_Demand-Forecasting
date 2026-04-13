"""
visualization.py
================
EY – EV Market Demand Forecasting
----------------------------------
Plotly-based interactive charts reused by both main.py and dashboard.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional


# ── Shared colour palette ─────────────────────────────────────────────────────
COLOUR = {
    "actual":   "#4CAF50",
    "svr":      "#2196F3",
    "xgb":      "#FF9800",
    "ensemble": "#E91E63",
    "error":    "#9C27B0",
    "bg":       "#0D1117",
    "grid":     "#21262D",
    "text":     "#C9D1D9",
}

LAYOUT_BASE = dict(
    paper_bgcolor = COLOUR["bg"],
    plot_bgcolor  = COLOUR["bg"],
    font          = dict(color=COLOUR["text"], family="Inter, sans-serif", size=13),
    legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hovermode     = "x unified",
)


def forecast_chart(
    dates,
    y_actual,
    preds: Dict[str, np.ndarray],
    title: str = "EV Charging Demand Forecast",
) -> go.Figure:
    """
    Multi-line forecast chart overlaying actual vs model predictions.

    Parameters
    ----------
    dates    : array-like of datetime
    y_actual : array-like – true values
    preds    : dict  {label: array}  e.g. {"SVR": ..., "XGBoost": ..., "Ensemble": ...}
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=y_actual,
        name="Actual",
        line=dict(color=COLOUR["actual"], width=2),
        opacity=0.85,
    ))

    colours = [COLOUR["svr"], COLOUR["xgb"], COLOUR["ensemble"]]
    for (label, pred), col in zip(preds.items(), colours):
        fig.add_trace(go.Scatter(
            x=dates, y=pred,
            name=label,
            line=dict(color=col, width=2, dash="dot"),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, x=0.04, font=dict(size=18, color=COLOUR["text"])),
        xaxis=dict(title="Date",                  gridcolor=COLOUR["grid"]),
        yaxis=dict(title="Sessions / Day",        gridcolor=COLOUR["grid"]),
        height=480,
    )
    return fig


def feature_importance_chart(importance: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances (top N)."""
    imp = importance.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        marker=dict(
            color=imp.values,
            colorscale="Plasma",
            showscale=False,
        ),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Top-{top_n} Feature Importances (XGBoost – Gain)",
                   x=0.04, font=dict(size=16)),
        xaxis=dict(title="Importance Score", gridcolor=COLOUR["grid"]),
        yaxis=dict(tickfont=dict(size=11)),
        height=520,
        margin=dict(l=180),
    )
    return fig


def residual_chart(dates, y_actual, y_pred, label: str = "Ensemble") -> go.Figure:
    """Dual-panel: actual vs predicted + residual distribution."""
    residuals = np.asarray(y_actual) - np.asarray(y_pred)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Actual vs Predicted", "Residuals Distribution"),
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=dates, y=y_actual, name="Actual",
                             line=dict(color=COLOUR["actual"], width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name=label,
                             line=dict(color=COLOUR["ensemble"], width=1.5, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=60, name="Residuals",
                               marker_color=COLOUR["error"], opacity=0.75),
                  row=2, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{label} – Residual Analysis", x=0.04, font=dict(size=16)),
        height=680,
        showlegend=True,
    )
    fig.update_xaxes(gridcolor=COLOUR["grid"])
    fig.update_yaxes(gridcolor=COLOUR["grid"])
    return fig


def metrics_bar_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing MAE and RMSE across models."""
    fig = go.Figure()
    for metric, col in [("MAE", COLOUR["svr"]), ("RMSE", COLOUR["xgb"])]:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df["Model"],
            y=comparison_df[metric],
            marker_color=col,
            text=comparison_df[metric].round(1),
            textposition="outside",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        title=dict(text="Model Comparison – MAE & RMSE", x=0.04, font=dict(size=16)),
        yaxis=dict(title="Sessions / Day", gridcolor=COLOUR["grid"]),
        height=400,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: Optional[list] = None) -> go.Figure:
    """Plotly heatmap of feature correlations."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()[:18]
    corr = df[cols].corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Feature Correlation Heatmap", x=0.04, font=dict(size=16)),
        height=580,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def gasoline_vs_demand(
    df: pd.DataFrame,
    date_col: str = "date",
    gas_col: str  = "gas_price_usd_gal",
    dem_col: str  = "ev_demand_sessions",
) -> go.Figure:
    """Dual-axis chart: gasoline price vs EV demand."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[gas_col],
        name="Gasoline Price ($/gal)",
        line=dict(color=COLOUR["xgb"], width=1.5)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[dem_col],
        name="EV Demand (sessions/day)",
        line=dict(color=COLOUR["actual"], width=1.5)
    ), secondary_y=True)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Gasoline Price vs EV Charging Demand",
                   x=0.04, font=dict(size=16)),
        height=420,
    )
    fig.update_yaxes(title_text="USD / Gallon",       gridcolor=COLOUR["grid"], secondary_y=False)
    fig.update_yaxes(title_text="Sessions / Day",     gridcolor=COLOUR["grid"], secondary_y=True)
    fig.update_xaxes(gridcolor=COLOUR["grid"])
    return fig
