"""
feature_engineering.py
=======================
EY – EV Market Demand Forecasting
----------------------------------
Transforms the raw daily dataset into a supervised-learning feature matrix:
  • Lag features         – temporal memory for ML models
  • Rolling statistics   – trend + volatility signals
  • Cyclical encodings   – day-of-week, month, quarter
  • Price differential   – economic driver of EV adoption
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


# ── Configuration ─────────────────────────────────────────────────────────────
LAG_DAYS_DEMAND   = [1, 2, 3, 7, 14, 30]
LAG_DAYS_SAVINGS  = [1, 3, 7, 14]
ROLLING_WINDOWS   = [7, 30]
TARGET_COL        = "ev_demand_sessions"


def _cyclical_encode(series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """Encode a periodic variable as (sin, cos) pair."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Output of data_pipeline.build_daily_dataset().

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (NaN rows from lags are dropped).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── 1. Lag features: EV demand ──────────────────────────────────────────
    for lag in LAG_DAYS_DEMAND:
        df[f"demand_lag_{lag}"] = df[TARGET_COL].shift(lag)

    # ── 2. Lag features: fuel savings ───────────────────────────────────────
    for lag in LAG_DAYS_SAVINGS:
        df[f"fuel_savings_lag_{lag}"] = df["fuel_savings_usd_day"].shift(lag)

    # ── 3. Lag features: gas price ──────────────────────────────────────────
    for lag in [1, 7]:
        df[f"gas_price_lag_{lag}"] = df["gas_price_usd_gal"].shift(lag)

    # ── 4. Rolling statistics for EV demand ─────────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"demand_roll_mean_{w}"] = (
            df[TARGET_COL].shift(1).rolling(window=w).mean()
        )
        df[f"demand_roll_std_{w}"]  = (
            df[TARGET_COL].shift(1).rolling(window=w).std()
        )

    # ── 5. Rolling statistics for GHG savings ───────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"ghg_roll_mean_{w}"] = (
            df["ghg_savings_kg_day"].shift(1).rolling(window=w).mean()
        )

    # ── 6. Rolling statistics for fuel savings ───────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"fuel_savings_roll_mean_{w}"] = (
            df["fuel_savings_usd_day"].shift(1).rolling(window=w).mean()
        )
        df[f"fuel_savings_roll_std_{w}"]  = (
            df["fuel_savings_usd_day"].shift(1).rolling(window=w).std()
        )

    # ── 7. Cyclical time encodings ────────────────────────────────────────
    df["dow_sin"], df["dow_cos"]     = _cyclical_encode(df["date"].dt.dayofweek,   7)
    df["month_sin"], df["month_cos"] = _cyclical_encode(df["date"].dt.month,       12)
    df["doy_sin"], df["doy_cos"]     = _cyclical_encode(df["date"].dt.dayofyear,  365)
    df["quarter"] = df["date"].dt.quarter

    # ── 8. Raw economic / environmental features ──────────────────────────
    # (already in df from pipeline; keep as-is)

    # ── 9. Interaction feature: price differential per kWh-equivalent ─────
    # Converts gasoline cost to a kWh-equivalent basis for direct comparison
    KWH_PER_GAL      = 33.7   # energy equivalence
    df["price_diff_per_kwh"] = (
        df["gas_price_usd_gal"] / KWH_PER_GAL
    ) - df["elec_price_usd_kwh"]

    # ── Drop NaN rows introduced by lags / rolling windows ────────────────
    df = df.dropna().reset_index(drop=True)

    print(f"[features] OK Feature matrix ready: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature-engineered DataFrame into X (features) and y (target).

    Returns
    -------
    X : pd.DataFrame – all columns except 'date' and TARGET_COL
    y : pd.Series    – target (ev_demand_sessions)
    """
    drop_cols = ["date", TARGET_COL]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL]
    return X, y


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names."""
    X, _ = get_X_y(df)
    return list(X.columns)


if __name__ == "__main__":
    from data_pipeline import build_daily_dataset
    raw  = build_daily_dataset()
    feat = engineer_features(raw)
    X, y = get_X_y(feat)
    print(f"  X shape: {X.shape}")
    print(f"  y range: {y.min():.0f} – {y.max():.0f}")
    print(f"  Features:\n  {list(X.columns)}")
