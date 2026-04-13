"""
evaluation.py
=============
EY – EV Market Demand Forecasting
----------------------------------
Out-of-sample evaluation metrics + residual analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Dict


def evaluate(y_true, y_pred, label: str = "Model") -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : array-like – actual values
    y_pred : array-like – predicted values
    label  : str        – name printed in the summary

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE, R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    r2   = r2_score(y_true, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    print(f"\n{'─'*50}")
    print(f"  {label} – Out-of-Sample Metrics")
    print(f"{'─'*50}")
    print(f"  MAE   : {mae:>10.2f}  sessions/day")
    print(f"  RMSE  : {rmse:>10.2f}  sessions/day")
    print(f"  MAPE  : {mape:>10.2f}  %")
    print(f"  R²    : {r2:>10.4f}")
    print(f"{'─'*50}\n")

    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing multiple model results.

    Parameters
    ----------
    results : dict  {model_name: {MAE, RMSE, MAPE, R2}}

    Returns
    -------
    pd.DataFrame sorted by MAE ascending
    """
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    df = df.sort_values("MAE").reset_index(drop=True)
    return df


def residual_stats(y_true, y_pred) -> Dict[str, float]:
    """Return basic statistics on the residuals."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean":   float(np.mean(residuals)),
        "std":    float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "q95":    float(np.percentile(np.abs(residuals), 95)),
    }
