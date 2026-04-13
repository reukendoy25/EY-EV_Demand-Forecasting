"""
hyperparameter_tuning.py
========================
EY – EV Market Demand Forecasting
----------------------------------
RandomizedSearchCV tuning for SVR and XGBoost.
Returns best estimators ready to be passed to the ensemble layer.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import loguniform, uniform, randint
from models import SVRForecaster, XGBForecaster


# ── Search spaces ─────────────────────────────────────────────────────────────

SVR_PARAM_DIST = {
    "C":       loguniform(1e-1, 1e3),          # [0.1, 100]
    "gamma":   loguniform(1e-4, 1e0),          # [0.0001, 1]
    "epsilon": uniform(0.01, 0.5),             # [0.01, 0.51]
}

XGB_PARAM_DIST = {
    "n_estimators":  randint(200, 1000),
    "learning_rate": loguniform(0.005, 0.2),
    "max_depth":     randint(3, 9),
    "subsample":     uniform(0.6, 0.4),        # [0.6, 1.0]
    "colsample":     uniform(0.6, 0.4),
    "reg_alpha":     loguniform(1e-3, 10),
    "reg_lambda":    loguniform(1e-1, 10),
}


def tune_svr(
    X_train, y_train,
    n_iter: int = 40,
    cv_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
    random_state: int = 42,
    verbose: bool = True,
) -> SVRForecaster:
    """
    Run RandomizedSearchCV over SVRForecaster hyperparameters.

    Uses TimeSeriesSplit to preserve temporal ordering during cross-validation.
    """
    print("[tuning] Tuning SVR …")
    tscv   = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator           = SVRForecaster(),
        param_distributions = SVR_PARAM_DIST,
        n_iter              = n_iter,
        scoring             = scoring,
        cv                  = tscv,
        refit               = True,
        n_jobs              = -1,
        random_state        = random_state,
        verbose             = 0,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    if verbose:
        print(f"  [SVR] Best params : {search.best_params_}")
        print(f"  [SVR] Best CV MAE : {-search.best_score_:.2f}")
    return best


def tune_xgb(
    X_train, y_train,
    n_iter: int = 40,
    cv_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
    random_state: int = 42,
    verbose: bool = True,
) -> XGBForecaster:
    """
    Run RandomizedSearchCV over XGBForecaster hyperparameters.
    """
    print("[tuning] Tuning XGBoost …")
    tscv   = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator           = XGBForecaster(),
        param_distributions = XGB_PARAM_DIST,
        n_iter              = n_iter,
        scoring             = scoring,
        cv                  = tscv,
        refit               = True,
        n_jobs              = -1,
        random_state        = random_state,
        verbose             = 0,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    if verbose:
        print(f"  [XGB] Best params : {search.best_params_}")
        print(f"  [XGB] Best CV MAE : {-search.best_score_:.2f}")
    return best
