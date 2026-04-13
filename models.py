"""
models.py
=========
EY – EV Market Demand Forecasting
----------------------------------
Defines two sklearn-compatible forecasters:

  • SVRForecaster  – Pipeline(StandardScaler → SVR(kernel='rbf'))
  • XGBForecaster  – XGBRegressor wrapped with early-stopping + feature validation
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor


# ── SVR Forecaster ────────────────────────────────────────────────────────────

class SVRForecaster(BaseEstimator, RegressorMixin):
    """
    SVR with RBF kernel wrapped in a StandardScaler pipeline.

    SVR is well-suited for capturing the smooth, continuous underlying
    trend of EV adoption growth while ignoring outlier noise via the
    epsilon-insensitive loss tube.

    Parameters
    ----------
    C       : float  – regularisation parameter (controls margin hardness)
    gamma   : float|str – kernel coefficient
    epsilon : float  – width of the epsilon-insensitive tube
    """

    def __init__(self, C: float = 1.0, gamma: str = "scale", epsilon: float = 0.1):
        self.C       = C
        self.gamma   = gamma
        self.epsilon = epsilon

    def _build_pipeline(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(
                kernel  = "rbf",
                C       = self.C,
                gamma   = self.gamma,
                epsilon = self.epsilon,
            )),
        ])

    def fit(self, X, y):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def get_params(self, deep=True):
        return {"C": self.C, "gamma": self.gamma, "epsilon": self.epsilon}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── XGBoost Forecaster ────────────────────────────────────────────────────────

class XGBForecaster(BaseEstimator, RegressorMixin):
    """
    XGBRegressor wrapper for sklearn pipelines.

    XGBoost sequentially corrects residual errors of prior trees, making
    it excellent at capturing abrupt, non-linear market shocks driven by
    the engineered economic and environmental features.

    Parameters
    ----------
    n_estimators  : int   – number of boosting rounds
    learning_rate : float – step-size shrinkage
    max_depth     : int   – maximum tree depth
    subsample     : float – row-sampling fraction per tree
    colsample     : float – feature-sampling fraction per tree
    reg_alpha     : float – L1 regularisation
    reg_lambda    : float – L2 regularisation
    """

    def __init__(
        self,
        n_estimators: int   = 500,
        learning_rate: float = 0.05,
        max_depth: int      = 5,
        subsample: float    = 0.8,
        colsample: float    = 0.8,
        reg_alpha: float    = 0.1,
        reg_lambda: float   = 1.0,
        random_state: int   = 42,
    ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.subsample     = subsample
        self.colsample     = colsample
        self.reg_alpha     = reg_alpha
        self.reg_lambda    = reg_lambda
        self.random_state  = random_state

    def fit(self, X, y, eval_set=None):
        self.model_ = XGBRegressor(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            max_depth         = self.max_depth,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            random_state      = self.random_state,
            verbosity         = 0,
            n_jobs            = -1,
        )
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"]  = False
        self.model_.fit(X, y, **fit_kwargs)
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importances(self) -> pd.Series:
        """Return feature importances (gain-based) as a sorted Series."""
        names = (
            self.feature_names_in_
            if self.feature_names_in_ is not None
            else [f"f{i}" for i in range(len(self.model_.feature_importances_))]
        )
        return (
            pd.Series(self.model_.feature_importances_, index=names)
            .sort_values(ascending=False)
        )

    def get_params(self, deep=True):
        return {
            "n_estimators":  self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth":     self.max_depth,
            "subsample":     self.subsample,
            "colsample":     self.colsample,
            "reg_alpha":     self.reg_alpha,
            "reg_lambda":    self.reg_lambda,
            "random_state":  self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
