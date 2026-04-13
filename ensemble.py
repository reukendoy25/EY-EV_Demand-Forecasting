"""
ensemble.py
===========
EY – EV Market Demand Forecasting
----------------------------------
Combines SVR and XGBoost predictions via:
  (a) Numerically optimised weighted average (minimises validation MAE)
  (b) Sklearn VotingRegressor (uniform weights — baseline comparison)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Numerically optimises the blend weight α such that:
        ŷ = α · ŷ_SVR + (1-α) · ŷ_XGB
    minimises MAE on a held-out validation fold.

    Parameters
    ----------
    svr_model : fitted SVRForecaster
    xgb_model : fitted XGBForecaster
    alpha     : float (0–1) – weight on SVR; (1–alpha) on XGBoost.
                Set automatically during fit().
    """

    def __init__(self, svr_model=None, xgb_model=None, alpha: float = 0.5):
        self.svr_model = svr_model
        self.xgb_model = xgb_model
        self.alpha      = alpha

    def fit(self, X_val, y_val):
        """
        Optimise blend weight using Nelder-Mead on validation data.
        Both SVR and XGBoost must already be fitted before calling this.
        """
        y_svr = self.svr_model.predict(X_val)
        y_xgb = self.xgb_model.predict(X_val)

        def _objective(alpha_arr):
            a  = np.clip(alpha_arr[0], 0, 1)
            yp = a * y_svr + (1 - a) * y_xgb
            return mean_absolute_error(y_val, yp)

        result     = minimize(_objective, x0=[0.5], method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
        self.alpha = float(np.clip(result.x[0], 0, 1))
        print(f"[ensemble] ✔ Optimised blend weight: α={self.alpha:.4f} "
              f"(SVR={self.alpha:.2%} | XGB={(1-self.alpha):.2%})")
        return self

    def predict(self, X):
        y_svr = self.svr_model.predict(X)
        y_xgb = self.xgb_model.predict(X)
        return self.alpha * y_svr + (1 - self.alpha) * y_xgb

    def get_params(self, deep=True):
        return {"svr_model": self.svr_model,
                "xgb_model": self.xgb_model,
                "alpha":      self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_voting_ensemble(svr_model, xgb_model):
    """
    Build a sklearn VotingRegressor with equal weights as a baseline.
    Note: VotingRegressor calls .fit() internally, so use only with
    unfitted estimators for training purposes.
    """
    return VotingRegressor(
        estimators=[("svr", svr_model), ("xgb", xgb_model)],
        weights=[1, 1],
        n_jobs=-1,
    )
