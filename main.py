"""
main.py
=======
EY – EV Market Demand Forecasting
----------------------------------
End-to-end orchestration:
  1. Scrape & build daily dataset
  2. Engineer features
  3. Train/test split (chronological)
  4. Hyperparameter tuning (SVR + XGBoost)
  5. Ensemble optimisation
  6. Evaluate on out-of-sample test set
  7. Save artefacts (models + dataset)
  8. (Optional) launch Streamlit dashboard

Usage:
    python main.py              # Full run
    python main.py --dashboard  # Run + launch dashboard
    python main.py --fast       # Skip tuning, use defaults
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Local modules ─────────────────────────────────────────────────────────────
from data_pipeline       import build_daily_dataset
from feature_engineering import engineer_features, get_X_y
from models              import SVRForecaster, XGBForecaster
from hyperparameter_tuning import tune_svr, tune_xgb
from ensemble            import WeightedEnsemble
from evaluation          import evaluate, compare_models
from visualization       import (
    forecast_chart, feature_importance_chart,
    residual_chart, metrics_bar_chart, gasoline_vs_demand,
)

ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def main(fast_mode: bool = False, launch_dashboard: bool = False):

    print("\n" + "═"*60)
    print("  EY – EV Market Demand Forecasting Pipeline")
    print("  Ensemble: XGBoost + SVR  |  Web Scraped Data: EIA")
    print("═"*60 + "\n")

    # ── 1. Data ───────────────────────────────────────────────────────────
    raw_df = build_daily_dataset(start="2018-01-01", end="2025-12-31")
    raw_df.to_csv(ARTEFACT_DIR / "raw_dataset.csv", index=False)
    print(f"[main] Raw dataset saved → artefacts/raw_dataset.csv")

    # ── 2. Feature engineering ────────────────────────────────────────────
    feat_df = engineer_features(raw_df)
    feat_df.to_csv(ARTEFACT_DIR / "feature_dataset.csv", index=False)
    print(f"[main] Feature dataset saved → artefacts/feature_dataset.csv")

    X, y = get_X_y(feat_df)

    # ── 3. Chronological train/val/test split ─────────────────────────────
    n        = len(X)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.15)

    X_train, y_train = X.iloc[:n_train],        y.iloc[:n_train]
    X_val,   y_val   = X.iloc[n_train:n_train+n_val], y.iloc[n_train:n_train+n_val]
    X_test,  y_test  = X.iloc[n_train+n_val:],  y.iloc[n_train+n_val:]
    dates_test       = feat_df["date"].iloc[n_train+n_val:].values

    print(f"\n[main] Split → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 4. Model training / hyperparameter tuning ─────────────────────────
    if fast_mode:
        print("\n[main] Fast mode – using default hyperparameters")
        svr_model = SVRForecaster(C=10.0, gamma=0.01, epsilon=0.1)
        xgb_model = XGBForecaster(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8, colsample=0.8)
        svr_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
    else:
        # Train on train+val data for tuning, then fit best params on full train
        X_tv = pd.concat([X_train, X_val])
        y_tv = pd.concat([y_train, y_val])
        svr_model = tune_svr(X_tv, y_tv, n_iter=30)
        xgb_model = tune_xgb(X_tv, y_tv, n_iter=30)

    # ── 5. Ensemble ───────────────────────────────────────────────────────
    ensemble = WeightedEnsemble(svr_model=svr_model, xgb_model=xgb_model)
    ensemble.fit(X_val, y_val)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    y_svr_test = svr_model.predict(X_test)
    y_xgb_test = xgb_model.predict(X_test)
    y_ens_test = ensemble.predict(X_test)

    results = {}
    results["SVR"]      = evaluate(y_test, y_svr_test, "SVR")
    results["XGBoost"]  = evaluate(y_test, y_xgb_test, "XGBoost")
    results["Ensemble"] = evaluate(y_test, y_ens_test,  "Ensemble")

    comparison = compare_models(results)
    print("\n[main] Model Comparison:")
    print(comparison.to_string(index=False))

    # ── 7. Save artefacts ─────────────────────────────────────────────────
    joblib.dump(svr_model,  ARTEFACT_DIR / "svr_model.pkl")
    joblib.dump(xgb_model,  ARTEFACT_DIR / "xgb_model.pkl")
    joblib.dump(ensemble,   ARTEFACT_DIR / "ensemble_model.pkl")
    comparison.to_csv(ARTEFACT_DIR / "model_comparison.csv", index=False)
    print("\n[main] ✔ Models saved → artefacts/")

    # ── 8. Save prediction CSV ────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":        dates_test,
        "actual":      y_test.values,
        "svr_pred":    y_svr_test,
        "xgb_pred":    y_xgb_test,
        "ensemble_pred": y_ens_test,
    })
    pred_df.to_csv(ARTEFACT_DIR / "test_predictions.csv", index=False)
    print("[main] ✔ Predictions saved → artefacts/test_predictions.csv")

    # ── 9. Save HTML charts ───────────────────────────────────────────────
    charts_dir = ARTEFACT_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)

    fc = forecast_chart(
        dates_test, y_test.values,
        {"SVR": y_svr_test, "XGBoost": y_xgb_test, "Ensemble": y_ens_test},
    )
    fc.write_html(str(charts_dir / "forecast.html"))

    fi = feature_importance_chart(xgb_model.feature_importances())
    fi.write_html(str(charts_dir / "feature_importance.html"))

    rc = residual_chart(dates_test, y_test.values, y_ens_test)
    rc.write_html(str(charts_dir / "residuals.html"))

    mc = metrics_bar_chart(comparison)
    mc.write_html(str(charts_dir / "model_comparison.html"))

    gd = gasoline_vs_demand(raw_df)
    gd.write_html(str(charts_dir / "gasoline_vs_demand.html"))

    print(f"[main] ✔ Charts saved → artefacts/charts/")

    print("\n" + "═"*60)
    print("  Pipeline complete ✔")
    best = comparison.iloc[0]
    print(f"  Best model : {best['Model']}")
    print(f"  MAE        : {best['MAE']:.2f}  sessions/day")
    print(f"  RMSE       : {best['RMSE']:.2f}  sessions/day")
    print(f"  R²         : {best['R2']:.4f}")
    print("═"*60 + "\n")

    if launch_dashboard:
        print("[main] Launching Streamlit dashboard …")
        os.system("streamlit run dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EY EV Demand Forecasting Pipeline")
    parser.add_argument("--fast",      action="store_true",
                        help="Skip hyperparameter tuning (use defaults)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Streamlit dashboard after training")
    args = parser.parse_args()

    main(fast_mode=args.fast, launch_dashboard=args.dashboard)
