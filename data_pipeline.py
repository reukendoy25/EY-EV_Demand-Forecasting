"""
data_pipeline.py
================
EY - EV Market Demand Forecasting
----------------------------------
Scrapes real-world data from:
  1. U.S. Energy Information Administration (EIA) - weekly retail gasoline prices
  2. EIA - monthly average electricity retail prices (residential)

Derives / synthesises:
  3. EV charging demand (sessions/day) using fuel-price differential as the
     primary driver, plus seasonality noise.
  4. GHG / CO2 savings (kg/day) derived from fuel differential.
  5. Gasoline savings (USD/day) derived from price spread.

No external proprietary APIs or yfinance are used.
"""

import io
import sys
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# EIA endpoints
EIA_GAS_PRICE_URL = (
    "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx"
    "?n=PET&s=EMM_EPM0_PTE_NUS_DPG&f=W"
)
EIA_ELEC_CSV_URL = (
    "https://www.eia.gov/electricity/monthly/xls/table_5_6_a.xlsx"
)

# EV efficiency assumptions
KWH_PER_MILE      = 0.30       # average EV energy consumption
GAS_MILES_PER_GAL = 28.0       # average ICE fuel economy (mpg)
CO2_PER_GAL_KG    = 8.887      # kg CO2 per gallon of gasoline burned
DAILY_MILES       = 37.0       # average US daily driving miles (BLS)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# 1. Scrape EIA weekly gasoline price
# ---------------------------------------------------------------------------

def scrape_eia_gasoline_weekly(start="2018-01-01", end="2025-12-31") -> pd.DataFrame:
    """
    Scrapes the EIA weekly US retail gasoline price page (HTML table).
    Returns DataFrame with columns ['date', 'gas_price_usd_gal'],
    filtered to [start, end] range.
    Falls back to synthetic if scrape fails.
    """
    print("  [scraper] Fetching EIA weekly gasoline prices ...")
    try:
        resp = requests.get(EIA_GAS_PRICE_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))

        df = None
        for tbl in tables:
            cols_lower = [str(c).lower() for c in tbl.columns]
            if any("date" in c for c in cols_lower):
                df = tbl.copy()
                break

        if df is None:
            raise ValueError("No date-bearing table found on EIA page.")

        df.columns = [str(c).strip() for c in df.columns]
        date_col  = [c for c in df.columns if "date" in c.lower()][0]
        price_col = [c for c in df.columns if c != date_col][0]

        result = pd.DataFrame({
            "date":              pd.to_datetime(df[date_col],  errors="coerce"),
            "gas_price_usd_gal": pd.to_numeric(df[price_col], errors="coerce"),
        }).dropna().sort_values("date").reset_index(drop=True)

        # Filter to requested range
        result = result[
            (result["date"] >= start) & (result["date"] <= end)
        ].reset_index(drop=True)

        if len(result) < 50:
            raise ValueError(f"Too few rows after date filter: {len(result)}")

        print(f"  [scraper] Gasoline OK: {len(result)} rows "
              f"({result['date'].min().date()} -> {result['date'].max().date()})")
        return result

    except Exception as exc:
        print(f"  [scraper] Gasoline scrape failed ({exc}); using synthetic data.")
        return _synthetic_gasoline(start, end)


def _synthetic_gasoline(start="2018-01-01", end="2025-12-31") -> pd.DataFrame:
    """Realistic synthetic weekly US gasoline price series."""
    dates  = pd.date_range(start=start, end=end, freq="W-MON")
    n      = len(dates)
    np.random.seed(42)
    trend  = np.linspace(2.60, 3.90, n)
    season = 0.18 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise  = np.random.normal(0, 0.10, n)
    shocks = np.where(np.random.rand(n) > 0.96, np.random.choice([-0.45, 0.55], n), 0)
    prices = np.clip(trend + season + noise + shocks, 2.00, 5.80)
    return pd.DataFrame({"date": dates, "gas_price_usd_gal": prices})


# ---------------------------------------------------------------------------
# 2. Scrape EIA monthly electricity price
# ---------------------------------------------------------------------------

def scrape_eia_electricity_monthly(start="2018-01-01", end="2025-12-31") -> pd.DataFrame:
    """
    Scrapes EIA Table 5.6.A monthly residential electricity price.
    Returns DataFrame with columns ['date', 'elec_price_usd_kwh'].
    Falls back to synthetic if scrape fails.
    """
    print("  [scraper] Fetching EIA monthly residential electricity prices ...")
    try:
        resp = requests.get(EIA_ELEC_CSV_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(resp.content))

        # Try different skiprows to find the data table
        raw = None
        for skip in [4, 3, 5, 2]:
            try:
                tmp = xls.parse(xls.sheet_names[0], skiprows=skip, header=0)
                tmp.columns = [str(c).strip() for c in tmp.columns]
                res_candidates = [c for c in tmp.columns
                                  if "residential" in c.lower()]
                if res_candidates:
                    raw = tmp
                    res_col = res_candidates[0]
                    break
            except Exception:
                continue

        if raw is None:
            raise ValueError("Could not parse EIA electricity xlsx.")

        year_col = raw.columns[0]
        result = pd.DataFrame({
            "date": pd.to_datetime(raw[year_col].astype(str), errors="coerce"),
            "elec_price_usd_kwh": pd.to_numeric(raw[res_col], errors="coerce") / 100.0,
        }).dropna().sort_values("date").reset_index(drop=True)

        result = result[
            (result["date"] >= start) & (result["date"] <= end)
        ].reset_index(drop=True)

        if len(result) < 12:
            raise ValueError(f"Too few rows after date filter: {len(result)}")

        print(f"  [scraper] Electricity OK: {len(result)} rows "
              f"({result['date'].min().date()} -> {result['date'].max().date()})")
        return result

    except Exception as exc:
        print(f"  [scraper] Electricity scrape failed ({exc}); using synthetic data.")
        return _synthetic_electricity(start, end)


def _synthetic_electricity(start="2018-01-01", end="2025-12-31") -> pd.DataFrame:
    """Realistic synthetic monthly US residential electricity price series."""
    dates  = pd.date_range(start=start, end=end, freq="MS")
    n      = len(dates)
    np.random.seed(7)
    prices = np.clip(
        np.linspace(0.1240, 0.1490, n) + np.random.normal(0, 0.003, n),
        0.10, 0.20,
    )
    return pd.DataFrame({"date": dates, "elec_price_usd_kwh": prices})


# ---------------------------------------------------------------------------
# 3. Build daily dataset
# ---------------------------------------------------------------------------

def build_daily_dataset(
    start: str = "2018-01-01",
    end: str   = "2025-12-31",
) -> pd.DataFrame:
    """
    Assembles a clean daily DataFrame by:
    1. Scraping EIA weekly gasoline & monthly electricity prices.
    2. Reindexing onto a continuous daily date spine with forward-fill.
    3. Deriving GHG savings, fuel savings, and synthetic EV demand.

    Returns
    -------
    pd.DataFrame with columns:
        date, gas_price_usd_gal, elec_price_usd_kwh,
        fuel_cost_ice_usd_day, fuel_cost_ev_usd_day, fuel_savings_usd_day,
        ghg_savings_kg_day, ev_demand_sessions
    """
    print("\n[pipeline] Building daily dataset ...")

    gas_df  = scrape_eia_gasoline_weekly(start, end)
    elec_df = scrape_eia_electricity_monthly(start, end)

    # Build continuous daily date spine
    date_idx = pd.date_range(start=start, end=end, freq="D")

    # Reindex gas price to daily (ffill then bfill for leading NaNs)
    gas_series = (
        gas_df.set_index("date")["gas_price_usd_gal"]
        .reindex(date_idx, method="ffill")
    )
    # Handle leading NaNs with bfill
    gas_series = gas_series.bfill()

    # Reindex electricity price to daily
    elec_series = (
        elec_df.set_index("date")["elec_price_usd_kwh"]
        .reindex(date_idx, method="ffill")
    )
    elec_series = elec_series.bfill()

    # Assemble base dataframe
    daily = pd.DataFrame({
        "date":               date_idx,
        "gas_price_usd_gal":  gas_series.values,
        "elec_price_usd_kwh": elec_series.values,
    })

    # Verify no remaining NaNs in price series
    assert daily["gas_price_usd_gal"].isna().sum() == 0, "NaN in gas prices!"
    assert daily["elec_price_usd_kwh"].isna().sum() == 0, "NaN in elec prices!"

    # Derived economic signals
    daily["fuel_cost_ice_usd_day"] = (
        DAILY_MILES / GAS_MILES_PER_GAL
    ) * daily["gas_price_usd_gal"]

    daily["fuel_cost_ev_usd_day"] = (
        DAILY_MILES * KWH_PER_MILE
    ) * daily["elec_price_usd_kwh"]

    daily["fuel_savings_usd_day"] = (
        daily["fuel_cost_ice_usd_day"] - daily["fuel_cost_ev_usd_day"]
    )

    # GHG savings (kg CO2/day avoided compared to ICE vehicle)
    daily["ghg_savings_kg_day"] = float(
        (DAILY_MILES / GAS_MILES_PER_GAL) * CO2_PER_GAL_KG
    )

    # -----------------------------------------------------------------------
    # Synthetic EV charging demand (sessions/day)
    # Driven by: logistic adoption trend + fuel-savings elasticity
    #            + weekly seasonality + annual seasonality + noise + shocks
    # -----------------------------------------------------------------------
    np.random.seed(2024)
    n = len(daily)
    t = np.arange(n)

    # S-curve adoption (logistic function, midpoint at ~50% of the series)
    midpoint  = n * 0.50
    steepness = 0.0075
    adoption  = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))

    # Base demand: ~500 sessions/day (2018) scaling to ~12,000 (2025)
    base_demand = 500.0 + adoption * 11_500.0

    # Fuel-savings elasticity: normalised 0-1, adds up to 25% uplift
    fs      = daily["fuel_savings_usd_day"].values
    fs_norm = (fs - fs.min()) / (fs.max() - fs.min() + 1e-9)
    elasticity = 1.0 + 0.25 * fs_norm

    # Weekly seasonality (higher mid-week)
    dow           = daily["date"].dt.dayofweek.values   # 0=Mon, 6=Sun
    weekly_season = 1.0 + 0.08 * np.sin(2 * np.pi * dow / 7)

    # Annual seasonality (peaks in summer, ~Jul)
    doy           = daily["date"].dt.dayofyear.values
    annual_season = 1.0 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365)

    # Market noise and random shocks
    noise  = np.random.normal(0, 0.04, n)
    shock_pool = [0.0] * 90 + [0.18, -0.13, 0.22, -0.09, 0.28,
                                0.12, -0.15, 0.30, -0.10, 0.16]
    shocks = np.random.choice(shock_pool, n)

    demand = np.maximum(
        0.0,
        base_demand * elasticity * weekly_season * annual_season * (1.0 + noise + shocks)
    )
    daily["ev_demand_sessions"] = demand.round(0)

    # Final check
    assert daily.isna().sum().sum() == 0, "Unexpected NaNs in final dataset!"

    print(f"[pipeline] Dataset OK: {len(daily)} rows, {len(daily.columns)} cols")
    print(f"           Date range : {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print(f"           EV demand  : {daily['ev_demand_sessions'].min():.0f} - "
          f"{daily['ev_demand_sessions'].max():.0f} sessions/day")
    print(f"           Fuel saving: ${daily['fuel_savings_usd_day'].mean():.2f}/day avg")
    return daily


if __name__ == "__main__":
    df = build_daily_dataset()
    print(df.tail(5).to_string())
