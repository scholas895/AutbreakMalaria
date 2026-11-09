"""
Feature engineering: horizon labels, lag features, seasonality, covariates.
"""
import numpy as np
import pandas as pd

from .config import HORIZONS

# Columns that are never model inputs.
NON_FEATURES = {
    "District_Std", "Year", "Month_Num", "Date", "Cases",
    "Threshold_75", "Threshold_80", "Threshold_90",
    "Outbreak_80",  "Outbreak_75",  "Outbreak_90",
    "Outbreak_80_h1", "Outbreak_80_h2", "Outbreak_80_h3",
}


def add_horizon_labels(panel: pd.DataFrame,
                       horizons: list[int] | None = None,
                       label_col: str = "Outbreak_80") -> pd.DataFrame:
    """Add ``label_col_h{h}`` = future label at t+h (per district)."""
    horizons = horizons or HORIZONS
    panel = panel.sort_values(["District_Std", "Date"]).reset_index(drop=True)
    for h in horizons:
        panel[f"{label_col}_h{h}"] = panel.groupby("District_Std")[label_col].shift(-h)
    return panel


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag-1/2/3 cases + 3- and 6-month rolling means (per district)."""
    for lag in (1, 2, 3):
        panel[f"Cases_Lag{lag}"] = panel.groupby("District_Std")["Cases"].shift(lag)
    panel["Cases_Roll3"] = panel.groupby("District_Std")["Cases"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    panel["Cases_Roll6"] = panel.groupby("District_Std")["Cases"].transform(
        lambda x: x.rolling(6, min_periods=1).mean())
    return panel


def add_seasonal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cyclic month encodings."""
    panel["Month_Sin"] = np.sin(2 * np.pi * panel["Month_Num"] / 12)
    panel["Month_Cos"] = np.cos(2 * np.pi * panel["Month_Num"] / 12)
    return panel


def add_env_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1/2 for each environmental covariate (if present)."""
    for col in ("Rainfall", "Temp", "Humidity", "Lake_Level", "River_Level"):
        if col in panel.columns:
            for lag in (1, 2):
                panel[f"{col}_Lag{lag}"] = (
                    panel.groupby("District_Std")[col].shift(lag)
                )
    return panel


def get_features(panel: pd.DataFrame) -> list[str]:
    """Return the model-input column list."""
    return [c for c in panel.columns if c not in NON_FEATURES]
