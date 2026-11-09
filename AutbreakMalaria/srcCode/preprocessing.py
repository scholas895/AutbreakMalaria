"""
Low-level cleaning utilities: numeric parsing & district name standardisation.
"""
import numpy as np
import pandas as pd


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert strings containing spaces / thin-spaces / commas to float."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "ND": np.nan, "": np.nan, "None": np.nan}),
        errors="coerce",
    )


def standardise_district(name) -> str | None:
    """Return canonical district name (strip ``DS `` prefix, fix Bujumbura sub-cases)."""
    if pd.isna(name):
        return None
    s = str(name).strip()
    for pref in ("DS ", "DS  "):
        if s.startswith(pref):
            s = s[len(pref):]
    s = (s.replace("Bujumbura centre", "Bujumbura Centre")
           .replace("Bujumbura nord",   "Bujumbura Nord")
           .replace("Bujumbura sud",    "Bujumbura Sud"))
    return s.strip()
