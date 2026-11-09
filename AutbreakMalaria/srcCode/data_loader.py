"""
Load malaria cases and environmental covariates from ``BdiDATASETS.xlsx``.
"""
from pathlib import Path

import pandas as pd

from .config import CANDIDATE_PATHS, MONTH_MAP, MONTH_NAMES
from .preprocessing import clean_numeric, standardise_district


# --------------------------------------------------------- file discovery ----
def locate_data_file(path: str | Path | None = None) -> Path:
    """Return a resolved Path to the Excel workbook (prompts if not found)."""
    if path is not None:
        p = Path(path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"{path} not found")

    for candidate in CANDIDATE_PATHS:
        if candidate.exists():
            return candidate.resolve()

    user_input = input("Path to BdiDATASETS.xlsx: ").strip().strip('"').strip("'")
    p = Path(user_input)
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"Could not find BdiDATASETS.xlsx at {user_input}")


# ---------------------------------------------------- district case loader ----
def extract_year_sheet(data_file, sheet_name, year, header_row, district_col, month_cols):
    """Pull one yearly sheet into a long panel: District_Std x Year x Month."""
    df = pd.read_excel(data_file, sheet_name=sheet_name, header=None)
    block = df.iloc[header_row:, [district_col] + list(month_cols)].copy()
    block.columns = ["District"] + MONTH_NAMES
    block = block[block["District"].notna()]
    block = block[block["District"].astype(str).str.startswith("DS ", na=False)]

    long_df = block.melt(id_vars="District", var_name="Month", value_name="Cases")
    long_df["Month_Num"] = long_df["Month"].map(MONTH_MAP)
    long_df["Year"] = year
    long_df["Date"] = pd.to_datetime(
        f"{year}-" + long_df["Month_Num"].astype(int).astype(str) + "-01")
    long_df["Cases"] = clean_numeric(long_df["Cases"])
    long_df["District_Std"] = long_df["District"].apply(standardise_district)
    return long_df[["District_Std", "Year", "Month_Num", "Date", "Cases"]]


def load_malaria_cases(data_file) -> pd.DataFrame:
    """Assemble the district-month case panel across 2022-2024."""
    print("\n--- 1. Extracting district-level malaria cases ---")

    panel_2022 = extract_year_sheet(data_file, "2022", 2022, header_row=2,
                                    district_col=0, month_cols=range(1, 13))
    panel_2023 = extract_year_sheet(data_file, "2023", 2023, header_row=2,
                                    district_col=0, month_cols=range(1, 13))

    # 2024 sheet uses a two-table layout: district-months begin at column T (19),
    # with the header at row 3 (0-based) and districts starting at row 4.
    df_2024 = pd.read_excel(data_file, sheet_name="2024", header=None)
    block = df_2024.iloc[4:, [19] + list(range(20, 32))].copy()
    block.columns = ["District"] + MONTH_NAMES
    block = block[block["District"].astype(str).str.startswith("DS ", na=False)]

    long_2024 = block.melt(id_vars="District", var_name="Month", value_name="Cases")
    long_2024["Month_Num"] = long_2024["Month"].map(MONTH_MAP)
    long_2024["Year"] = 2024
    long_2024["Date"] = pd.to_datetime(
        "2024-" + long_2024["Month_Num"].astype(int).astype(str) + "-01")
    long_2024["Cases"] = clean_numeric(long_2024["Cases"])
    long_2024["District_Std"] = long_2024["District"].apply(standardise_district)
    panel_2024 = long_2024[["District_Std", "Year", "Month_Num", "Date", "Cases"]]

    panel = pd.concat([panel_2022, panel_2023, panel_2024], ignore_index=True)
    panel = panel.dropna(subset=["District_Std", "Cases"])
    panel = panel.sort_values(["District_Std", "Date"]).reset_index(drop=True)

    print(f"  Panel shape        : {panel.shape}")
    print(f"  Districts          : {panel['District_Std'].nunique()}")
    print(f"  Date range         : {panel['Date'].min().date()} -> {panel['Date'].max().date()}")
    return panel


# ------------------------------------------------- environmental loaders -----
def _national_monthly(data_file, sheet_name, year_col, month_col, value_col,
                      header_row, y0=2022, y1=2024) -> pd.DataFrame:
    """Extract a monthly national series (Year/Month/Value) from a sheet."""
    try:
        df = pd.read_excel(data_file, sheet_name=sheet_name, header=None)
        sub = df.iloc[header_row:, [year_col, month_col, value_col]].copy()
        sub.columns = ["Year", "Month", "Value"]
        sub["Year"]  = pd.to_numeric(sub["Year"],  errors="coerce")
        sub["Month"] = pd.to_numeric(sub["Month"], errors="coerce")
        sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
        sub = sub.dropna().astype({"Year": int, "Month": int})
        sub = sub[(sub["Year"] >= y0) & (sub["Year"] <= y1)]
        agg = sub.groupby(["Year", "Month"])["Value"].mean().reset_index()
        agg["Date"] = pd.to_datetime(
            agg["Year"].astype(str) + "-" + agg["Month"].astype(str) + "-01")
        return agg[["Date", "Value"]]
    except Exception as e:
        print(f"  [warn] {sheet_name} extraction failed: {e}")
        return pd.DataFrame(columns=["Date", "Value"])


def load_environmental(data_file) -> pd.DataFrame:
    """Return a Date-indexed DataFrame with Temp / Humidity / Rainfall / Lake / River."""
    print("\n--- 2. Extracting environmental covariates ---")

    temp  = _national_monthly(data_file, "Temperaturemoyen",               4, 5, 6, 1).rename(columns={"Value": "Temp"})
    humid = _national_monthly(data_file, "Humiditerelatives  stations",    4, 5, 6, 1).rename(columns={"Value": "Humidity"})
    lake  = _national_monthly(data_file, "Niveau du Lac Tanganyika",       4, 5, 6, 1).rename(columns={"Value": "Lake_Level"})
    river = _national_monthly(data_file, "Niveau d'eau de la RiviereRusiz",4, 5, 6, 1).rename(columns={"Value": "River_Level"})

    # Rainfall sheet is wide; reshape it.
    try:
        raw = pd.read_excel(data_file, sheet_name="precipitationmensuel30years", header=None)
        rb = raw.iloc[1:, [4] + list(range(5, 17))].copy()
        rb.columns = ["Year"] + MONTH_NAMES
        rb["Year"] = pd.to_numeric(rb["Year"], errors="coerce")
        for m in MONTH_NAMES:
            rb[m] = pd.to_numeric(rb[m], errors="coerce")
        rb = rb.dropna(subset=["Year"]).astype({"Year": int})
        rb = rb[(rb["Year"] >= 2022) & (rb["Year"] <= 2024)]
        rl = rb.melt(id_vars="Year", var_name="Month", value_name="Rainfall")
        rl["Month_Num"] = rl["Month"].map(MONTH_MAP)
        rl = rl.groupby(["Year", "Month_Num"])["Rainfall"].mean().reset_index()
        rl["Date"] = pd.to_datetime(
            rl["Year"].astype(str) + "-" + rl["Month_Num"].astype(str) + "-01")
        rain = rl[["Date", "Rainfall"]]
    except Exception as e:
        print(f"  [warn] Rainfall extraction failed: {e}")
        rain = pd.DataFrame(columns=["Date", "Rainfall"])

    frames = [df for df in (temp, humid, rain, lake, river) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["Date"])

    env = frames[0]
    for f in frames[1:]:
        env = env.merge(f, on="Date", how="outer")
    return env
