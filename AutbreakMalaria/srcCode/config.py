"""
Configuration, constants, and shared save helpers.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# ------------------------------------------------------------------ Paths ----
BASE_DIR  = Path("Chapter5_EWS_outputs")
FIG_PNG   = BASE_DIR / "figures" / "png"
FIG_PDF   = BASE_DIR / "figures" / "pdf"
TABLE_DIR = BASE_DIR / "tables"

# Files searched in order when no explicit path is given.
CANDIDATE_PATHS = [
    Path("BdiDATASETS.xlsx"),
    Path("data/BdiDATASETS.xlsx"),
    Path("Chapter5_EarlyWarning/BdiDATASETS.xlsx"),
    Path("../BdiDATASETS.xlsx"),
    Path("../../BdiDATASETS.xlsx"),
    Path("experiment2/BdiDATASETS.xlsx"),
    Path.home() / "BdiDATASETS.xlsx",
    Path.home() / "Downloads" / "BdiDATASETS.xlsx",
]

# --------------------------------------------------------------- Constants ---
HORIZONS = [1, 2, 3]

MONTH_MAP   = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
MONTH_NAMES = list(MONTH_MAP.keys())

TRAIN_END = "2023-12-01"   # inclusive
VAL_END   = "2024-06-01"   # inclusive -> test set starts after this

RANDOM_STATE = 42

# ----------------------------------------------------------------- Styling ---
MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#2ca02c",
    "XGBoost":             "#ff7f0e",
    "LSTM-proxy":          "#d62728",
    "TFT-proxy":           "#9467bd",
    "Ensemble":            "#8c564b",
}
MODEL_ORDER = list(MODEL_COLORS.keys())

# ------------------------------------------------------------------ Utils ----
def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for d in (FIG_PNG, FIG_PDF, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """Apply global matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.3,
    })


def save_fig(fig, name: str) -> None:
    """Persist a matplotlib figure to PNG and PDF."""
    fig.savefig(FIG_PNG / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_PDF / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure]  {name}.png | {name}.pdf")


def save_table(df, name: str) -> None:
    """Persist a DataFrame to CSV."""
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    print(f"  [table]   {name}.csv")
