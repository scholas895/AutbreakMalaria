"""
Compute district-specific outbreak thresholds and binary outbreak labels.
"""
import pandas as pd


def compute_thresholds(panel: pd.DataFrame, train_end: str,
                       quantiles=(0.75, 0.80, 0.90)) -> dict[str, pd.DataFrame]:
    """
    Compute per-district case quantiles over the training period.

    Returns
    -------
    dict of {column_name: DataFrame[District_Std, column_name]}
    e.g. {"Threshold_80": ...}.
    """
    train_mask = panel["Date"] <= pd.Timestamp(train_end)
    thresholds: dict[str, pd.DataFrame] = {}
    for q in quantiles:
        col = f"Threshold_{int(q * 100)}"
        thr = (panel[train_mask]
               .groupby("District_Std")["Cases"]
               .quantile(q)
               .reset_index())
        thr.columns = ["District_Std", col]
        thresholds[col] = thr
    return thresholds


def add_outbreak_labels(panel: pd.DataFrame,
                        thresholds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge thresholds back and add binary ``Outbreak_XX`` columns."""
    for thr_col, thr_df in thresholds.items():
        out_col = thr_col.replace("Threshold_", "Outbreak_")
        panel = panel.merge(thr_df, on="District_Std", how="left")
        panel[out_col] = (panel["Cases"] > panel[thr_col]).astype(int)
    return panel
