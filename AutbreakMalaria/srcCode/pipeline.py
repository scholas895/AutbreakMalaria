"""
End-to-end orchestrator for Experiment 2 (Malaria Outbreak Early Warning).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import visualization as viz
from .config import (HORIZONS, MODEL_ORDER, TRAIN_END, VAL_END,
                     ensure_dirs, save_table, setup_plot_style)
from .data_loader import (load_environmental, load_malaria_cases,
                          locate_data_file)
from .evaluation import (evaluate_ensemble, evaluate_model, mean_lead_time,
                         split_and_impute)
from .feature_engineering import (add_env_lag_features, add_horizon_labels,
                                  add_lag_features, add_seasonal_features,
                                  get_features)
from .models import build_models
from .outbreak_detection import add_outbreak_labels, compute_thresholds

warnings.filterwarnings("ignore")
np.random.seed(42)


# --------------------------------------------------------------- training ----
def train_all(panel: pd.DataFrame, features: list[str]):
    rows, prob_store, label_store = [], {}, {}

    for h in HORIZONS:
        print(f"\n   Horizon h = {h} month(s)")
        Xtr, ytr, Xva, yva, Xte, yte, *_ = split_and_impute(
            panel, features, h, TRAIN_END, VAL_END)
        label_store[h] = yte

        pos_w = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
        models = build_models(pos_w)

        prob_val, prob_test = {}, {}
        for name, mdl in models.items():
            m, pt, pv = evaluate_model(mdl, Xtr, ytr, Xva, yva, Xte, yte, name, h)
            rows.append(m)
            prob_store[(name, h)] = pt
            prob_val[name] = pv
            prob_test[name] = pt
            print(f"     {name:<22}  F1={m['F1']:.3f}  AUROC={m['AUROC']:.3f}  Brier={m['Brier']:.3f}")

        m, ens_test = evaluate_ensemble(prob_val, prob_test, yva, yte, h)
        rows.append(m)
        prob_store[("Ensemble", h)] = ens_test
        print(f"     {'Ensemble':<22}  F1={m['F1']:.3f}  AUROC={m['AUROC']:.3f}  Brier={m['Brier']:.3f}")

    return pd.DataFrame(rows), prob_store, label_store


# --------------------------------------------------------------- tables ------
def export_tables(panel, results_df, thresholds, district_f1, lead_df) -> None:
    print("\n--- 6. Exporting tables ---")

    tbl = (results_df[results_df["Horizon"] == 1]
           .sort_values("F1", ascending=False).reset_index(drop=True))
    save_table(tbl, "table_5_1_performance_h1")

    t2 = results_df.pivot_table(index="Model", columns="Horizon", values="F1").reset_index()
    t2.columns = ["Model", "F1_h1", "F1_h2", "F1_h3"]
    save_table(t2, "table_5_2_f1_by_horizon")

    t2b = results_df.pivot_table(index="Model", columns="Horizon", values="AUROC").reset_index()
    t2b.columns = ["Model", "AUROC_h1", "AUROC_h2", "AUROC_h3"]
    save_table(t2b, "table_5_2b_auroc_by_horizon")

    t3 = results_df[["Model", "Horizon", "Sensitivity", "Specificity",
                     "False_Alarm_Rate", "Missed_Alert_Rate",
                     "Brier", "Threshold"]].copy()
    save_table(t3, "table_5_3_operational_metrics")

    t4 = results_df[["Model", "Horizon", "Brier", "Log_Loss", "AUPRC", "AUROC"]].copy()
    save_table(t4, "table_5_4_calibration_discrimination")

    t5 = results_df[["Model", "Horizon", "TP", "FP", "TN", "FN"]].copy()
    save_table(t5, "table_5_5_confusion_matrices")

    t6 = (panel.groupby("Year")["Outbreak_80"]
          .agg(Total="count", Outbreaks="sum", Proportion="mean").reset_index())
    t6["Proportion"] = (t6["Proportion"] * 100).round(2)
    save_table(t6, "table_5_6_class_distribution_by_year")

    t7 = thresholds["Threshold_80"].merge(thresholds["Threshold_75"], on="District_Std")
    t7 = t7.merge(thresholds["Threshold_90"], on="District_Std")
    save_table(t7, "table_5_7_district_thresholds")

    save_table(results_df, "table_5_8_full_results")
    save_table(district_f1, "table_5_9_district_f1_ensemble_h1")
    save_table(lead_df, "table_5_10_operational_lead_time")


# --------------------------------------------------------------- pipeline ----
def run_pipeline(data_file: str | None = None) -> None:
    print("=" * 78)
    print("CPST-MalariaNet - Experiment 2: Malaria Outbreak Early Warning")
    print("=" * 78)

    ensure_dirs()
    setup_plot_style()

    df_path = locate_data_file(data_file)
    print(f"[ok] Using Excel file: {df_path}")

    # 1. Load
    panel = load_malaria_cases(df_path)
    env = load_environmental(df_path)
    if not env.empty:
        panel = panel.merge(env, on="Date", how="left")
    print(f"  Panel shape with env: {panel.shape}")
    print(f"  Env columns present: "
          f"{[c for c in ['Temp','Humidity','Rainfall','Lake_Level','River_Level'] if c in panel.columns]}")

    # 2. Labels & features
    print("\n--- 3. Defining outbreak thresholds and labels ---")
    thresholds = compute_thresholds(panel, TRAIN_END)
    panel = add_outbreak_labels(panel, thresholds)
    panel = add_horizon_labels(panel)
    panel = add_lag_features(panel)
    panel = add_seasonal_features(panel)
    panel = add_env_lag_features(panel)
    features = get_features(panel)

    print(f"  Feature count: {len(features)}")
    print(f"  Class balance (80th percentile): "
          f"{panel['Outbreak_80'].sum()} positive / {len(panel)} total "
          f"({panel['Outbreak_80'].mean()*100:.1f} %)")

    # 3. Descriptive figures
    print("\n--- 4. Descriptive figures ---")
    viz.plot_class_distribution(panel)
    viz.plot_district_thresholds(thresholds["Threshold_80"])
    viz.plot_timeseries_outbreaks(panel)

    # 4. Train
    print("\n--- 5. Training classification models ---")
    results_df, prob_store, label_store = train_all(panel, features)

    # 5. District-level F1 (Ensemble, h=1)
    _, _, _, _, _, yte, _, _, te = split_and_impute(
        panel, features, 1, TRAIN_END, VAL_END)
    te = te.copy()
    te["Pred"] = (prob_store[("Ensemble", 1)] >= 0.5).astype(int)
    district_f1 = (te.groupby("District_Std")
                   .apply(lambda g: f1_score(g["Outbreak_80_h1"], g["Pred"],
                                             zero_division=0))
                   .reset_index(name="F1").sort_values("F1", ascending=False))

    # 6. Lead time
    lead_rows = []
    for m in MODEL_ORDER:
        if (m, 1) not in prob_store:
            continue
        thr = results_df[(results_df["Model"] == m) &
                         (results_df["Horizon"] == 1)]["Threshold"].values[0]
        lead_rows.append({
            "Model": m,
            "Mean_Lead_Time_Months": round(
                mean_lead_time(yte, prob_store[(m, 1)], thr), 2)})
    lead_df = pd.DataFrame(lead_rows)

    # 7. Export tables
    export_tables(panel, results_df, thresholds, district_f1, lead_df)

    # 8. Performance figures
    print("\n--- 7. Performance figures ---")
    viz.plot_f1_by_model_horizon(results_df)
    viz.plot_performance_heatmap(results_df)
    viz.plot_roc_curves_h1(prob_store, label_store)
    viz.plot_roc_curves_all_horizons(prob_store, label_store)
    viz.plot_pr_curves_h1(prob_store, label_store)
    viz.plot_pr_curves_all_horizons(prob_store, label_store)
    viz.plot_calibration_h1(prob_store, label_store)
    viz.plot_sens_spec_tradeoff(results_df)
    viz.plot_operational_metrics(results_df)
    viz.plot_master_grid(results_df)
    viz.plot_threshold_sensitivity(prob_store, label_store)
    viz.plot_confusion_matrices_h1(results_df)
    viz.plot_district_f1(district_f1)
    viz.plot_lead_time(lead_df)

    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - COMPLETE")
    print("=" * 78)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
