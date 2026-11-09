"""
Figure generation for Experiment 2 (17 figures, PNG + PDF).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .config import HORIZONS, MODEL_COLORS, MODEL_ORDER, save_fig


def _present_models(results_df: pd.DataFrame) -> list[str]:
    return [m for m in MODEL_ORDER if any(results_df["Model"] == m)]


# =====================================================================
# Descriptive figures
# =====================================================================
def plot_class_distribution(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = panel["Outbreak_80"].value_counts().sort_index()
    bars = axes[0].bar(["Non-outbreak (0)", "Outbreak (1)"], counts.values,
                       color=["#4c72b0", "#c44e52"], edgecolor="black")
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 3, str(v),
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Number of district-months")
    axes[0].set_title("(a) Class distribution (80th percentile)")

    by_year = panel.groupby("Year")["Outbreak_80"].mean() * 100
    bars = axes[1].bar(by_year.index.astype(str), by_year.values,
                       color="#55a868", edgecolor="black")
    for b, v in zip(bars, by_year.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Outbreak proportion (%)")
    axes[1].set_ylim(0, max(by_year.values) * 1.3)
    axes[1].set_title("(b) Outbreak proportion by year")

    plt.tight_layout()
    save_fig(fig, "fig5_1_class_distribution")


def plot_district_thresholds(thr80: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    thr_sorted = thr80.sort_values("Threshold_80", ascending=False).reset_index(drop=True)
    ax.bar(range(len(thr_sorted)), thr_sorted["Threshold_80"],
           color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(thr_sorted)))
    ax.set_xticklabels(thr_sorted["District_Std"], rotation=90, fontsize=6)
    ax.set_ylabel("Outbreak threshold (cases)")
    ax.set_title("District-specific outbreak thresholds (80th percentile, training period)")
    ax.axhline(thr_sorted["Threshold_80"].mean(), color="red", linestyle="--",
               label=f"Mean = {thr_sorted['Threshold_80'].mean():,.0f}")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "fig5_2_district_thresholds")


def plot_timeseries_outbreaks(panel: pd.DataFrame) -> None:
    sample_districts = sorted(panel["District_Std"].dropna().unique())[:4]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, dist in zip(axes, sample_districts):
        sub = panel[panel["District_Std"] == dist].sort_values("Date")
        ax.plot(sub["Date"], sub["Cases"], color="black", lw=1.2, label="Cases")
        if sub["Threshold_80"].notna().any():
            thr = sub["Threshold_80"].iloc[0]
            ax.axhline(thr, color="red", ls="--", lw=1, label=f"Threshold ({thr:,.0f})")
        outbreak = sub[sub["Outbreak_80"] == 1]
        ax.scatter(outbreak["Date"], outbreak["Cases"], color="red", s=30,
                   zorder=5, label="Outbreak")
        ax.set_title(dist, fontsize=10)
        ax.set_ylabel("Cases")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    save_fig(fig, "fig5_3_timeseries_outbreaks")


# =====================================================================
# Performance figures
# =====================================================================
def plot_f1_by_model_horizon(results_df: pd.DataFrame) -> None:
    present = _present_models(results_df)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(present))
    w = 0.25
    for i, h in enumerate(HORIZONS):
        vals = []
        for m in present:
            v = results_df[(results_df["Model"] == m) &
                           (results_df["Horizon"] == h)]["F1"].values
            vals.append(v[0] if len(v) else 0.0)
        ax.bar(x + i * w, vals, w, label=f"h = {h} month(s)",
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + w)
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score by model and forecast horizon")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "fig5_4_f1_by_model_horizon")


def plot_performance_heatmap(results_df: pd.DataFrame) -> None:
    present = _present_models(results_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = (results_df.pivot_table(index="Model", columns="Horizon", values="F1")
             .reindex(present))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "F1-score"})
    ax.set_title("F1-score heatmap (model x forecast horizon)")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("")
    plt.tight_layout()
    save_fig(fig, "fig5_5_performance_heatmap")


def plot_roc_curves_h1(prob_store: dict, label_store: dict) -> None:
    yte = label_store[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in MODEL_ORDER:
        if (m, 1) not in prob_store:
            continue
        pt = prob_store[(m, 1)]
        fpr, tpr, _ = roc_curve(yte, pt)
        auc = roc_auc_score(yte, pt)
        ax.plot(fpr, tpr, lw=1.8, color=MODEL_COLORS[m],
                label=f"{m} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC curves - forecast horizon h = 1 month")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_6_roc_curves_h1")


def plot_roc_curves_all_horizons(prob_store: dict, label_store: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        yte = label_store[h]
        for m in MODEL_ORDER:
            if (m, h) not in prob_store:
                continue
            pt = prob_store[(m, h)]
            fpr, tpr, _ = roc_curve(yte, pt)
            auc = roc_auc_score(yte, pt)
            ax.plot(fpr, tpr, lw=1.4, color=MODEL_COLORS[m],
                    label=f"{m} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"h = {h} month(s)")
        ax.set_xlabel("FPR")
        if h == 1:
            ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    save_fig(fig, "fig5_7_roc_curves_all_horizons")


def plot_pr_curves_h1(prob_store: dict, label_store: dict) -> None:
    yte = label_store[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in MODEL_ORDER:
        if (m, 1) not in prob_store:
            continue
        pt = prob_store[(m, 1)]
        prec, rec, _ = precision_recall_curve(yte, pt)
        ap = average_precision_score(yte, pt)
        ax.plot(rec, prec, lw=1.8, color=MODEL_COLORS[m],
                label=f"{m} (AUPRC={ap:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves - h = 1 month")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_8_pr_curves_h1")


def plot_pr_curves_all_horizons(prob_store: dict, label_store: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        yte = label_store[h]
        for m in MODEL_ORDER:
            if (m, h) not in prob_store:
                continue
            pt = prob_store[(m, h)]
            prec, rec, _ = precision_recall_curve(yte, pt)
            ap = average_precision_score(yte, pt)
            ax.plot(rec, prec, lw=1.4, color=MODEL_COLORS[m],
                    label=f"{m} ({ap:.2f})")
        ax.set_title(f"h = {h} month(s)")
        ax.set_xlabel("Recall")
        if h == 1:
            ax.set_ylabel("Precision")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    save_fig(fig, "fig5_9_pr_curves_all_horizons")


def plot_calibration_h1(prob_store: dict, label_store: dict) -> None:
    yte = label_store[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in MODEL_ORDER:
        if (m, 1) not in prob_store:
            continue
        pt = prob_store[(m, 1)]
        try:
            prob_true, prob_pred = calibration_curve(yte, pt, n_bins=8, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", lw=1.5,
                    color=MODEL_COLORS[m], label=m)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves - h = 1 month")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_10_calibration_h1")


def plot_sens_spec_tradeoff(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in _present_models(results_df):
        sub = results_df[results_df["Model"] == m]
        ax.plot(sub["Sensitivity"], sub["Specificity"], marker="o",
                color=MODEL_COLORS[m], label=m, lw=1.5)
        for _, row in sub.iterrows():
            ax.annotate(f"h={int(row['Horizon'])}",
                        (row["Sensitivity"], row["Specificity"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Sensitivity (Recall)")
    ax.set_ylabel("Specificity")
    ax.set_title("Sensitivity-Specificity trade-off by model and horizon")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_11_sens_spec_tradeoff")


def plot_operational_metrics(results_df: pd.DataFrame) -> None:
    present = _present_models(results_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(
            axes,
            ["False_Alarm_Rate", "Missed_Alert_Rate", "Brier"],
            ["False-Alarm Rate", "Missed-Alert Rate", "Brier Score"]):
        x = np.arange(len(present))
        for i, h in enumerate(HORIZONS):
            vals = []
            for m in present:
                v = results_df[(results_df["Model"] == m) &
                               (results_df["Horizon"] == h)][metric].values
                vals.append(v[0] if len(v) else 0.0)
            ax.bar(x + i * 0.25, vals, 0.25, label=f"h={h}",
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(present, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        if metric == "False_Alarm_Rate":
            ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_12_operational_metrics")


def plot_master_grid(results_df: pd.DataFrame) -> None:
    present = _present_models(results_df)
    metrics_grid = ["Sensitivity", "Specificity", "Precision", "F1",
                    "AUROC", "AUPRC", "False_Alarm_Rate",
                    "Missed_Alert_Rate", "Brier"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, metric in zip(axes.ravel(), metrics_grid):
        x = np.arange(len(present))
        for i, h in enumerate(HORIZONS):
            vals = []
            for m in present:
                v = results_df[(results_df["Model"] == m) &
                               (results_df["Horizon"] == h)][metric].values
                vals.append(v[0] if len(v) else 0.0)
            ax.bar(x + i * 0.25, vals, 0.25, label=f"h={h}",
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    save_fig(fig, "fig5_13_master_grid")


def plot_threshold_sensitivity(prob_store: dict, label_store: dict) -> None:
    yte = label_store[1]
    ths = np.arange(0.05, 0.96, 0.01)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODEL_ORDER:
        pt = prob_store.get((m, 1))
        if pt is None:
            continue
        f1s = [f1_score(yte, (pt >= t).astype(int), zero_division=0) for t in ths]
        ax.plot(ths, f1s, lw=1.6, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("F1-score")
    ax.set_title("Threshold sensitivity - F1 vs decision threshold (h = 1)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig5_14_threshold_sensitivity")


def plot_confusion_matrices_h1(results_df: pd.DataFrame) -> None:
    present = _present_models(results_df)
    n_rows = (len(present) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    axes = np.array(axes).ravel()
    for ax, m in zip(axes, present):
        row = results_df[(results_df["Model"] == m) & (results_df["Horizon"] == 1)]
        if len(row) == 0:
            ax.axis("off"); continue
        cm = np.array([[row["TN"].values[0], row["FP"].values[0]],
                       [row["FN"].values[0], row["TP"].values[0]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        ax.set_title(m, fontsize=10)
    for ax in axes[len(present):]:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig5_15_confusion_matrices_h1")


def plot_district_f1(district_f1: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(district_f1["District_Std"], district_f1["F1"],
            color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("F1-score")
    ax.set_title("District-level F1 - Ensemble model (h = 1)")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, "fig5_16_district_f1")


def plot_lead_time(lead_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(lead_df["Model"], lead_df["Mean_Lead_Time_Months"],
           color=[MODEL_COLORS.get(m, "grey") for m in lead_df["Model"]],
           edgecolor="black", linewidth=0.4)
    for i, v in enumerate(lead_df["Mean_Lead_Time_Months"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean lead time (months)")
    ax.set_title("Operational lead time by model (h = 1)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_fig(fig, "fig5_17_lead_time")
