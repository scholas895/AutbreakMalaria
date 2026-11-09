"""
Train/val/test splitting, metric computation, and lead-time estimation.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, roc_auc_score,
)


# ------------------------------------------------------------------ split ----
def split_and_impute(panel: pd.DataFrame, features: list[str], horizon: int,
                     train_end: str, val_end: str):
    """
    Temporal split + median imputation.

    Returns
    -------
    Xtr, ytr, Xva, yva, Xte, yte, df_tr, df_va, df_te
    """
    label_col = f"Outbreak_80_h{horizon}"
    train_end = pd.Timestamp(train_end)
    val_end   = pd.Timestamp(val_end)

    tr = panel[panel["Date"] <= train_end].dropna(subset=[label_col]).copy()
    va = panel[(panel["Date"] > train_end) &
               (panel["Date"] <= val_end)].dropna(subset=[label_col]).copy()
    te = panel[panel["Date"] > val_end].dropna(subset=[label_col]).copy()

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[features])
    Xva = imp.transform(va[features])
    Xte = imp.transform(te[features])

    return (Xtr, tr[label_col].values,
            Xva, va[label_col].values,
            Xte, te[label_col].values,
            tr, va, te)


# ---------------------------------------------------------------- metrics ----
def _compute_metrics(yte, yva, pv, pt, name, h):
    """Threshold on validation F1, then evaluate on test."""
    ths = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(yva, (pv >= t).astype(int), zero_division=0) for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])

    pred = (pt >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()

    return dict(
        Model=name, Horizon=h, Threshold=round(best_t, 2),
        Sensitivity      = tp / (tp + fn) if (tp + fn) else 0.0,
        Specificity      = tn / (tn + fp) if (tn + fp) else 0.0,
        Precision        = tp / (tp + fp) if (tp + fp) else 0.0,
        F1               = f1_score(yte, pred, zero_division=0),
        Accuracy         = accuracy_score(yte, pred),
        False_Alarm_Rate = fp / (fp + tn) if (fp + tn) else 0.0,
        Missed_Alert_Rate= fn / (fn + tp) if (fn + tp) else 0.0,
        AUROC            = roc_auc_score(yte, pt),
        AUPRC            = average_precision_score(yte, pt),
        Brier            = brier_score_loss(yte, pt),
        Log_Loss         = log_loss(yte, np.clip(pt, 1e-6, 1 - 1e-6)),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )


def evaluate_model(model, Xtr, ytr, Xva, yva, Xte, yte, name, h):
    """Fit + evaluate a single classifier. Returns (metrics, pt, pv)."""
    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xva)[:, 1]
    pt = model.predict_proba(Xte)[:, 1]
    return _compute_metrics(yte, yva, pv, pt, name, h), pt, pv


def evaluate_ensemble(prob_val: dict, prob_test: dict, yva, yte, h):
    """Probability-averaged ensemble (mean over base models)."""
    ens_val  = np.mean([prob_val[k]  for k in prob_val],  axis=0)
    ens_test = np.mean([prob_test[k] for k in prob_test], axis=0)
    m = _compute_metrics(yte, yva, ens_val, ens_test, "Ensemble", h)
    return m, ens_test


# --------------------------------------------------------------- lead time ---
def mean_lead_time(y_true, y_prob, thr: float, max_lookback: int = 3) -> float:
    """Mean months between first alert and observed outbreak (<= max_lookback)."""
    alerts = (y_prob >= thr).astype(int)
    leads = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            for j in range(i, max(-1, i - max_lookback - 1), -1):
                if alerts[j] == 1:
                    leads.append(i - j)
                    break
    return float(np.mean(leads)) if leads else 0.0
