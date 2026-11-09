# Methodology Notes — Experiment 2

## 1. Objective

Predict **district-level malaria outbreak onset** in Burundi at 1-, 2- and
3-month horizons from recent case counts, seasonality and environmental
covariates.

## 2. Label definition

For each district `d`:

- Compute `Threshold_80(d)` = 80th percentile of monthly cases over the
  **training** period (`Date <= 2023-12-01`).
- `Outbreak_80(d, t) = 1` iff `Cases(d, t) > Threshold_80(d)`.

At time `t`, the model predicts `Outbreak_80(d, t+h)` for `h in {1, 2, 3}`.

This "exceed-the-80th-percentile" definition leads to a ~20-23% positive
class across all district-months, which is realistic for a rare-event
early-warning task.

## 3. Temporal split

| Split | Date range |
|---|---|
| **Train** | 2022-01-01 -> 2023-12-01 |
| **Validation** | 2023-12-02 -> 2024-06-01 (used for threshold selection) |
| **Test** | 2024-06-02 -> 2024-12-01 |

Because the target is defined at `t+h`, `NaN` rows are dropped per split.

## 4. Feature set (22 features)

- `Cases_Lag1..3` — past cases.
- `Cases_Roll3`, `Cases_Roll6` — recent means.
- `Month_Sin`, `Month_Cos` — cyclic month.
- `Temp`, `Humidity`, `Rainfall`, `Lake_Level`, `River_Level` — national
  monthly means.
- `{covariate}_Lag1..2` — 1-2 month lags of each environmental covariate.

Missing values are median-imputed, with the imputer fit on the training
split only.

## 5. Models

| Name | Backend |
|---|---|
| Logistic Regression | `sklearn` LR with balanced class weights |
| Random Forest | 400 trees, depth 10, balanced class weights |
| XGBoost | 350 trees, depth 6, `scale_pos_weight` = neg/pos |
| LSTM-proxy | `GradientBoostingClassifier` — proxy for LSTM |
| TFT-proxy | `GradientBoostingClassifier` — proxy for TFT |
| Ensemble | Mean of predicted probabilities from all base models |

> **Note on proxies.** `LSTM-proxy` and `TFT-proxy` are **not** real neural
> networks — they are small gradient-boosted models used as deterministic,
> lightweight stand-ins in this experiment. When the pipeline is promoted
> to production, these slots can be swapped for real recurrent models
> without changing the surrounding API.

## 6. Decision threshold

For each (model, horizon), the operating threshold is chosen on the
**validation** set to maximize F1, then **frozen** for the test set.

## 7. Metrics reported

| Category | Metrics |
|---|---|
| Discrimination | AUROC, AUPRC |
| Calibration | Brier, Log-Loss |
| Operational | Sensitivity, Specificity, Precision, F1, Accuracy |
| Operational (error) | False-Alarm Rate, Missed-Alert Rate |
| Temporal | Mean lead time (<= 3-month look-back) |

## 8. Reproducibility

- Global `numpy` seed = 42
- All `sklearn` / `xgboost` models set `random_state=42`
- Requirements pinned in `requirements.txt`

## 9. Known limitations

- National (not district) environmental averages.
- Proxy models for LSTM/TFT.
- Threshold selection at h = 1 uses the same validation split as h = 3 —
  in a strict setting, rolling-origin evaluation would be preferable.
