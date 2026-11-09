"""
Classifier factory. All models share a common ``.predict_proba`` API.

``LSTM-proxy`` and ``TFT-proxy`` are stand-ins trained as Gradient Boosting
to keep the pipeline lightweight and deterministic.
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warning] xgboost not installed — XGBoost model will be skipped.")


def build_models(pos_weight: float = 1.0) -> dict:
    """Return the dictionary of base classifiers for one horizon."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "LSTM-proxy": GradientBoostingClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE),
        "TFT-proxy": GradientBoostingClassifier(
            n_estimators=260, max_depth=4, learning_rate=0.03,
            random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", use_label_encoder=False, verbosity=0)
    return models
