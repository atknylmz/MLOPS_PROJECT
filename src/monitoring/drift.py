from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.config import settings
from src.utils.io import ensure_feature_cross

@dataclass
class DriftReport:
    unseen_rate: dict
    psi_numeric: dict
    metrics: dict | None

def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index (PSI) for numeric drift."""
    # remove nan
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    def bin_counts(x):
        c, _ = np.histogram(x, bins=quantiles)
        p = c / max(1, len(x))
        return np.clip(p, 1e-6, 1.0)

    e = bin_counts(expected)
    a = bin_counts(actual)
    return float(np.sum((a - e) * np.log(a / e)))

def load_production_model():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.model_name}/Production"
    return mlflow.sklearn.load_model(uri)

def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, out_path: Path, y_true_col: str | None = None) -> DriftReport:
    reference = ensure_feature_cross(reference)
    current = ensure_feature_cross(current)

    cat_cols = [c for c in settings.categorical_cols if c in reference.columns and c in current.columns]
    target = settings.target_col

    # unseen category rate
    unseen = {}
    for c in cat_cols:
        ref_vals = set(reference[c].astype(str).unique().tolist())
        cur_vals = current[c].astype(str).unique().tolist()
        unseen[c] = float(sum(v not in ref_vals for v in cur_vals) / max(1, len(set(cur_vals))))

    # numeric PSI
    psi = {}
    numeric_cols = [c for c in reference.columns if c not in cat_cols + [target]]
    for c in numeric_cols:
        psi[c] = _psi(reference[c].to_numpy(dtype=float), current[c].to_numpy(dtype=float))

    metrics = None
    if y_true_col and y_true_col in current.columns:
        model = load_production_model()
        X_cols = [c for c in current.columns if c != y_true_col]
        prob = model.predict_proba(current[X_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        y_true = current[y_true_col].astype(int).to_numpy()
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = DriftReport(unseen_rate=unseen, psi_numeric=psi, metrics=metrics)
    out_path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    return report
