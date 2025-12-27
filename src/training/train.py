from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import settings
from src.utils.io import ensure_feature_cross
from src.features.hashed_features import HashedCategoricalTransformer


@dataclass
class TrainResult:
    run_id: str
    metrics: dict


def _make_pipeline(numeric_cols: list[str], categorical_cols: list[str], hash_buckets: int):
    hashed = HashedCategoricalTransformer(cols=categorical_cols, n_features=hash_buckets)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("hashcat", hashed, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Ensemble pattern: combine different model families
    clf1 = LogisticRegression(max_iter=2000)
    clf2 = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    clf3 = GradientBoostingClassifier(random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gb", clf3)],
        voting="soft",
    )

    # Rebalancing pattern with SMOTE (only on training)
    pipe = ImbPipeline(
        steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=42)),
            ("model", ensemble),
        ]
    )
    return pipe


def _checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_progressive_metrics(pipe, X_train, y_train, X_test, y_test, out_dir: Path) -> None:
    """
    Train setin yüzdeliğini artırarak (%10..%100) her adımda test metriklerini loglar.
    MLflow'da step=10,15,...,100 olarak grafik çıkar.
    """
    fractions = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    n = len(X_train)
    rng = np.random.default_rng(42)
    idx_all = np.arange(n)
    rng.shuffle(idx_all)

    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for frac in fractions:
        k = max(10, int(n * frac))
        idx = idx_all[:k]

        X_sub = X_train.iloc[idx].copy()
        y_sub = y_train.iloc[idx].copy()

        # Aynı pipeline'ı clone'layıp her adımda yeniden eğitiyoruz
        m = clone(pipe)
        m.fit(X_sub, y_sub)

        prob = m.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, zero_division=0))
        rec = float(recall_score(y_test, pred, zero_division=0))
        prec = float(precision_score(y_test, pred, zero_division=0))

        step = int(round(frac * 100))

        # MLflow'a "adım adım" log
        mlflow.log_metric("progress_accuracy", acc, step=step)
        mlflow.log_metric("progress_f1", f1, step=step)
        mlflow.log_metric("progress_recall", rec, step=step)
        mlflow.log_metric("progress_precision", prec, step=step)

        rows.append({
            "train_percent": step,
            "train_rows": int(k),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    # Sunum/rapor için artifact olarak tablo kaydet
    table_path = out_dir / "progress_table.json"
    table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(table_path), artifact_path="reports")


def train_and_register(df: pd.DataFrame) -> TrainResult:
    df = ensure_feature_cross(df)

    # Identify columns
    target = settings.target_col
    if target not in df.columns:
        raise KeyError(f"Target column missing: {target}")

    cat_cols = [c for c in settings.categorical_cols if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in cat_cols + [target]]

    X = df[numeric_cols + cat_cols].copy()
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = _make_pipeline(numeric_cols, cat_cols, settings.hash_buckets)

    # MLflow setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("manufacturing_defect_mlopsl2")

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_param("hash_buckets", settings.hash_buckets)
        mlflow.log_param("categorical_cols", ",".join(cat_cols))
        mlflow.log_param("numeric_cols", ",".join(numeric_cols))
        mlflow.log_param("rebalance", "SMOTE")
        mlflow.log_param("ensemble", "Voting(LR+RF+GB)")

        # ✅ Progress metrics (10% -> 100%)
        _log_progressive_metrics(pipe, X_train, y_train, X_test, y_test, settings.reports_dir)

        # Train final model on full training data
        pipe.fit(X_train, y_train)

        # Final Metrics on test set
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }

        try:
            metrics["auc"] = float(roc_auc_score(y_test, prob))
        except Exception:
            metrics["auc"] = None

        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Checkpoint pattern
        model_path = settings.checkpoint_dir / f"model_{run.info.run_id}.joblib"
        dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="checkpoints")

        # Register model with input example
        sample_input = X_test.iloc[:5].copy()
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=settings.model_name,
            input_example=sample_input,
        )

        # Acceptance gate
        acceptance = {
            "min_f1": 0.60,
            "min_recall": 0.55,
            "passed": bool(metrics["f1"] >= 0.60 and metrics["recall"] >= 0.55),
        }
        _checkpoint(settings.checkpoint_dir / f"acceptance_{run.info.run_id}.json", acceptance)
        mlflow.log_artifact(
            str(settings.checkpoint_dir / f"acceptance_{run.info.run_id}.json"),
            artifact_path="acceptance"
        )

        return TrainResult(run_id=run.info.run_id, metrics=metrics)


if __name__ == "__main__":
    # Load data
    data_path = settings.raw_csv
    if not data_path.exists():
        data_path = settings.raw_xlsx
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path) if str(data_path).endswith('.csv') else pd.read_excel(data_path)
    
    print(f"Training model on {len(df)} samples...")
    result = train_and_register(df)
    
    print(f"Training completed!")
    print(f"Run ID: {result.run_id}")
    print(f"Metrics: {result.metrics}")
