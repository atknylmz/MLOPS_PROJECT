from __future__ import annotations

from pathlib import Path
import pandas as pd
from prefect import flow, task

from src.config import settings
from src.utils.io import load_dataset, ensure_feature_cross
from src.validation.ge_validate import validate_training_data
from src.training.train import train_and_register
from src.registry.promote import promote_latest_to_stage

@task
def load_data() -> pd.DataFrame:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(settings.raw_csv, settings.raw_xlsx)

@task
def validate(df: pd.DataFrame) -> None:
    validate_training_data(df, settings.target_col, settings.reports_dir)

@task
def train(df: pd.DataFrame):
    return train_and_register(df)

@task
def promote_if_ok(result) -> None:
    # Simple policy: always push latest to Staging, and to Production if F1>=0.65
    promote_latest_to_stage("Staging")
    if result.metrics.get("f1", 0) >= 0.65:
        promote_latest_to_stage("Production")

@flow(name="manufacturing-defect-mlops-pipeline")
def run():
    df = load_data()
    df = ensure_feature_cross(df)
    validate(df)
    result = train(df)
    promote_if_ok(result)
    return result

if __name__ == "__main__":
    run()
