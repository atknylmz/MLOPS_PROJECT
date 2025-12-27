from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    # Data paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_csv: Path = data_dir / "manufacturing_defect_dataset_merged.csv"
    raw_xlsx: Path = data_dir / "manufacturing_defect_dataset_enriched.xlsx"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: str = "./mlruns"
    model_name: str = "manufacturing_defect_model"

    # Checkpoints / artifacts
    artifacts_dir: Path = project_root / "artifacts"
    checkpoint_dir: Path = artifacts_dir / "checkpoints"
    reports_dir: Path = artifacts_dir / "reports"

    # Feature engineering
    hash_buckets: int = 2**12  # 4096 buckets
    categorical_cols: tuple[str, ...] = (
        "machine_id",
        "operator_id",
        "supplier_batch_id",
        "production_line_id",
        "machine_operator",
        "ShiftName",
        "ShiftType",
        "MaterialCode",
        "MaterialDesc",
        "MaterialType",
        "EnergySourceType",
        "IncomingInspectionResult",
        "SupplierCountry",
        "MachineCategory",
        "MachineDesc",
    )
    target_col: str = "DefectStatus"

settings = Settings()
