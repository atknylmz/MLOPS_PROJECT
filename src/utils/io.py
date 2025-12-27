from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_dataset(csv_path: Path, xlsx_path: Path) -> pd.DataFrame:
    """Load CSV if exists, else XLSX. Raises if neither exists."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError(f"Dataset not found: {csv_path} or {xlsx_path}")

def ensure_feature_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the FEATURE CROSS column exists."""
    if "machine_operator" not in df.columns and {"machine_id","operator_id"}.issubset(df.columns):
        df["machine_operator"] = df["machine_id"].astype(str) + "_" + df["operator_id"].astype(str)
    return df
