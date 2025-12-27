from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def validate_training_data(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    """
    Lightweight data validation (Great Expectations yerine).
    PDF gereksinimini karşılar, stabil ve cross-platform.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "rows": len(df),
        "columns": list(df.columns),
        "checks": {},
        "success": True
    }

    # Target kontrolü
    if target_col not in df.columns:
        report["checks"]["target_exists"] = False
        report["success"] = False
    else:
        vals = set(df[target_col].dropna().unique().tolist())
        ok = vals.issubset({0, 1})
        report["checks"]["target_binary"] = ok
        if not ok:
            report["success"] = False

    # Null oranı kontrolü
    null_rates = df.isna().mean().to_dict()
    report["checks"]["null_rates"] = null_rates
    for col, rate in null_rates.items():
        if rate > 0.05:
            report["success"] = False

    # Basit numeric sanity
    numeric_cols = df.select_dtypes(include="number").columns
    stats = {}
    for c in numeric_cols:
        stats[c] = {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean()),
        }
    report["checks"]["numeric_stats"] = stats

    # Kaydet
    out_path = out_dir / "data_validation_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not report["success"]:
        raise ValueError(f"Data validation failed. See {out_path}")
