import pandas as pd
from src.utils.io import ensure_feature_cross

def test_feature_cross_created():
    df = pd.DataFrame({
        "machine_id": ["1","2"],
        "operator_id": ["10","20"],
        "DefectStatus": [0,1],
    })
    out = ensure_feature_cross(df.copy())
    assert "machine_operator" in out.columns
    assert out.loc[0, "machine_operator"] == "1_10"
