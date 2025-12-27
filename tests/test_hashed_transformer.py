import pandas as pd
from src.features.hashed_features import HashedCategoricalTransformer

def test_hashed_transformer_shape():
    df = pd.DataFrame({
        "machine_id": ["1","2","3"],
        "operator_id": ["10","20","30"],
        "supplier_batch_id": ["7","8","9"],
        "production_line_id": ["1","1","2"],
        "machine_operator": ["1_10","2_20","3_30"],
    })
    tr = HashedCategoricalTransformer(cols=list(df.columns), n_features=128)
    X = tr.transform(df)
    assert X.shape[0] == 3
    assert X.shape[1] == 128
