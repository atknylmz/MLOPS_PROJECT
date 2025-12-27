from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

class HashedCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Hashed Feature pattern for high-cardinality categorical columns.

    Transforms multiple categorical columns into a fixed-size sparse vector using FeatureHasher.
    """
    def __init__(self, cols: list[str], n_features: int = 4096):
        self.cols = cols
        self.n_features = n_features
        self._hasher = FeatureHasher(n_features=self.n_features, input_type="pair")

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        # Build list-of-pairs per row: [(col, value), ...]
        pairs = []
        for _, row in X[self.cols].astype(str).iterrows():
            pairs.append([(c, row[c]) for c in self.cols])
        return self._hasher.transform(pairs)
