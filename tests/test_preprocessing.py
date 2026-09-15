"""
Unit tests for Preprocessing Pipeline and Missing/Unseen Value Handling.
"""

import numpy as np
import pandas as pd
import pytest
from ml.preprocessing import build_preprocessor, extract_transformed_feature_names


def test_preprocessing_handles_missing_values():
    # DataFrame with missing numeric and categorical values
    train_df = pd.DataFrame({
        "NUM_1": [10.0, np.nan, 30.0, 40.0],
        "CAT_1": ["A", "B", None, "A"],
    })

    num_cols = ["NUM_1"]
    cat_cols = ["CAT_1"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(train_df)

    # Output should have no NaNs
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 4


def test_preprocessing_handles_unseen_categories():
    train_df = pd.DataFrame({
        "NUM_1": [10.0, 20.0],
        "CAT_1": ["A", "B"],
    })
    test_df = pd.DataFrame({
        "NUM_1": [15.0],
        "CAT_1": ["UNSEEN_CATEGORY_XYZ"],
    })

    preprocessor = build_preprocessor(["NUM_1"], ["CAT_1"])
    preprocessor.fit(train_df)

    # Should not raise ValueError on unseen category because handle_unknown='ignore'
    transformed_test = preprocessor.transform(test_df)
    assert transformed_test.shape[0] == 1
    assert not np.isnan(transformed_test).any()
