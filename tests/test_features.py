"""
Unit tests for Feature Preparation and Schema Validation.
"""

import numpy as np
import pandas as pd
import pytest

from ml.config import EXCLUDE_COLUMNS, TARGET_COLUMN
from ml.features import get_target, prepare_features


@pytest.fixture
def mock_claims_df():
    return pd.DataFrame({
        "MONTHS_AS_CUSTOMER": [100, 200, 50],
        "AGE": [35, 45, 25],
        "POLICY_STATE": ["OH", "IL", "IN"],
        "POLICY_ANNUAL_PREMIUM": [1200.5, 1400.0, 950.0],
        "TOTAL_CLAIM_AMOUNT": [50000, 75000, 12000],
        "SEVERITY_SCORE": [3, 4, 1],  # Redundant column to exclude
        "CALCULATED_CLAIM_TOTAL": [50000, 75000, 12000],  # Leakage column to exclude
        "FRAUD_FLAG": [1, 0, 1],
    })


def test_get_target(mock_claims_df):
    target = get_target(mock_claims_df)
    assert len(target) == 3
    assert set(target.unique()).issubset({0, 1})
    assert target.tolist() == [1, 0, 1]


def test_prepare_features_excludes_leakage(mock_claims_df):
    X, y, num_cols, cat_cols = prepare_features(mock_claims_df)

    # Verify target and excluded columns are stripped from X
    for col in ["FRAUD_FLAG", "SEVERITY_SCORE", "CALCULATED_CLAIM_TOTAL"]:
        assert col not in X.columns
        assert col.lower() not in X.columns

    # Verify predictors shape and types
    assert len(X) == 3
    assert "AGE" in num_cols
    assert "POLICY_STATE" in cat_cols
    assert len(y) == 3
