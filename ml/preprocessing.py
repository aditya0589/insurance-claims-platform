"""
Preprocessing Pipelines for Insurance Claims Features.

Builds leakage-free scikit-learn ColumnTransformer objects encapsulating
imputation, scaling, and categorical encoding.
"""

from typing import List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles missing value imputation,
    standard scaling for numeric features, and one-hot encoding for categoricals.

    Parameters:
        numerical_features: List of continuous / discrete numeric column names
        categorical_features: List of categorical / string column names

    Returns:
        Configured scikit-learn ColumnTransformer
    """
    # Numerical pipeline: impute missing values with median, scale features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing values with mode, encode with unknown handling
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def extract_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts post-transformation feature names (including one-hot dummy column names).
    """
    try:
        return fitted_preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback for earlier scikit-learn or manual assembly
        names = []
        for name, trans, cols in fitted_preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols).tolist())
            elif hasattr(trans, "named_steps") and hasattr(trans.named_steps.get("onehot"), "get_feature_names_out"):
                names.extend(trans.named_steps["onehot"].get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)
        return names
