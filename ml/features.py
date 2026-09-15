"""
Feature Engineering & Schema Definitions for Fraud Detection.

Handles feature identification, target separation, leakage prevention,
and train/serving schema alignment.
"""

from typing import Dict, List, Tuple
import pandas as pd
from ml.config import EXCLUDE_COLUMNS, TARGET_COLUMN, TARGET_COLUMN_LOWER


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extracts the binary target variable series.
    """
    if TARGET_COLUMN in df.columns:
        return df[TARGET_COLUMN].astype(int)
    elif TARGET_COLUMN_LOWER in df.columns:
        return df[TARGET_COLUMN_LOWER].astype(int)
    else:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separates predictors X and target y, strips data leakage/redundant features,
    and categorizes features into numerical and categorical types.

    Returns:
        Tuple of (X, y, numerical_features, categorical_features)
    """
    y = get_target(df)

    # Identify drop columns (case-insensitive)
    cols_to_drop = [c for c in df.columns if c.upper() in [x.upper() for x in EXCLUDE_COLUMNS]]
    X = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    return X, y, numerical_features, categorical_features


def get_feature_domain_groups() -> Dict[str, List[str]]:
    """
    Returns feature definitions grouped by business domain for UI and reporting.
    """
    return {
        "Customer Demographics": [
            "AGE",
            "MONTHS_AS_CUSTOMER",
            "INSURED_SEX",
            "INSURED_EDUCATION_LEVEL",
            "INSURED_OCCUPATION",
            "INSURED_RELATIONSHIP",
        ],
        "Policy Information": [
            "POLICY_STATE",
            "POLICY_CSL",
            "POLICY_DEDUCTIBLE",
            "POLICY_ANNUAL_PREMIUM",
            "UMBRELLA_LIMIT",
            "POLICY_TENURE_MONTHS",
            "POLICY_BIND_YEAR",
        ],
        "Incident Details": [
            "INCIDENT_TYPE",
            "COLLISION_TYPE",
            "INCIDENT_SEVERITY",
            "AUTHORITIES_CONTACTED",
            "INCIDENT_STATE",
            "INCIDENT_CITY",
            "INCIDENT_HOUR_OF_THE_DAY",
            "NUMBER_OF_VEHICLES_INVOLVED",
            "PROPERTY_DAMAGE",
            "BODILY_INJURIES",
            "WITNESSES",
            "POLICE_REPORT_AVAILABLE",
            "INCIDENT_YEAR",
            "INCIDENT_MONTH",
        ],
        "Financial & Claim": [
            "CAPITAL_GAINS",
            "CAPITAL_LOSS",
            "TOTAL_CLAIM_AMOUNT",
            "INJURY_CLAIM",
            "PROPERTY_CLAIM",
            "VEHICLE_CLAIM",
            "CLAIM_TO_PREMIUM_RATIO",
        ],
        "Vehicle Details": [
            "AUTO_MAKE",
            "AUTO_MODEL",
            "AUTO_YEAR",
            "VEHICLE_AGE",
        ],
    }
