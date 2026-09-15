"""
Prediction and Inference Engine for Insurance Fraud Risk Scoring.

Loads the registered MLflow champion model (with automatic fallback to local artifact)
and calculates calibrated fraud risk scores and categorization tiers.
"""

import sys
from pathlib import Path

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Any, Dict, List, Union
import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from ml.config import (
    ARTIFACTS_DIR,
    CHAMPION_ALIAS,
    DEFAULT_OPERATING_THRESHOLD,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    get_risk_level,
)

logger = logging.getLogger(__name__)

_CACHED_MODEL = None
_CACHED_METADATA = None


def load_model_and_metadata():
    """
    Loads the trained champion model pipeline and metadata.
    Attempts MLflow model registry first, falling back to local artifacts.
    """
    global _CACHED_MODEL, _CACHED_METADATA
    if _CACHED_MODEL is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL, _CACHED_METADATA

    metadata_path = ARTIFACTS_DIR / "champion_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # 1. Attempt MLflow Model Registry
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
        logger.info("Attempting to load champion model from MLflow: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Successfully loaded champion model from MLflow registry.")
        _CACHED_MODEL = model
        _CACHED_METADATA = metadata
        return _CACHED_MODEL, _CACHED_METADATA
    except Exception as mlflow_err:
        logger.warning(
            "Could not load model from MLflow registry (%s). Falling back to local artifact.",
            mlflow_err,
        )

    # 2. Local artifact fallback
    local_path = ARTIFACTS_DIR / "champion_model.joblib"
    if not local_path.exists():
        raise FileNotFoundError(
            f"No champion model found in MLflow or at {local_path}. Please execute 'python ml/train.py' first."
        )

    logger.info("Loading model from local artifact: %s", local_path)
    _CACHED_MODEL = joblib.load(local_path)
    _CACHED_METADATA = metadata
    return _CACHED_MODEL, _CACHED_METADATA


def predict_claim_risk(claim_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Predicts fraud risk score and business tier for an incoming claim.

    Parameters:
        claim_data: Dict or DataFrame containing claim features.

    Returns:
        Dict with fraud_probability, risk_level, threshold, and feature summary.
    """
    model, metadata = load_model_and_metadata()
    threshold = metadata.get("operating_threshold", DEFAULT_OPERATING_THRESHOLD)

    if isinstance(claim_data, dict):
        # Normalize keys to uppercase
        normalized_data = {k.upper(): [v] for k, v in claim_data.items()}
        df_input = pd.DataFrame(normalized_data)
    elif isinstance(claim_data, pd.DataFrame):
        df_input = claim_data.copy()
        df_input.columns = [c.upper() for c in df_input.columns]
    else:
        raise TypeError(f"Expected dict or DataFrame, received {type(claim_data)}")

    # Ensure all expected feature columns exist (fill missing with None/NaN so Pipeline imputer handles it)
    expected_cols = metadata.get("feature_names", [])
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    # Predict probability of fraud (class 1)
    proba = model.predict_proba(df_input)[:, 1]
    predicted_prob = float(proba[0])
    risk_level = get_risk_level(predicted_prob, threshold=threshold)

    # Key risk indicator signals
    signals = []
    total_claim = df_input.get("TOTAL_CLAIM_AMOUNT", [0]).iloc[0] if "TOTAL_CLAIM_AMOUNT" in df_input else 0
    claim_ratio = df_input.get("CLAIM_TO_PREMIUM_RATIO", [0]).iloc[0] if "CLAIM_TO_PREMIUM_RATIO" in df_input else 0
    incident_severity = df_input.get("INCIDENT_SEVERITY", [""]).iloc[0] if "INCIDENT_SEVERITY" in df_input else ""
    police_rep = df_input.get("POLICE_REPORT_AVAILABLE", [""]).iloc[0] if "POLICE_REPORT_AVAILABLE" in df_input else ""
    hour = df_input.get("INCIDENT_HOUR_OF_THE_DAY", [-1]).iloc[0] if "INCIDENT_HOUR_OF_THE_DAY" in df_input else -1

    if claim_ratio is not None and claim_ratio > 30:
        signals.append(f"High Claim-to-Premium Ratio ({claim_ratio:.1f}x annual premium)")
    if incident_severity in ["Major Damage", "Total Loss"]:
        signals.append(f"Severe Incident Category ({incident_severity})")
    if police_rep == "NO" or police_rep is None:
        signals.append("No Official Police Report Filed")
    if hour in [0, 1, 2, 3, 4, 23]:
        signals.append(f"Late Night / Early Morning Incident ({hour}:00 hrs)")
    if total_claim is not None and total_claim > 60000:
        signals.append(f"High Total Incurred Amount (${total_claim:,.2f})")

    return {
        "fraud_probability": round(predicted_prob * 100, 2),
        "raw_probability": round(predicted_prob, 4),
        "risk_level": risk_level,
        "operating_threshold": round(threshold, 2),
        "is_flagged_for_investigation": bool(predicted_prob >= threshold),
        "risk_signals": signals,
        "model_champion": metadata.get("champion_model_name", "Random Forest"),
        "model_version_run": metadata.get("run_id", "local"),
    }


if __name__ == "__main__":
    # Test single-claim inference with sample claim
    sample_claim = {
        "months_as_customer": 228,
        "age": 44,
        "policy_state": "IL",
        "policy_csl": "500/1000",
        "policy_deductible": 1000,
        "policy_annual_premium": 1583.91,
        "umbrella_limit": 6000000,
        "insured_sex": "MALE",
        "insured_education_level": "Associate",
        "insured_occupation": "sales",
        "insured_relationship": "not-in-family",
        "policy_tenure_months": 228,
        "policy_bind_year": 2014,
        "incident_type": "Multi-vehicle Collision",
        "collision_type": "Rear Collision",
        "incident_severity": "Major Damage",
        "authorities_contacted": "Police",
        "incident_state": "NY",
        "incident_city": "Columbus",
        "incident_hour_of_the_day": 2,
        "number_of_vehicles_involved": 3,
        "property_damage": "NO",
        "bodily_injuries": 1,
        "witnesses": 2,
        "police_report_available": "NO",
        "capital_gains": 0,
        "capital_loss": -46000,
        "total_claim_amount": 75000,
        "injury_claim": 15000,
        "property_claim": 15000,
        "vehicle_claim": 45000,
        "claim_to_premium_ratio": 47.35,
        "auto_make": "Saab",
        "auto_model": "92x",
        "auto_year": 2004,
        "vehicle_age": 11,
        "incident_year": 2015,
        "incident_month": 2,
    }
    result = predict_claim_risk(sample_claim)
    print("\nSample Claim Risk Scoring Result:")
    print(json.dumps(result, indent=2))
