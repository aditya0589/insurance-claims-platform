"""
Centralized Configuration for Insurance Claims Intelligence & Fraud Detection Platform.

Defines reproducible hyperparameters, paths, thresholds, and environment settings.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ML_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_PATH = BASE_DIR / "data" / "fraud_model_dataset.csv"

# Target & Column Handling
TARGET_COLUMN = "FRAUD_FLAG"
TARGET_COLUMN_LOWER = "fraud_flag"

# Columns to exclude from training (data leakage or redundancy with raw features)
EXCLUDE_COLUMNS = [
    "FRAUD_FLAG",
    "fraud_flag",
    "SEVERITY_SCORE",
    "severity_score",
    "CALCULATED_CLAIM_TOTAL",
    "calculated_claim_total",
    "_C39",
    "_c39",
]

# Train-Test Split Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Modeling Configuration
PRIMARY_METRIC = "roc_auc"
SECONDARY_METRICS = ["fraud_recall", "fraud_f1", "fraud_precision", "accuracy"]

# Candidate operating thresholds for evaluation
THRESHOLD_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
DEFAULT_OPERATING_THRESHOLD = 0.30

# Business Risk Bands based on probability
RISK_LEVEL_HIGH_THRESHOLD = 0.60
RISK_LEVEL_MEDIUM_THRESHOLD = 0.30

def get_risk_level(prob: float, threshold: float = DEFAULT_OPERATING_THRESHOLD) -> str:
    """
    Categorizes predicted fraud probability into business-friendly risk tiers.
    """
    if prob >= RISK_LEVEL_HIGH_THRESHOLD:
        return "HIGH RISK"
    elif prob >= threshold:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(BASE_DIR / 'mlruns.db').as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "insurance_claim_fraud_detection")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "insurance_fraud_model")
CHAMPION_ALIAS = "champion"

# Snowflake Configuration
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "INSURANCE_DB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS"),
}
