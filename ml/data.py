"""
Data Access Layer for Insurance Claims Intelligence Platform.

Provides secure Snowflake connectivity to fetch the analytical fraud feature table:
INSURANCE_DB.ANALYTICS.FRAUD_MODEL_DATASET.
Includes graceful local fallback for offline development and testing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from ml.config import (
    SNOWFLAKE_CONFIG,
    LOCAL_DATA_PATH,
    TARGET_COLUMN,
    TARGET_COLUMN_LOWER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def connect_snowflake():
    """
    Establishes connection to Snowflake using environment configuration.
    Returns snowflake.connector connection object or raises Exception.
    """
    import snowflake.connector

    missing_keys = [
        k for k in ["account", "user", "password"] if not SNOWFLAKE_CONFIG.get(k)
    ]
    if missing_keys:
        raise ValueError(f"Missing required Snowflake credentials: {missing_keys}")

    conn = snowflake.connector.connect(
        account=SNOWFLAKE_CONFIG["account"],
        user=SNOWFLAKE_CONFIG["user"],
        password=SNOWFLAKE_CONFIG["password"],
        role=SNOWFLAKE_CONFIG["role"],
        warehouse=SNOWFLAKE_CONFIG["warehouse"],
        database=SNOWFLAKE_CONFIG["database"],
        schema=SNOWFLAKE_CONFIG["schema"],
    )
    return conn


def load_from_snowflake(table_name: str = "FRAUD_MODEL_DATASET") -> pd.DataFrame:
    """
    Executes query against Snowflake to retrieve analytical feature dataset.
    """
    logger.info("Attempting connection to Snowflake database '%s.%s'...", 
                SNOWFLAKE_CONFIG.get("database"), SNOWFLAKE_CONFIG.get("schema"))
    conn = connect_snowflake()
    try:
        query = f"SELECT * FROM {table_name}"
        logger.info("Executing Snowflake query: %s", query)
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        logger.info("Successfully fetched %d records from Snowflake table '%s'", len(df), table_name)
        return df
    finally:
        conn.close()


def load_from_local(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads local analytical dataset (for offline development and testing).
    """
    path = file_path or LOCAL_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found at: {path}")
    logger.info("Loading analytical dataset from local fallback: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d records from %s", len(df), path.name)
    return df


def load_dataset(prefer_snowflake: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Loads data using production Snowflake flow first; gracefully falls back to local data if unconfigured.
    Returns:
        Tuple of (pd.DataFrame, source_description)
    """
    if prefer_snowflake:
        try:
            df = load_from_snowflake()
            # Standardize columns to uppercase
            df.columns = [c.upper() for c in df.columns]
            validate_dataset(df)
            return df, "Snowflake (INSURANCE_DB.ANALYTICS.FRAUD_MODEL_DATASET)"
        except Exception as e:
            logger.warning("Snowflake query unavailable (%s). Falling back to local data.", e)

    df = load_from_local()
    df.columns = [c.upper() for c in df.columns]
    validate_dataset(df)
    return df, f"Local analytical store ({LOCAL_DATA_PATH.name})"


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates that the dataset conforms to required schema expectations.
    """
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    if TARGET_COLUMN not in df.columns and TARGET_COLUMN_LOWER not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset columns: {df.columns.tolist()}"
        )

    # Validate target column values
    target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else TARGET_COLUMN_LOWER
    unique_vals = set(df[target_col].dropna().unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target column contains unexpected values: {unique_vals}. Expected binary {0, 1}.")
    
    logger.info("Dataset validation passed: %d rows, %d columns. Target distribution: %s", 
                len(df), len(df.columns), dict(df[target_col].value_counts()))
