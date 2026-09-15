"""
Snowflake Data Connector & Query Manager for Streamlit Dashboard.

Executes cached analytical queries against INSURANCE_DB.ANALYTICS views.
Includes automatic offline fallback to local analytical store.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from ml.config import LOCAL_DATA_PATH, SNOWFLAKE_CONFIG

logger = logging.getLogger(__name__)


def _get_snowflake_credentials() -> Dict[str, Any]:
    """
    Extracts credentials prioritizing Streamlit secrets, then environment variables.
    """
    config = dict(SNOWFLAKE_CONFIG)
    if hasattr(st, "secrets") and "snowflake" in st.secrets:
        sec = st.secrets["snowflake"]
        for key in ["account", "user", "password", "role", "warehouse", "database", "schema"]:
            if key in sec:
                config[key] = sec[key]
    return config


def _execute_snowflake_query(query: str) -> pd.DataFrame:
    """Connects to Snowflake and returns query results as a DataFrame."""
    import snowflake.connector
    cfg = _get_snowflake_credentials()
    if not cfg.get("account") or not cfg.get("user") or not cfg.get("password"):
        raise ConnectionError("Missing Snowflake account credentials.")

    conn = snowflake.connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        role=cfg.get("role", "ACCOUNTADMIN"),
        warehouse=cfg.get("warehouse", "COMPUTE_WH"),
        database=cfg.get("database", "INSURANCE_DB"),
        schema=cfg.get("schema", "ANALYTICS"),
    )
    try:
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = [c.upper() for c in df.columns]
        return df
    finally:
        conn.close()


@st.cache_data(ttl=600)
def load_analytical_base_data() -> Tuple[pd.DataFrame, bool, str]:
    """
    Loads full dataset from Snowflake or local fallback.
    Returns: (DataFrame, is_live_snowflake, source_status_string)
    """
    try:
        df = _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.FRAUD_MODEL_DATASET")
        return df, True, "Connected to Snowflake (INSURANCE_DB.ANALYTICS)"
    except Exception as err:
        logger.info("Using local analytical fallback (%s)", err)
        df = pd.read_csv(LOCAL_DATA_PATH)
        df.columns = [c.upper() for c in df.columns]
        return df, False, "Offline Mode (Local Analytical Cache)"


@st.cache_data(ttl=600)
def get_kpis() -> Dict[str, Any]:
    """
    Queries VW_CLAIM_KPIS or computes exact aggregation locally.
    """
    try:
        df = _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_CLAIM_KPIS")
        row = df.iloc[0].to_dict()
        return {k.lower(): v for k, v in row.items()}
    except Exception:
        df, _, _ = load_analytical_base_data()
        total_claims = len(df)
        fraud_claims = int((df["FRAUD_FLAG"] == 1).sum())
        total_amount = float(df["TOTAL_CLAIM_AMOUNT"].sum())
        avg_amount = float(df["TOTAL_CLAIM_AMOUNT"].mean())
        med_amount = float(df["TOTAL_CLAIM_AMOUNT"].median())
        fraud_rate = round(100.0 * fraud_claims / total_claims, 2)
        avg_premium = float(df["POLICY_ANNUAL_PREMIUM"].mean())
        avg_age = float(df["AGE"].mean())

        return {
            "total_claims": total_claims,
            "total_claim_amount": total_amount,
            "average_claim_amount": avg_amount,
            "median_claim_amount": med_amount,
            "fraudulent_claims": fraud_claims,
            "fraud_rate": fraud_rate,
            "average_annual_premium": avg_premium,
            "average_customer_age": avg_age,
        }


@st.cache_data(ttl=600)
def get_claims_by_state() -> pd.DataFrame:
    """Queries VW_CLAIMS_BY_STATE or aggregates locally."""
    try:
        return _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_CLAIMS_BY_STATE")
    except Exception:
        df, _, _ = load_analytical_base_data()
        grouped = df.groupby("INCIDENT_STATE").agg(
            CLAIM_COUNT=("TOTAL_CLAIM_AMOUNT", "count"),
            TOTAL_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "sum"),
            AVERAGE_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "mean"),
            FRAUDULENT_CLAIMS=("FRAUD_FLAG", lambda s: (s == 1).sum()),
        ).reset_index()
        grouped["FRAUD_RATE"] = (grouped["FRAUDULENT_CLAIMS"] / grouped["CLAIM_COUNT"] * 100).round(2)
        return grouped.sort_values(by="CLAIM_COUNT", ascending=False)


@st.cache_data(ttl=600)
def get_claims_by_incident_type() -> pd.DataFrame:
    """Queries VW_CLAIMS_BY_INCIDENT_TYPE or aggregates locally."""
    try:
        return _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_CLAIMS_BY_INCIDENT_TYPE")
    except Exception:
        df, _, _ = load_analytical_base_data()
        grouped = df.groupby("INCIDENT_TYPE").agg(
            CLAIM_COUNT=("TOTAL_CLAIM_AMOUNT", "count"),
            TOTAL_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "sum"),
            AVERAGE_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "mean"),
            FRAUDULENT_CLAIMS=("FRAUD_FLAG", lambda s: (s == 1).sum()),
        ).reset_index()
        return grouped.sort_values(by="CLAIM_COUNT", ascending=False)


@st.cache_data(ttl=600)
def get_claims_by_severity() -> pd.DataFrame:
    """Queries VW_CLAIMS_BY_SEVERITY or aggregates locally."""
    try:
        return _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_CLAIMS_BY_SEVERITY")
    except Exception:
        df, _, _ = load_analytical_base_data()
        grouped = df.groupby("INCIDENT_SEVERITY").agg(
            CLAIM_COUNT=("TOTAL_CLAIM_AMOUNT", "count"),
            TOTAL_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "sum"),
            AVERAGE_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "mean"),
            FRAUDULENT_CLAIMS=("FRAUD_FLAG", lambda s: (s == 1).sum()),
        ).reset_index()
        grouped["FRAUD_RATE"] = (grouped["FRAUDULENT_CLAIMS"] / grouped["CLAIM_COUNT"] * 100).round(2)
        return grouped.sort_values(by="CLAIM_COUNT", ascending=False)


@st.cache_data(ttl=600)
def get_monthly_claims() -> pd.DataFrame:
    """Queries VW_MONTHLY_CLAIMS or aggregates locally."""
    try:
        return _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_MONTHLY_CLAIMS")
    except Exception:
        df, _, _ = load_analytical_base_data()
        df["INCIDENT_PERIOD"] = (
            df["INCIDENT_YEAR"].astype(str) + "-" + df["INCIDENT_MONTH"].astype(str).str.zfill(2)
        )
        grouped = df.groupby("INCIDENT_PERIOD").agg(
            CLAIM_COUNT=("TOTAL_CLAIM_AMOUNT", "count"),
            TOTAL_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "sum"),
            AVERAGE_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "mean"),
            FRAUDULENT_CLAIMS=("FRAUD_FLAG", lambda s: (s == 1).sum()),
        ).reset_index()
        return grouped.sort_values(by="INCIDENT_PERIOD")


@st.cache_data(ttl=600)
def get_fraud_analysis() -> pd.DataFrame:
    """Queries VW_FRAUD_ANALYSIS or aggregates locally."""
    try:
        return _execute_snowflake_query("SELECT * FROM INSURANCE_DB.ANALYTICS.VW_FRAUD_ANALYSIS")
    except Exception:
        df, _, _ = load_analytical_base_data()
        grouped = df.groupby("FRAUD_FLAG").agg(
            CLAIM_COUNT=("TOTAL_CLAIM_AMOUNT", "count"),
            TOTAL_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "sum"),
            AVERAGE_CLAIM_AMOUNT=("TOTAL_CLAIM_AMOUNT", "mean"),
            AVERAGE_POLICY_PREMIUM=("POLICY_ANNUAL_PREMIUM", "mean"),
            AVERAGE_CUSTOMER_AGE=("AGE", "mean"),
            AVG_VEHICLES_INVOLVED=("NUMBER_OF_VEHICLES_INVOLVED", "mean"),
            AVERAGE_WITNESSES=("WITNESSES", "mean"),
            AVERAGE_BODILY_INJURIES=("BODILY_INJURIES", "mean"),
        ).reset_index()
        grouped["FRAUD_STATUS"] = grouped["FRAUD_FLAG"].map({0: "Non-Fraud (0)", 1: "Fraud (1)"})
        return grouped
