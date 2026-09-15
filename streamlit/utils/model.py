"""
Model Serving and Inference Helpers for Streamlit.

Uses st.cache_resource to load the champion pipeline once and serve predictions.
"""

from typing import Any, Dict
import streamlit as st

from ml.predict import load_model_and_metadata, predict_claim_risk


@st.cache_resource(show_spinner="Loading trained ML model & artifacts...")
def get_cached_champion_model():
    """
    Cached loader for the trained scikit-learn pipeline and metadata.
    """
    return load_model_and_metadata()


def score_single_claim(claim_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scores a single insurance claim payload.
    """
    return predict_claim_risk(claim_dict)
