"""
Unit tests for End-to-End Model Prediction and Risk Tier Logic.
"""

import pytest
from ml.config import get_risk_level
from ml.predict import predict_claim_risk
from ml.utils import get_sample_claims


def test_get_risk_level_thresholds():
    # Operating threshold = 0.35, High threshold = 0.60
    assert get_risk_level(0.75, threshold=0.35) == "HIGH RISK"
    assert get_risk_level(0.60, threshold=0.35) == "HIGH RISK"
    assert get_risk_level(0.45, threshold=0.35) == "MEDIUM RISK"
    assert get_risk_level(0.35, threshold=0.35) == "MEDIUM RISK"
    assert get_risk_level(0.20, threshold=0.35) == "LOW RISK"
    assert get_risk_level(0.05, threshold=0.35) == "LOW RISK"


def test_predict_claim_risk_end_to_end():
    sample_claims = get_sample_claims()
    assert len(sample_claims) >= 2

    for name, claim_payload in sample_claims.items():
        result = predict_claim_risk(claim_payload)

        assert "fraud_probability" in result
        assert 0.0 <= result["fraud_probability"] <= 100.0
        assert result["risk_level"] in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
        assert "is_flagged_for_investigation" in result
        assert isinstance(result["risk_signals"], list)
