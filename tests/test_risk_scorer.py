import pytest
from src.risk_engine.scorer import RiskScorer

def test_risk_score_calculation_bounds():
    scorer = RiskScorer()
    assert scorer.calculate_risk_score(0.0) == 0.0
    assert scorer.calculate_risk_score(1.0) == 100.0
    assert scorer.calculate_risk_score(0.45678) == 45.68
    # Edge bounds
    assert scorer.calculate_risk_score(-0.2) == 0.0
    assert scorer.calculate_risk_score(1.5) == 100.0

def test_evaluate_transaction_tiers():
    scorer = RiskScorer()
    
    # Low risk
    res_low = scorer.evaluate_transaction(0.05)
    assert res_low["risk_level"] == "Low Risk"
    assert res_low["is_fraud"] is False
    assert "Approve" in res_low["decision"]

    # Medium risk
    res_med = scorer.evaluate_transaction(0.45)
    assert res_med["risk_level"] == "Medium Risk"
    assert "Step-Up" in res_med["decision"]

    # High risk
    res_high = scorer.evaluate_transaction(0.78)
    assert res_high["risk_level"] == "High Risk"
    assert "Review" in res_high["decision"]

    # Critical risk
    res_crit = scorer.evaluate_transaction(0.95)
    assert res_crit["risk_level"] == "Critical Risk"
    assert "Hold" in res_crit["decision"]
