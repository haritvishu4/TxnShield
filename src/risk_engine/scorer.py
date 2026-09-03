from typing import Dict, Any
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("risk_engine")

class RiskScorer:
    """Calculates granular risk scores (0-100), risk tiers, and business decision recommendations."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.thresholds = self.config["risk_engine"]["thresholds"]
        self.tiers = self.config["risk_engine"]["tiers"]

    def calculate_risk_score(self, fraud_probability: float) -> float:
        """Converts raw fraud probability (0.0 - 1.0) to standard Risk Score (0.0 - 100.0)."""
        # Ensure bounded [0, 1]
        p = max(0.0, min(1.0, float(fraud_probability)))
        score = round(p * 100.0, 2)
        return score

    def evaluate_transaction(self, fraud_probability: float, decision_threshold: float = 0.5) -> Dict[str, Any]:
        """Returns complete risk intelligence payload including tier, score, decision, and suggested action."""
        risk_score = self.calculate_risk_score(fraud_probability)
        is_fraud_predicted = bool(fraud_probability >= decision_threshold)

        if risk_score < self.thresholds["low_max"]:
            tier_key = "LOW"
        elif risk_score < self.thresholds["medium_max"]:
            tier_key = "MEDIUM"
        elif risk_score < self.thresholds["high_max"]:
            tier_key = "HIGH"
        else:
            tier_key = "CRITICAL"

        tier_info = self.tiers[tier_key]

        return {
            "fraud_probability": round(float(fraud_probability), 4),
            "risk_score": risk_score,
            "risk_level": tier_info["label"],
            "prediction": "Potential Fraud" if is_fraud_predicted else "Legitimate Transaction",
            "is_fraud": is_fraud_predicted,
            "decision": tier_info["action"],
            "badge_color": tier_info["color"],
            "decision_threshold_used": decision_threshold
        }
