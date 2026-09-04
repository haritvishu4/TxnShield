import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
from datetime import datetime, timedelta, timezone
import json
from src.database.connection import init_db, get_session_direct
from src.database.models import TransactionAudit
from src.risk_engine.scorer import RiskScorer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("seed_db")

SAMPLE_TRANSACTIONS = [
    # Legit transactions
    {"amount": 12.50, "prob": 0.02, "factor": "V14 (Decreases Risk)"},
    {"amount": 45.00, "prob": 0.05, "factor": "V4 (Decreases Risk)"},
    {"amount": 89.99, "prob": 0.12, "factor": "V10 (Decreases Risk)"},
    {"amount": 150.00, "prob": 0.18, "factor": "V12 (Decreases Risk)"},
    {"amount": 25.40, "prob": 0.01, "factor": "V17 (Decreases Risk)"},
    {"amount": 210.00, "prob": 0.25, "factor": "scaled_Amount (Decreases Risk)"},

    # Medium risk transactions (Prompt Step-up OTP/2FA)
    {"amount": 620.00, "prob": 0.38, "factor": "scaled_Amount (Increases Risk)"},
    {"amount": 450.50, "prob": 0.44, "factor": "V11 (Increases Risk)"},
    {"amount": 890.00, "prob": 0.58, "factor": "V2 (Increases Risk)"},
    {"amount": 750.00, "prob": 0.65, "factor": "scaled_Time (Increases Risk)"},

    # High risk transactions (Flag for Fraud Analyst Review)
    {"amount": 1450.00, "prob": 0.76, "factor": "V14 (Increases Risk)"},
    {"amount": 2300.00, "prob": 0.82, "factor": "V4 (Increases Risk)"},
    {"amount": 1890.00, "prob": 0.88, "factor": "V10 (Increases Risk)"},

    # Critical risk transactions (Hold for Manual Review)
    {"amount": 4890.00, "prob": 0.96, "factor": "V12 (Increases Risk)"},
    {"amount": 3200.00, "prob": 0.94, "factor": "V17 (Increases Risk)"},
    {"amount": 9500.00, "prob": 0.99, "factor": "V14 (Increases Risk)"}
]

def seed_database():
    config = load_config()
    metrics_path = Path(config["paths"]["metrics_path"])
    decision_threshold = 0.5
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        decision_threshold = float(metrics.get("optimal_threshold", decision_threshold))
    risk_scorer = RiskScorer(config)
    init_db()
    session = get_session_direct()
    logger.info("Seeding realistic sample transactions into SQLite audit database...")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for idx, sample in enumerate(SAMPLE_TRANSACTIONS):
        txn_time = now - timedelta(minutes=random.randint(5, 300))
        risk = risk_scorer.evaluate_transaction(sample["prob"], decision_threshold)
        entry = TransactionAudit(
            transaction_id=f"TXN-{1000 + idx:04d}",
            timestamp=txn_time,
            amount=sample["amount"],
            fraud_probability=risk["fraud_probability"],
            risk_score=risk["risk_score"],
            risk_level=risk["risk_level"],
            is_fraud=risk["is_fraud"],
            decision=risk["decision"],
            model_version="1.0.0",
            latency_ms=round(random.uniform(4.5, 14.8), 2),
            top_factor=sample["factor"],
            features_json=json.dumps({"Amount": sample["amount"], "sample_seed": True})
        )
        session.add(entry)

    session.commit()
    count = session.query(TransactionAudit).count()
    session.close()
    logger.info(f"Database seeded successfully. Total audit records: {count}")

if __name__ == "__main__":
    seed_database()
