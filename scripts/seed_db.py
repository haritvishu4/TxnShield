import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import uuid
from datetime import datetime, timedelta
import json
from src.database.connection import init_db, get_session_direct
from src.database.models import TransactionAudit
from src.utils.logger import get_logger

logger = get_logger("seed_db")

SAMPLE_TRANSACTIONS = [
    # Legit transactions
    {"amount": 12.50, "prob": 0.02, "score": 2.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "V14 (Decreases Risk)"},
    {"amount": 45.00, "prob": 0.05, "score": 5.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "V4 (Decreases Risk)"},
    {"amount": 89.99, "prob": 0.12, "score": 12.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "V10 (Decreases Risk)"},
    {"amount": 150.00, "prob": 0.18, "score": 18.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "V12 (Decreases Risk)"},
    {"amount": 25.40, "prob": 0.01, "score": 1.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "V17 (Decreases Risk)"},
    {"amount": 210.00, "prob": 0.25, "score": 25.0, "tier": "Low Risk", "is_fraud": False, "decision": "Approve Transaction", "factor": "scaled_Amount (Decreases Risk)"},

    # Medium risk transactions (Prompt Step-up OTP/2FA)
    {"amount": 620.00, "prob": 0.38, "score": 38.0, "tier": "Medium Risk", "is_fraud": False, "decision": "Step-Up Authentication (2FA/OTP)", "factor": "scaled_Amount (Increases Risk)"},
    {"amount": 450.50, "prob": 0.44, "score": 44.0, "tier": "Medium Risk", "is_fraud": False, "decision": "Step-Up Authentication (2FA/OTP)", "factor": "V11 (Increases Risk)"},
    {"amount": 890.00, "prob": 0.58, "score": 58.0, "tier": "Medium Risk", "is_fraud": False, "decision": "Step-Up Authentication (2FA/OTP)", "factor": "V2 (Increases Risk)"},
    {"amount": 750.00, "prob": 0.65, "score": 65.0, "tier": "Medium Risk", "is_fraud": False, "decision": "Step-Up Authentication (2FA/OTP)", "factor": "scaled_Time (Increases Risk)"},

    # High risk transactions (Flag for Fraud Analyst Review)
    {"amount": 1450.00, "prob": 0.76, "score": 76.0, "tier": "High Risk", "is_fraud": True, "decision": "Flag for Manual Fraud Analyst Review", "factor": "V14 (Increases Risk)"},
    {"amount": 2300.00, "prob": 0.82, "score": 82.0, "tier": "High Risk", "is_fraud": True, "decision": "Flag for Manual Fraud Analyst Review", "factor": "V4 (Increases Risk)"},
    {"amount": 1890.00, "prob": 0.88, "score": 88.0, "tier": "High Risk", "is_fraud": True, "decision": "Flag for Manual Fraud Analyst Review", "factor": "V10 (Increases Risk)"},

    # Critical risk transactions (Immediate Freeze / Decline)
    {"amount": 4890.00, "prob": 0.96, "score": 96.0, "tier": "Critical Risk", "is_fraud": True, "decision": "Decline / Immediate Account Freeze", "factor": "V12 (Increases Risk)"},
    {"amount": 3200.00, "prob": 0.94, "score": 94.0, "tier": "Critical Risk", "is_fraud": True, "decision": "Decline / Immediate Account Freeze", "factor": "V17 (Increases Risk)"},
    {"amount": 9500.00, "prob": 0.99, "score": 99.0, "tier": "Critical Risk", "is_fraud": True, "decision": "Decline / Immediate Account Freeze", "factor": "V14 (Increases Risk)"}
]

def seed_database():
    init_db()
    session = get_session_direct()
    logger.info("Seeding realistic sample transactions into SQLite audit database...")

    now = datetime.utcnow()
    for idx, sample in enumerate(SAMPLE_TRANSACTIONS):
        txn_time = now - timedelta(minutes=random.randint(5, 300))
        entry = TransactionAudit(
            transaction_id=f"TXN-{1000 + idx:04d}",
            timestamp=txn_time,
            amount=sample["amount"],
            fraud_probability=sample["prob"],
            risk_score=sample["score"],
            risk_level=sample["tier"],
            is_fraud=sample["is_fraud"],
            decision=sample["decision"],
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
