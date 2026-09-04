from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def utc_now_naive():
    """Return naive UTC for compatibility with the existing SQLite column."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class TransactionAudit(Base):
    """Stores every transaction evaluated by the fraud detection system for auditing and monitoring."""
    __tablename__ = "transaction_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=utc_now_naive, nullable=False)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    decision = Column(String(128), nullable=False)
    model_version = Column(String(32), default="1.0.0")
    latency_ms = Column(Float, nullable=True)
    top_factor = Column(String(256), nullable=True)
    features_json = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "amount": self.amount,
            "fraud_probability": self.fraud_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "is_fraud": self.is_fraud,
            "decision": self.decision,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "top_factor": self.top_factor
        }
