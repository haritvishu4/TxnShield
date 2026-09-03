import time
import uuid
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionRequest,
    BatchPredictionResponse,
    HealthResponse,
    FeatureExplanation
)
from src.database.connection import get_db_session
from src.database.models import TransactionAudit
from src.utils.logger import get_logger

logger = get_logger("api_routes")
router = APIRouter()

# Global runtime state populated during app startup
RUNTIME_STATE = {
    "model": None,
    "preprocessor": None,
    "explainer": None,
    "risk_scorer": None,
    "metrics": {},
    "optimal_threshold": 0.5,
    "version": "1.0.0"
}

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check verifying model loading and system availability."""
    model_loaded = RUNTIME_STATE["model"] is not None
    model_family = type(RUNTIME_STATE["model"]).__name__ if model_loaded else "None"
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_family=model_family,
        optimal_threshold=RUNTIME_STATE["optimal_threshold"],
        version=RUNTIME_STATE["version"]
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(payload: TransactionInput, db: Session = Depends(get_db_session)):
    """Real-time transaction risk scoring and fraud classification endpoint."""
    start_time = time.perf_counter()

    if RUNTIME_STATE["model"] is None or RUNTIME_STATE["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="ML Model or Preprocessor is not loaded.")

    txn_id = payload.transaction_id or f"TXN-{uuid.uuid4().hex[:10].upper()}"
    input_dict = payload.model_dump(exclude={"transaction_id"})

    try:
        # 1. Feature Preprocessing
        df_single = pd.DataFrame([input_dict])
        transformed_features = RUNTIME_STATE["preprocessor"].transform_dataframe(df_single)

        # 2. Probability Estimation
        model = RUNTIME_STATE["model"]
        if hasattr(model, "predict_proba"):
            fraud_prob = float(model.predict_proba(transformed_features)[0, 1])
        else:
            raw = model.decision_function(transformed_features)[0]
            fraud_prob = float(1 / (1 + np.exp(-raw)))

        # 3. Risk Intelligence Evaluation
        threshold = RUNTIME_STATE["optimal_threshold"]
        risk_result = RUNTIME_STATE["risk_scorer"].evaluate_transaction(
            fraud_probability=fraud_prob,
            decision_threshold=threshold
        )

        # 4. Explainability (SHAP top factors)
        top_factors = []
        top_factor_str = "Standard legitimate profile"
        if RUNTIME_STATE["explainer"] is not None:
            explanation = RUNTIME_STATE["explainer"].explain_instance(transformed_features, top_k=3)
            for item in explanation.get("top_features", []):
                top_factors.append(FeatureExplanation(**item))
            if explanation.get("risk_elevating_factors"):
                top_risk = explanation["risk_elevating_factors"][0]
                top_factor_str = f"{top_risk['feature']} ({top_risk['impact']})"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 5. Persist to Transaction Audit Log
        audit_entry = TransactionAudit(
            transaction_id=txn_id,
            amount=float(payload.Amount),
            fraud_probability=risk_result["fraud_probability"],
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            is_fraud=risk_result["is_fraud"],
            decision=risk_result["decision"],
            model_version=RUNTIME_STATE["version"],
            latency_ms=latency_ms,
            top_factor=top_factor_str,
            features_json=json.dumps(input_dict)
        )
        db.add(audit_entry)
        db.commit()

        return PredictionResponse(
            transaction_id=txn_id,
            is_fraud=risk_result["is_fraud"],
            prediction=risk_result["prediction"],
            fraud_probability=risk_result["fraud_probability"],
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            decision=risk_result["decision"],
            badge_color=risk_result["badge_color"],
            latency_ms=latency_ms,
            top_risk_drivers=top_factors,
            model_version=RUNTIME_STATE["version"]
        )
    except Exception as e:
        logger.error(f"Inference error for transaction {txn_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {str(e)}")

@router.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(payload: BatchTransactionRequest, db: Session = Depends(get_db_session)):
    """High-throughput batch transaction inference."""
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Transaction list cannot be empty.")

    predictions = []
    fraud_count = 0
    total_score = 0.0

    for txn in payload.transactions:
        res = predict_transaction(txn, db)
        predictions.append(res)
        if res.is_fraud:
            fraud_count += 1
        total_score += res.risk_score

    total = len(predictions)
    return BatchPredictionResponse(
        total_processed=total,
        fraud_detected=fraud_count,
        legitimate_detected=total - fraud_count,
        average_risk_score=round(total_score / total, 2) if total > 0 else 0.0,
        predictions=predictions
    )

@router.get("/history", tags=["Monitoring"])
def get_transaction_history(
    limit: int = Query(default=50, le=500),
    risk_level: Optional[str] = None,
    is_fraud: Optional[bool] = None,
    db: Session = Depends(get_db_session)
):
    """Retrieves logged transactions from SQLite audit database."""
    query = db.query(TransactionAudit)
    if risk_level:
        query = query.filter(TransactionAudit.risk_level == risk_level)
    if is_fraud is not None:
        query = query.filter(TransactionAudit.is_fraud == is_fraud)

    records = query.order_by(TransactionAudit.id.desc()).limit(limit).all()
    return [r.to_dict() for r in records]

@router.get("/metrics", tags=["Monitoring"])
def get_model_metrics():
    """Returns model performance metrics and benchmark comparison results."""
    return RUNTIME_STATE.get("metrics", {})
