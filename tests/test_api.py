import pytest
from fastapi.testclient import TestClient
from api.app import app
from api.routes import RUNTIME_STATE
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.risk_engine.scorer import RiskScorer
import numpy as np
import pandas as pd

@pytest.fixture(scope="module", autouse=True)
def setup_runtime_state():
    """Sets up lightweight fitted mock model & preprocessor for API testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(np.random.normal(0, 1, (n, 30)), columns=[f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])
    df["Class"] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    
    preprocessor = DataPreprocessor()
    X_tr_df, X_v_df, X_ts_df, y_tr, y_v, y_ts = preprocessor.split_data(df)
    X_tr, _, _ = preprocessor.fit_transform(X_tr_df, X_v_df, X_ts_df)

    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X_tr, y_tr.values)

    RUNTIME_STATE["model"] = model
    RUNTIME_STATE["preprocessor"] = preprocessor
    RUNTIME_STATE["risk_scorer"] = RiskScorer()
    RUNTIME_STATE["optimal_threshold"] = 0.5
    RUNTIME_STATE["version"] = "1.0.0"

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_endpoint_valid():
    payload = {
        "transaction_id": "TXN-TEST-123",
        "Amount": 150.0,
        "Time": 450.0,
        "V1": 0.5,
        "V4": 1.2,
        "V14": -2.1
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["transaction_id"] == "TXN-TEST-123"
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "decision" in data
    assert 0.0 <= data["risk_score"] <= 100.0

def test_predict_endpoint_invalid_amount():
    payload = {
        "transaction_id": "TXN-INVALID",
        "Amount": -50.0 # Invalid negative amount
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422 # Pydantic validation error

def test_batch_predict_endpoint():
    payload = {
        "transactions": [
            {"transaction_id": "TXN-B1", "Amount": 25.0},
            {"transaction_id": "TXN-B2", "Amount": 890.0, "V14": -3.5}
        ]
    }
    resp = client.post("/batch-predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2

def test_history_endpoint():
    resp = client.get("/history?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
