from typing import List, Optional
from pydantic import BaseModel, Field

class TransactionInput(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Unique transaction identifier")
    Time: float = Field(default=100.0, description="Seconds elapsed since initial reference transaction")
    Amount: float = Field(default=50.0, ge=0.0, description="Transaction monetary amount")
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN-DEMO-001",
                "Time": 406.0,
                "Amount": 149.62,
                "V1": -2.3122,
                "V2": 1.9519,
                "V3": -1.6098,
                "V4": 3.9979,
                "V5": -0.5221,
                "V6": -1.4265,
                "V7": -2.5373,
                "V8": 1.3916,
                "V9": -2.7700,
                "V10": -2.7722,
                "V11": 3.2020,
                "V12": -2.8999,
                "V13": -0.5952,
                "V14": -4.2892,
                "V15": 0.3897,
                "V16": -1.1407,
                "V17": -2.8300,
                "V18": -0.0168,
                "V19": 0.4169,
                "V20": 0.1269,
                "V21": 0.5172,
                "V22": -0.0350,
                "V23": -0.4652,
                "V24": 0.3201,
                "V25": 0.0445,
                "V26": 0.1778,
                "V27": 0.2611,
                "V28": -0.1432
            }
        }
    }

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    impact: str

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    prediction: str
    fraud_probability: float
    risk_score: float
    risk_level: str
    decision: str
    badge_color: str
    latency_ms: float
    top_risk_drivers: List[FeatureExplanation]
    model_version: str

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionInput]

class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    legitimate_detected: int
    average_risk_score: float
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_family: str
    optimal_threshold: float
    version: str
