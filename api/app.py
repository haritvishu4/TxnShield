from contextlib import asynccontextmanager
from pathlib import Path
import json
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, RUNTIME_STATE
from src.database.connection import init_db
from src.risk_engine.scorer import RiskScorer
from src.models.explainability import FraudExplainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("api_app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models, database, and background services at startup."""
    logger.info("Initializing Fraud Detection API Service...")
    cfg = load_config()

    # 1. Initialize SQLite Database
    init_db(cfg["paths"]["database_url"])

    # 2. Load Preprocessor and Trained Model
    model_path = Path(cfg["paths"]["best_model_path"])
    preprocessor_path = Path(cfg["paths"]["preprocessor_path"])
    metrics_path = Path(cfg["paths"]["metrics_path"])

    if model_path.exists() and preprocessor_path.exists():
        logger.info(f"Loading best model from {model_path}...")
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        RUNTIME_STATE["model"] = model
        RUNTIME_STATE["preprocessor"] = preprocessor
        RUNTIME_STATE["risk_scorer"] = RiskScorer(cfg)

        # Load metrics & threshold
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
                RUNTIME_STATE["metrics"] = metrics_data
                RUNTIME_STATE["optimal_threshold"] = metrics_data.get("optimal_threshold", 0.5)
        else:
            RUNTIME_STATE["optimal_threshold"] = 0.5

        # Initialize Explainability
        logger.info("Initializing SHAP Explainer...")
        RUNTIME_STATE["explainer"] = FraudExplainer(
            model=model,
            feature_names=preprocessor.transformed_feature_names
        )
        logger.info("System fully loaded and operational.")
    else:
        logger.warning(
            f"Model or Preprocessor artifact not found at {model_path}. "
            "Please execute `python scripts/run_pipeline.py` to train and serialize the model."
        )

    yield
    logger.info("Shutting down Fraud Detection API Service.")

def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    cfg = load_config()
    app = FastAPI(
        title=cfg["api"]["title"],
        description=cfg["api"]["description"],
        version=cfg["api"]["version"],
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
