from typing import Dict, Any, Tuple
import joblib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("model_trainer")

class ModelTrainer:
    """Trains baseline and ensemble machine learning models with imbalanced data handling."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.cfg_lr = self.config["training"]["logistic_regression"]
        self.cfg_rf = self.config["training"]["random_forest"]
        self.cfg_xgb = self.config["training"]["xgboost"]
        self.models_dir = Path(self.config["paths"]["models_dir"])

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Trains Logistic Regression baseline with balanced class weights."""
        logger.info("Training Logistic Regression baseline...")
        model = LogisticRegression(
            C=float(self.cfg_lr["C"]),
            max_iter=int(self.cfg_lr["max_iter"]),
            class_weight=self.cfg_lr["class_weight"],
            solver=self.cfg_lr["solver"],
            random_state=42
        )
        model.fit(X_train, y_train)
        logger.info("Logistic Regression training completed.")
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Trains Random Forest Classifier with balanced class weights."""
        logger.info("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=int(self.cfg_rf["n_estimators"]),
            max_depth=int(self.cfg_rf["max_depth"]),
            class_weight=self.cfg_rf["class_weight"],
            random_state=int(self.cfg_rf["random_state"]),
            n_jobs=int(self.cfg_rf["n_jobs"])
        )
        model.fit(X_train, y_train)
        logger.info("Random Forest training completed.")
        return model

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """Trains XGBoost Classifier with scale_pos_weight for severe imbalance."""
        logger.info("Training XGBoost Classifier...")
        # Calculate dynamic scale_pos_weight if needed: N_neg / N_pos
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_weight = float(n_neg / max(1, n_pos))
        logger.info(f"XGBoost dynamic scale_pos_weight: {scale_weight:.2f}")

        model = XGBClassifier(
            n_estimators=int(self.cfg_xgb["n_estimators"]),
            max_depth=int(self.cfg_xgb["max_depth"]),
            learning_rate=float(self.cfg_xgb["learning_rate"]),
            scale_pos_weight=scale_weight,
            random_state=int(self.cfg_xgb["random_state"]),
            eval_metric=self.cfg_xgb["eval_metric"],
            n_jobs=int(self.cfg_xgb["n_jobs"])
        )
        model.fit(X_train, y_train)
        logger.info("XGBoost training completed.")
        return model

    def save_model(self, model: Any, filepath: Path):
        """Serializes trained model artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
