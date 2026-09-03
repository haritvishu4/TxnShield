from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
from src.utils.logger import get_logger

logger = get_logger("explainability")

class FraudExplainer:
    """Provides SHAP-based local (per-transaction) and global feature explanations."""

    def __init__(self, model: Any, feature_names: List[str], background_sample: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_sample = background_sample
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer depending on model family."""
        try:
            # Tree-based models (RandomForest, XGBoost)
            if hasattr(self.model, "estimators_") or "XGB" in type(self.model).__name__:
                logger.info("Initializing TreeExplainer for tree/ensemble model...")
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "coef_"):
                # Linear models
                logger.info("Initializing LinearExplainer for linear model...")
                if self.background_sample is not None:
                    self.explainer = shap.LinearExplainer(self.model, self.background_sample)
                else:
                    self.explainer = shap.Explainer(self.model)
            else:
                sample = self.background_sample[:50] if self.background_sample is not None else np.zeros((1, len(self.feature_names)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, sample)
        except Exception as e:
            logger.warning(f"Error initializing primary SHAP explainer: {e}. Falling back to default Explainer.")
            self.explainer = shap.Explainer(self.model)

    def explain_instance(self, instance_array: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Calculates SHAP values for a single transaction and extracts top risk drivers."""
        if instance_array.ndim == 1:
            instance_array = instance_array.reshape(1, -1)

        try:
            shap_values = self.explainer(instance_array)
            # Handle different shap output shapes (binary vs multiclass)
            if hasattr(shap_values, "values"):
                vals = shap_values.values
                # If output is 3D (samples, features, classes), pick class 1 (Fraud)
                if vals.ndim == 3:
                    vals = vals[0, :, 1]
                elif vals.ndim == 2:
                    vals = vals[0, :]
                else:
                    vals = vals[0]
                base_value = float(shap_values.base_values[0, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values[0])
            else:
                raw = self.explainer.shap_values(instance_array)
                if isinstance(raw, list) and len(raw) > 1:
                    vals = raw[1][0]
                else:
                    vals = raw[0]
                base_value = 0.0

            # Rank features by absolute SHAP impact
            contributions = []
            for name, val, feat_val in zip(self.feature_names, vals, instance_array[0]):
                contributions.append({
                    "feature": name,
                    "shap_value": round(float(val), 4),
                    "feature_value": round(float(feat_val), 4),
                    "impact": "Increases Risk" if val > 0 else "Decreases Risk"
                })

            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            top_risk_drivers = [c for c in contributions if c["shap_value"] > 0][:top_k]
            top_safety_drivers = [c for c in contributions if c["shap_value"] < 0][:top_k]

            return {
                "base_value": round(base_value, 4),
                "top_features": contributions[:top_k],
                "risk_elevating_factors": top_risk_drivers,
                "risk_mitigating_factors": top_safety_drivers,
                "all_contributions": contributions
            }
        except Exception as e:
            logger.warning(f"SHAP explanation computation encountered exception: {e}. Returning feature importance fallback.")
            return self._fallback_explanation(instance_array, top_k)

    def _fallback_explanation(self, instance_array: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Provides heuristic feature importance if SHAP fails on edge cases."""
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None and hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        if importances is None:
            importances = np.ones(len(self.feature_names))

        norm_imp = importances / np.sum(importances)
        ranked = sorted(zip(self.feature_names, norm_imp, instance_array[0]), key=lambda x: x[1], reverse=True)
        top_features = [{
            "feature": f,
            "shap_value": round(float(imp), 4),
            "feature_value": round(float(val), 4),
            "impact": "High Weight"
        } for f, imp, val in ranked[:top_k]]

        return {
            "base_value": 0.0,
            "top_features": top_features,
            "risk_elevating_factors": top_features[:2],
            "risk_mitigating_factors": [],
            "all_contributions": top_features
        }
