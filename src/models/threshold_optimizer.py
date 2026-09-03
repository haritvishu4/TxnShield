from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, fbeta_score

from src.utils.logger import get_logger

logger = get_logger("threshold_optimizer")

class ThresholdOptimizer:
    """Finds the optimal decision threshold on validation data to balance fraud detection recall vs false alarms."""

    @staticmethod
    def find_optimal_threshold(
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        beta: float = 1.0,
        min_precision: float = 0.5
    ) -> Dict[str, Any]:
        """Searches thresholds to maximize F-beta score (beta=1 is standard F1, beta=2 weighs recall higher than precision)."""
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_val)[:, 1]
        else:
            raw = model.decision_function(X_val)
            y_probs = 1 / (1 + np.exp(-raw))

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)

        best_threshold = 0.5
        best_score = -1.0
        best_p = 0.0
        best_r = 0.0

        # Evaluate candidate thresholds
        candidate_thresholds = np.linspace(0.01, 0.99, 100)
        curve_data = []

        for thresh in candidate_thresholds:
            preds = (y_probs >= thresh).astype(int)
            score = fbeta_score(y_val, preds, beta=beta, zero_division=0)
            p = float(np.sum((preds == 1) & (y_val == 1)) / max(1, np.sum(preds == 1)))
            r = float(np.sum((preds == 1) & (y_val == 1)) / max(1, np.sum(y_val == 1)))

            curve_data.append({
                "threshold": round(float(thresh), 3),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f_beta": round(float(score), 4)
            })

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_p = p
                best_r = r

        logger.info(
            f"Optimal threshold found: τ* = {best_threshold:.3f} "
            f"(F{beta:.0f}: {best_score:.4f}, Precision: {best_p:.4f}, Recall: {best_r:.4f})"
        )

        return {
            "optimal_threshold": round(float(best_threshold), 4),
            "best_f_score": round(float(best_score), 4),
            "precision_at_optimal": round(float(best_p), 4),
            "recall_at_optimal": round(float(best_r), 4),
            "beta": beta,
            "curve_data": curve_data
        }
