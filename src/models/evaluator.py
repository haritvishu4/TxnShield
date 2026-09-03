from typing import Dict, Any, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.utils.logger import get_logger

logger = get_logger("model_evaluator")

class ModelEvaluator:
    """Computes comprehensive evaluation metrics specifically suited for highly imbalanced fraud classification."""

    @staticmethod
    def evaluate(model: Any, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculates classification metrics, PR-AUC, ROC-AUC, and confusion matrix."""
        # Probabilities for class 1 (Fraud)
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X)
            y_probs = 1 / (1 + np.exp(-raw_scores))
        else:
            y_probs = model.predict(X)

        # Apply threshold
        y_pred = (y_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        
        roc_auc = float(roc_auc_score(y, y_probs)) if len(np.unique(y)) > 1 else 0.0
        pr_auc = float(average_precision_score(y, y_probs)) if len(np.unique(y)) > 1 else 0.0

        metrics = {
            "threshold": float(threshold),
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_evaluated": int(len(y))
        }

        logger.info(
            f"Metrics @ threshold {threshold:.2f} -> "
            f"PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | "
            f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
        )
        return metrics

    @staticmethod
    def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Converts model evaluation dictionaries into a benchmark comparison DataFrame."""
        rows = []
        for model_name, metrics in results.items():
            row = {"Model": model_name}
            row.update(metrics)
            rows.append(row)
        df_comp = pd.DataFrame(rows)
        return df_comp

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], filepath: Path):
        """Saves evaluation metrics summary to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {filepath}")
