import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.data.ingestion import DataIngestion
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.threshold_optimizer import ThresholdOptimizer
from src.models.explainability import FraudExplainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

def main():
    total_start = time.perf_counter()
    logger.info("=================================================================")
    logger.info("  STARTING END-TO-END FRAUD DETECTION ML TRAINING PIPELINE      ")
    logger.info("=================================================================")

    cfg = load_config()

    # Step 1: Ingestion
    logger.info("\n--- [Stage 1/6] Data Ingestion & Validation ---")
    ingestion = DataIngestion(cfg)
    raw_path = ingestion.download_dataset()
    df_raw = ingestion.load_data()
    integrity_summary = ingestion.validate_dataset_integrity(df_raw)

    # Step 2: Leakage-Free Preprocessing
    logger.info("\n--- [Stage 2/6] Data Splitting & Feature Transformation ---")
    preprocessor = DataPreprocessor(cfg)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = preprocessor.split_data(df_raw)
    X_train, X_val, X_test = preprocessor.fit_transform(X_train_df, X_val_df, X_test_df)
    preprocessor.save()

    # Step 3: Model Training
    logger.info("\n--- [Stage 3/6] Multi-Model Training & Benchmark ---")
    trainer = ModelTrainer(cfg)

    # 3.1 Baseline: Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train, y_train.values)

    # 3.2 Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train.values)

    # 3.3 XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train.values)

    models = {
        "Logistic Regression (Baseline)": lr_model,
        "Random Forest": rf_model,
        "XGBoost Classifier": xgb_model
    }

    # Step 4: Model Evaluation on Validation Set (Standard Threshold = 0.5)
    logger.info("\n--- [Stage 4/6] Validation Evaluation (Standard Threshold 0.5) ---")
    val_results = {}
    for name, model in models.items():
        logger.info(f"\nEvaluating {name} on Validation Set...")
        metrics = ModelEvaluator.evaluate(model, X_val, y_val.values, threshold=0.5)
        val_results[name] = metrics

    df_comp_val = ModelEvaluator.compare_models(val_results)
    logger.info("\nValidation Comparison Table (Threshold=0.5):")
    print(df_comp_val[["Model", "pr_auc", "roc_auc", "precision", "recall", "f1_score"]].to_string(index=False))

    # Step 5: Select Best Model & Optimize Decision Threshold on Validation
    logger.info("\n--- [Stage 5/6] Best Model Selection & Threshold Optimization ---")
    # Rank by PR-AUC (Average Precision is the standard gold metric for extreme class imbalance)
    best_model_name = max(val_results.keys(), key=lambda k: val_results[k]["pr_auc"])
    best_model = models[best_model_name]
    logger.info(f"Selected Best Model: '{best_model_name}' (Highest Validation PR-AUC: {val_results[best_model_name]['pr_auc']:.4f})")

    # Find optimal threshold on validation set
    threshold_info = ThresholdOptimizer.find_optimal_threshold(
        model=best_model,
        X_val=X_val,
        y_val=y_val.values,
        beta=1.0 # Optimize F1 score
    )
    optimal_thresh = threshold_info["optimal_threshold"]
    logger.info(f"Dynamically Selected Optimal Decision Threshold: τ* = {optimal_thresh}")

    # Step 6: Final Unbiased Test Set Evaluation
    logger.info("\n--- [Stage 6/6] Final Test Evaluation with Optimized Threshold ---")
    test_metrics_default = ModelEvaluator.evaluate(best_model, X_test, y_test.values, threshold=0.5)
    test_metrics_optimal = ModelEvaluator.evaluate(best_model, X_test, y_test.values, threshold=optimal_thresh)

    logger.info(f"Test Set Default Threshold (0.50): F1={test_metrics_default['f1_score']:.4f}, Recall={test_metrics_default['recall']:.4f}, Precision={test_metrics_default['precision']:.4f}")
    logger.info(f"Test Set Optimal Threshold ({optimal_thresh:.2f}): F1={test_metrics_optimal['f1_score']:.4f}, Recall={test_metrics_optimal['recall']:.4f}, Precision={test_metrics_optimal['precision']:.4f}")

    # Step 7: Serialize Best Model, Preprocessor & Metrics Artifacts
    best_model_path = Path(cfg["paths"]["best_model_path"])
    metrics_path = Path(cfg["paths"]["metrics_path"])

    trainer.save_model(best_model, best_model_path)

    # Verify explainability module
    logger.info("\nVerifying SHAP explainer initialization...")
    explainer = FraudExplainer(best_model, feature_names=preprocessor.transformed_feature_names)
    sample_explanation = explainer.explain_instance(X_test[0], top_k=3)
    logger.info(f"Sample explanation generated: {len(sample_explanation['top_features'])} features extracted.")

    # Save comprehensive metrics summary
    summary = {
        "best_model_name": best_model_name,
        "optimal_threshold": optimal_thresh,
        "validation_benchmark": val_results,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_optimal_threshold": test_metrics_optimal,
        "data_summary": integrity_summary,
        "threshold_curve_sample": threshold_info["curve_data"][::10]
    }
    ModelEvaluator.save_metrics(summary, metrics_path)

    elapsed = time.perf_counter() - total_start
    logger.info(f"\nPipeline successfully completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
