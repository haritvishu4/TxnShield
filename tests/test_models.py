import pytest
import numpy as np
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.threshold_optimizer import ThresholdOptimizer

@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    n = 300
    X = np.random.normal(0, 1, (n, 30))
    # Imbalanced: 10% positives
    y = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    return X, y

def test_logistic_regression_training(dummy_train_data):
    X, y = dummy_train_data
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X, y)
    
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

def test_evaluator_metrics(dummy_train_data):
    X, y = dummy_train_data
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X, y)
    
    metrics = ModelEvaluator.evaluate(model, X, y, threshold=0.5)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics

def test_threshold_optimizer(dummy_train_data):
    X, y = dummy_train_data
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X, y)

    res = ThresholdOptimizer.find_optimal_threshold(model, X, y)
    assert 0.0 < res["optimal_threshold"] < 1.0
    assert "best_f_score" in res
