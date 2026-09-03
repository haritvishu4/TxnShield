import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import DataPreprocessor

@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 200
    time = np.random.uniform(0, 1000, n)
    v_features = np.random.normal(0, 1, (n, 28))
    amount = np.random.exponential(50, n)
    # 5% fraud rate
    target = np.random.binomial(1, 0.05, n)

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time, v_features, amount, target])
    df = pd.DataFrame(data, columns=cols)
    df["Class"] = df["Class"].astype(int)
    return df

def test_split_data_shapes(sample_dataset):
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(sample_dataset)

    total_samples = len(sample_dataset)
    assert len(X_train) + len(X_val) + len(X_test) == total_samples
    assert len(X_train) > len(X_val)
    assert len(X_val) == len(X_test)

def test_fit_transform_leakage_free(sample_dataset):
    preprocessor = DataPreprocessor()
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = preprocessor.split_data(sample_dataset)
    X_train, X_val, X_test = preprocessor.fit_transform(X_train_df, X_val_df, X_test_df)

    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[1] == 30  # 28 V features + scaled_Amount + scaled_Time
    assert preprocessor.is_fitted is True

def test_transform_single_transaction(sample_dataset):
    preprocessor = DataPreprocessor()
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = preprocessor.split_data(sample_dataset)
    preprocessor.fit_transform(X_train_df, X_val_df, X_test_df)

    raw_sample = {col: 0.0 for col in preprocessor.feature_columns}
    raw_sample["Amount"] = 120.0
    raw_sample["Time"] = 500.0

    transformed = preprocessor.transform_single(raw_sample)
    assert transformed.shape == (1, 30)
