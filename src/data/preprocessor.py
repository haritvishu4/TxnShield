from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("data_preprocessor")

class DataPreprocessor:
    """Production preprocessor that prevents data leakage and handles feature scaling."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.target_col = self.config["data"]["target_column"]
        self.test_size = float(self.config["data"]["test_size"])
        self.val_size = float(self.config["data"]["val_size"])
        self.random_state = int(self.config["data"]["random_state"])
        self.amount_col = self.config["data"]["amount_feature"]
        self.time_col = self.config["data"]["time_feature"]
        self.preprocessor_path = Path(self.config["paths"]["preprocessor_path"])
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])

        # Scalers
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.feature_columns = None
        self.is_fitted = False

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Performs stratified Train / Validation / Test split strictly prior to fitting scalers."""
        logger.info("Splitting dataset into Train (70%), Validation (15%), and Test (15%) splits...")
        
        # Remove duplicates if any to prevent exact duplicate leakage across splits
        initial_len = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        dropped = initial_len - len(df_clean)
        if dropped > 0:
            logger.info(f"Removed {dropped:,} duplicate transaction rows before splitting.")

        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]
        self.feature_columns = list(X.columns)

        # 1st split: Train vs Temp (Val + Test)
        temp_ratio = self.val_size + self.test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=temp_ratio,
            stratify=y,
            random_state=self.random_state
        )

        # 2nd split: Val vs Test (50/50 of temp = 15% each)
        val_temp_ratio = self.val_size / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - val_temp_ratio),
            stratify=y_temp,
            random_state=self.random_state
        )

        logger.info(f"Train split: {len(X_train):,} samples (Fraud: {int(y_train.sum())})")
        logger.info(f"Val split:   {len(X_val):,} samples (Fraud: {int(y_val.sum())})")
        logger.info(f"Test split:  {len(X_test):,} samples (Fraud: {int(y_test.sum())})")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits RobustScaler on training set only, then transforms train, val, and test."""
        logger.info("Fitting feature scalers strictly on training data (preventing leakage)...")
        
        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy()
        X_test_proc = X_test.copy()

        # Fit & transform Amount and Time with RobustScaler
        if self.amount_col in X_train_proc.columns:
            X_train_proc[f"scaled_{self.amount_col}"] = self.amount_scaler.fit_transform(X_train_proc[[self.amount_col]])
            X_val_proc[f"scaled_{self.amount_col}"] = self.amount_scaler.transform(X_val_proc[[self.amount_col]])
            X_test_proc[f"scaled_{self.amount_col}"] = self.amount_scaler.transform(X_test_proc[[self.amount_col]])
            # Drop original raw amount to avoid multicollinearity
            X_train_proc = X_train_proc.drop(columns=[self.amount_col])
            X_val_proc = X_val_proc.drop(columns=[self.amount_col])
            X_test_proc = X_test_proc.drop(columns=[self.amount_col])

        if self.time_col in X_train_proc.columns:
            X_train_proc[f"scaled_{self.time_col}"] = self.time_scaler.fit_transform(X_train_proc[[self.time_col]])
            X_val_proc[f"scaled_{self.time_col}"] = self.time_scaler.transform(X_val_proc[[self.time_col]])
            X_test_proc[f"scaled_{self.time_col}"] = self.time_scaler.transform(X_test_proc[[self.time_col]])
            # Drop original raw time
            X_train_proc = X_train_proc.drop(columns=[self.time_col])
            X_val_proc = X_val_proc.drop(columns=[self.time_col])
            X_test_proc = X_test_proc.drop(columns=[self.time_col])

        self.transformed_feature_names = list(X_train_proc.columns)
        self.is_fitted = True

        return X_train_proc.values, X_val_proc.values, X_test_proc.values

    def transform_single(self, input_dict: Dict[str, float]) -> np.ndarray:
        """Transforms a single raw transaction dictionary into model-ready array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before calling transform_single.")

        df = pd.DataFrame([input_dict])
        return self.transform_dataframe(df)

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms an incoming DataFrame of raw features matching training schema."""
        df_proc = df.copy()
        if self.amount_col in df_proc.columns:
            df_proc[f"scaled_{self.amount_col}"] = self.amount_scaler.transform(df_proc[[self.amount_col]])
            df_proc = df_proc.drop(columns=[self.amount_col])
        if self.time_col in df_proc.columns:
            df_proc[f"scaled_{self.time_col}"] = self.time_scaler.transform(df_proc[[self.time_col]])
            df_proc = df_proc.drop(columns=[self.time_col])

        # Ensure correct column order
        df_proc = df_proc[self.transformed_feature_names]
        return df_proc.values

    def save(self, filepath: Path = None):
        """Serializes the preprocessor instance to disk."""
        target = filepath or self.preprocessor_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        logger.info(f"Preprocessor serialized and saved to {target}.")

    @classmethod
    def load(cls, filepath: Path) -> "DataPreprocessor":
        """Loads a serialized preprocessor."""
        logger.info(f"Loading preprocessor from {filepath}...")
        return joblib.load(filepath)
