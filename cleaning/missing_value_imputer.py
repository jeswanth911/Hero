import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Union
from sklearn.impute import KNNImputer

class MissingValueImputer:
    """
    Class for handling missing value imputation in Pandas DataFrames
    with various strategies, including ML-based imputation.

    Features:
    - Multiple imputation methods: mean, median, mode, forward-fill, backward-fill
    - Advanced ML-based imputation using scikit-learn's KNNImputer
    - Auto-select best imputation strategy based on data type and missingness per column
    - Detailed logging of steps and results
    - Robust error handling and input validation
    """

    SUPPORTED_METHODS = ["mean", "median", "mode", "ffill", "bfill", "knn", "auto"]

    def __init__(
        self,
        strategy: str = "auto",
        knn_params: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            strategy: Imputation strategy for all columns ('mean', 'median', 'mode', 'ffill', 'bfill', 'knn', 'auto').
                      'auto' will infer best strategy per column.
            knn_params: Parameters for KNNImputer. Used if strategy is 'knn' or selected by 'auto'.
            logger: Optional logger.
        """
        if strategy not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported strategy: {strategy}. Supported: {self.SUPPORTED_METHODS}")
        self.strategy = strategy
        self.knn_params = knn_params or {"n_neighbors": 3, "weights": "uniform"}
        self.logger = logger or logging.getLogger(__name__)
        self.imputation_plan_: Optional[Dict[str, str]] = None
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer":
        """
        Analyze the DataFrame and determine imputation strategy per column.

        Args:
            df: Input DataFrame.
        Returns:
            self
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        self.imputation_plan_ = {}
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            dtype = df[col].dtype
            if n_missing == 0:
                self.imputation_plan_[col] = "none"
                continue
            if self.strategy != "auto":
                self.imputation_plan_[col] = self.strategy
            else:
                if pd.api.types.is_numeric_dtype(dtype):
                    if n_missing / len(df) > 0.4:
                        self.imputation_plan_[col] = "median"
                    elif n_missing / len(df) > 0.1:
                        self.imputation_plan_[col] = "knn"
                    else:
                        self.imputation_plan_[col] = "mean"
                elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    if n_missing / len(df) > 0.2:
                        self.imputation_plan_[col] = "mode"
                    else:
                        self.imputation_plan_[col] = "ffill"
                else:
                    self.imputation_plan_[col] = "mode"
        self.fitted_ = True
        self.logger.info(f"Imputation plan: {self.imputation_plan_}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation according to fit plan.

        Args:
            df: Input DataFrame.
        Returns:
            Imputed DataFrame.
        """
        if not self.fitted_ or self.imputation_plan_ is None:
            raise RuntimeError("MissingValueImputer is not fitted. Call fit() first.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df_out = df.copy()
        changes: Dict[str, int] = {}
        knn_cols = [col for col, strat in self.imputation_plan_.items() if strat == "knn"]

        # ML-based imputation for KNN columns
        if knn_cols:
            knn_df = df_out[knn_cols]
            try:
                imputer = KNNImputer(**self.knn_params)
                imputed_knn = imputer.fit_transform(knn_df)
                for idx, col in enumerate(knn_cols):
                    n_filled = knn_df[col].isnull().sum()
                    df_out[col] = imputed_knn[:, idx]
                    changes[col] = n_filled
                    self.logger.info(f"KNN imputed {n_filled} missing values in '{col}'")
            except Exception as e:
                self.logger.error(f"KNN imputation failed: {e}")
                raise

        # Other columns
        for col, strat in self.imputation_plan_.items():
            if strat in ("none", "knn"):
                continue
            n_missing = df_out[col].isnull().sum()
            if n_missing == 0:
                continue
            try:
                if strat == "mean":
                    value = df_out[col].mean()
                    df_out[col] = df_out[col].fillna(value)
                elif strat == "median":
                    value = df_out[col].median()
                    df_out[col] = df_out[col].fillna(value)
                elif strat == "mode":
                    mode_vals = df_out[col].mode()
                    value = mode_vals.iloc[0] if not mode_vals.empty else None
                    df_out[col] = df_out[col].fillna(value)
                elif strat == "ffill":
                    df_out[col] = df_out[col].fillna(method="ffill")
                elif strat == "bfill":
                    df_out[col] = df_out[col].fillna(method="bfill")
                else:
                    raise ValueError(f"Unknown imputation strategy: {strat}")
                changes[col] = n_missing
                self.logger.info(f"Imputed {n_missing} missing values in '{col}' using '{strat}'")
            except Exception as e:
                self.logger.error(f"Imputation failed on column '{col}' with strategy '{strat}': {e}")
                raise

        self.logger.info(f"Imputation complete. Summary: {changes}")
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the DataFrame.

        Args:
            df: Input DataFrame.
        Returns:
            Imputed DataFrame.
        """
        return self.fit(df).transform(df)
