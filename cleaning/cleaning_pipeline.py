import pandas as pd
import logging
from typing import Callable, Dict, List, Optional, Any, Tuple

class PipelineStepError(Exception):
    """Custom exception raised when a pipeline step fails."""

class DataCleaningPipeline:
    """
    Orchestrates data cleaning steps such as deduplication, missing value imputation,
    anomaly detection, PII masking, and schema normalization on a Pandas DataFrame.

    Features:
        - Configurable steps and execution order
        - Extensible with custom cleaning functions
        - Logging hooks for each step
        - Error handling and rollback support
        - Type hints and detailed docstrings

    Usage:
        pipeline = DataCleaningPipeline(config={
            "steps": ["deduplicate", "impute_missing", "detect_anomalies", "mask_pii", "normalize_schema"]
        })
        cleaned_df = pipeline.run(df)
    """

    # Built-in step registry mapping step names to their implementation
    _builtin_steps: Dict[str, Callable[['DataCleaningPipeline', pd.DataFrame], pd.DataFrame]] = {}

    def __init__(
        self,
        config: Dict[str, Any],
        custom_steps: Optional[Dict[str, Callable[['DataCleaningPipeline', pd.DataFrame], pd.DataFrame]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataCleaningPipeline.

        Args:
            config: Dictionary with pipeline configuration. Must contain "steps" as a list of step names.
            custom_steps: Optional mapping of custom step names to functions.
            logger: Optional logger instance to use for logging hooks.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.custom_steps = custom_steps or {}
        self._register_builtin_steps()
        self._validate_config()
        self._step_history: List[Tuple[str, pd.DataFrame]] = []

    def _register_builtin_steps(self):
        """Registers the built-in steps in the pipeline."""
        self._builtin_steps = {
            "deduplicate": self._deduplicate,
            "impute_missing": self._impute_missing,
            "detect_anomalies": self._detect_anomalies,
            "mask_pii": self._mask_pii,
            "normalize_schema": self._normalize_schema,
        }

    def _validate_config(self):
        if "steps" not in self.config or not isinstance(self.config["steps"], list):
            raise ValueError("Config must contain a 'steps' key with a list of step names.")

    def add_step(self, name: str, func: Callable[['DataCleaningPipeline', pd.DataFrame], pd.DataFrame]) -> None:
        """
        Add a custom cleaning step to the pipeline.

        Args:
            name: Name of the step.
            func: Function implementing the cleaning logic.
        """
        self.custom_steps[name] = func

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the configured cleaning steps on the given DataFrame.

        Args:
            df: Input Pandas DataFrame.

        Returns:
            Cleaned Pandas DataFrame.

        Raises:
            PipelineStepError: If any step fails.
        """
        current_df = df.copy(deep=True)
        self._step_history = [("start", current_df.copy(deep=True))]

        for step_name in self.config["steps"]:
            step_func = self._get_step_func(step_name)
            self._log_hook(f"Starting step: {step_name}")

            try:
                before_step = current_df.copy(deep=True)
                current_df = step_func(current_df)
                self._log_hook(f"Completed step: {step_name}")
                self._step_history.append((step_name, current_df.copy(deep=True)))
            except Exception as e:
                self._log_hook(f"Error in step '{step_name}': {e}", level="error")
                self.rollback()
                raise PipelineStepError(f"Step '{step_name}' failed: {e}") from e

        return current_df

    def rollback(self) -> pd.DataFrame:
        """
        Rollback to the previous DataFrame state before the most recent step.

        Returns:
            The DataFrame from the last successful step.
        """
        if len(self._step_history) < 2:
            self._log_hook("No step to rollback to.", level="warning")
            return self._step_history[0][1].copy(deep=True)
        # Remove the failed step
        last_good_step = self._step_history[-2]
        self._log_hook(f"Rolled back to step: {last_good_step[0]}")
        return last_good_step[1].copy(deep=True)

    def _get_step_func(self, name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Fetches the function for a step (builtin or custom)."""
        if name in self.custom_steps:
            return lambda df: self.custom_steps[name](self, df)
        elif name in self._builtin_steps:
            return lambda df: self._builtin_steps[name](df)
        else:
            raise ValueError(f"Step '{name}' is not a recognized built-in or custom step.")

    def _log_hook(self, message: str, level: str = "info"):
        """Logging hook for steps."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.debug(message)

    # --- Built-in step implementations below ---

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: Input DataFrame.

        Returns:
            Deduplicated DataFrame.
        """
        return df.drop_duplicates(ignore_index=True)

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using simple strategies (mean for numeric, mode for categorical).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values imputed.
        """
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING")
        return df

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag anomalies in numeric columns using IQR.
        Adds a boolean column for each numeric field: <col>_anomaly

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with anomaly flags.
        """
        for col in df.select_dtypes(include="number").columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            df[f"{col}_anomaly"] = mask
        return df

    def _mask_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mask columns likely to contain PII (e.g., columns containing 'email', 'phone', or 'ssn' in their name).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with PII columns masked.
        """
        import re

        pii_patterns = {
            "email": r"[^@]+@[^@]+\.[^@]+",
            "phone": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        }
        pii_keywords = ["email", "phone", "ssn", "name", "dob", "address"]

        def mask_value(val: Any) -> str:
            if pd.isnull(val):
                return val
            return "***MASKED***"

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in pii_keywords):
                df[col] = df[col].apply(mask_value)
            elif df[col].dtype == "object":
                # Try regex masking for known patterns
                for pattern in pii_patterns.values():
                    df[col] = df[col].replace(pattern, "***MASKED***", regex=True)
        return df

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to snake_case and ensure consistent types.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with normalized schema.
        """
        import re

        def to_snake_case(name: str) -> str:
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
            name = name.replace(" ", "_").replace("-", "_")
            return name.lower()

        df = df.rename(columns={col: to_snake_case(col) for col in df.columns})
        # Optionally, ensure string columns are object dtype
        for col in df.select_dtypes(include=["string"]).columns:
            df[col] = df[col].astype("object")
        return df

    # --- End of built-in steps ---

    def get_step_history(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Get the history of DataFrames after each step.

        Returns:
            A list of (step_name, DataFrame) tuples.
        """
        return self._step_history

    def get_available_steps(self) -> List[str]:
        """
        List all available built-in and custom step names.

        Returns:
            A list of step names.
        """
        return list(self._builtin_steps.keys()) + list(self.custom_steps.keys())
