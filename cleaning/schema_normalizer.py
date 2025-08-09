import pandas as pd
import numpy as np
import re
import logging
from typing import Any, Dict, Optional, List, Tuple, Union

def to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"__+", "_", name)
    return name.lower()

def normalize_column_names(
    columns: List[str],
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """Normalize column names to snake_case and deduplicate if necessary."""
    normalized = []
    seen = {}
    for col in columns:
        orig = col
        col = to_snake_case(str(col))
        col = re.sub(r"[^a-z0-9_]", "_", col)
        # Deduplicate
        base = col
        i = 1
        while col in seen:
            col = f"{base}_{i}"
            i += 1
            if logger:
                logger.warning(f"Duplicate column name '{base}' detected. Renamed to '{col}'.")
        seen[col] = True
        normalized.append(col)
        if logger and orig != col:
            logger.info(f"Renamed column '{orig}' to '{col}'.")
    return normalized

def standardize_types(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    categorical_threshold: int = 20,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Standardize column data types: parse dates, convert numerics, categorize categoricals.
    """
    df_clean = df.copy()
    date_columns = date_columns or []

    for col in df_clean.columns:
        # Parse dates
        if col in date_columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                if logger:
                    logger.info(f"Parsed column '{col}' as datetime.")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not parse column '{col}' as datetime: {e}")
            continue

        # Convert numerics
        if pd.api.types.is_object_dtype(df_clean[col]) or pd.api.types.is_string_dtype(df_clean[col]):
            try:
                converted = pd.to_numeric(df_clean[col], errors='coerce')
                n_converted = converted.notna().sum()
                if n_converted > 0.8 * len(df_clean[col]):
                    df_clean[col] = converted
                    if logger:
                        logger.info(f"Converted column '{col}' to numeric.")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not convert column '{col}' to numeric: {e}")

        # Categorize categoricals
        nunique = df_clean[col].nunique(dropna=True)
        if nunique <= categorical_threshold and not pd.api.types.is_categorical_dtype(df_clean[col]):
            try:
                df_clean[col] = df_clean[col].astype("category")
                if logger:
                    logger.info(f"Categorized column '{col}' as 'category' ({nunique} unique values).")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not categorize column '{col}': {e}")

    return df_clean

def validate_schema(
    df: pd.DataFrame,
    template: Optional[Dict[str, Union[type, str, Dict[str, Any]]]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema against a template or data dictionary.

    Args:
        df: DataFrame to validate.
        template: Dict with expected column names and types, e.g.:
                  {"date": "datetime64[ns]", "age": "int", "name": "category"}
        logger: Logger for logging.

    Returns:
        (is_valid, issues): Tuple of validation result and list of issues.
    """
    issues = []
    if template is None:
        if logger:
            logger.info("No template provided for schema validation.")
        return True, issues

    for col, expected in template.items():
        if col not in df.columns:
            issues.append(f"Missing column: '{col}'")
            if logger:
                logger.warning(f"Schema validation: Missing column '{col}'")
            continue
        actual_dtype = str(df[col].dtype)
        expected_dtype = expected if isinstance(expected, str) else str(expected)
        if expected_dtype not in actual_dtype:
            issues.append(f"Column '{col}' type mismatch: expected '{expected_dtype}', got '{actual_dtype}'")
            if logger:
                logger.warning(f"Schema validation: Column '{col}' type mismatch: expected '{expected_dtype}', got '{actual_dtype}'")
    is_valid = len(issues) == 0
    if logger:
        if is_valid:
            logger.info("Schema validation passed.")
        else:
            logger.warning(f"Schema validation failed with issues: {issues}")
    return is_valid, issues

def clean_schema(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    categorical_threshold: int = 20,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Clean DataFrame schema: normalize column names and standardize types.

    Args:
        df: Input DataFrame.
        date_columns: List of columns to treat as dates.
        categorical_threshold: Max unique values for categoricals.
        logger: Logger for logging.

    Returns:
        Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df_clean = df.copy()
    orig_columns = df_clean.columns.tolist()
    new_columns = normalize_column_names(orig_columns, logger=logger)
    df_clean.columns = new_columns

    df_clean = standardize_types(df_clean, date_columns=date_columns, 
                                 categorical_threshold=categorical_threshold, logger=logger)
    return df_clean

# Example usage:
# logger = logging.getLogger("schema_normalizer")
# df = clean_schema(df, date_columns=["date_col"], logger=logger)
# is_valid, issues = validate_schema(df, template={"date_col": "datetime64", "amount": "float"}, logger=logger)
