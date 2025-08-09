import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Detect duplicate rows in a DataFrame.

    Args:
        df: Input DataFrame.
        subset: List of columns to consider for duplication (None means all columns).
        keep: Which duplicates to mark as True. Default is "first".
        logger: Optional logger for process logging.

    Returns:
        A boolean Series where True indicates duplicates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if subset is not None:
        if not isinstance(subset, list) or not all(isinstance(col, str) for col in subset):
            raise ValueError("subset must be a list of strings representing column names.")
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    if logger:
        logger.info(f"Detecting duplicates using subset: {subset if subset else 'all columns'} and keep='{keep}'.")

    # Use pandas duplicated, which handles NaNs as equal, unless subset columns are of mixed types (edge case)
    try:
        duplicates = df.duplicated(subset=subset, keep=keep)
    except Exception as e:
        if logger:
            logger.error(f"Error detecting duplicates: {e}")
        raise RuntimeError(f"Error detecting duplicates: {e}")

    if logger:
        logger.info(f"Detected {duplicates.sum()} duplicates.")
    return duplicates

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicates from a DataFrame and return a cleaned DataFrame and summary report.

    Args:
        df: Input DataFrame.
        subset: List of columns to consider for duplication (None means all columns).
        keep: Which duplicates to keep: 'first', 'last', or False.
        logger: Optional logger for process logging.

    Returns:
        cleaned_df: DataFrame with duplicates removed.
        report: Dictionary with summary of duplicates found and removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    original_shape = df.shape
    if logger:
        logger.info(f"Starting deduplication. Initial shape: {original_shape}")

    # Detect duplicates
    try:
        dup_mask = detect_duplicates(df, subset=subset, keep=keep, logger=logger)
    except Exception as e:
        if logger:
            logger.error(f"Failed to detect duplicates: {e}")
        raise

    n_duplicates = dup_mask.sum()

    # Remove duplicates
    try:
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    except Exception as e:
        if logger:
            logger.error(f"Failed to remove duplicates: {e}")
        raise RuntimeError(f"Error removing duplicates: {e}")

    cleaned_shape = cleaned_df.shape
    report = {
        "initial_rows": original_shape[0],
        "final_rows": cleaned_shape[0],
        "duplicates_found": int(n_duplicates),
        "duplicates_removed": int(original_shape[0] - cleaned_shape[0]),
        "deduplication_mode": "all columns" if subset is None else f"columns: {subset}",
        "kept": keep,
    }

    if logger:
        logger.info(
            f"Deduplication complete. Removed {report['duplicates_removed']} duplicates. "
            f"Final shape: {cleaned_shape}"
        )
        logger.debug(f"Deduplication report: {report}")

    return cleaned_df, report

# Example logger setup for unit testing/log capture
def get_default_logger(name: str = "deduplication_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Example usage:
# logger = get_default_logger()
# cleaned_df, report = remove_duplicates(df, subset=['col1', 'col2'], keep='first', logger=logger)
