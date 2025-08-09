import pandas as pd
import re
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

class ValidationError(Exception):
    """Custom exception raised for validation errors."""

def load_rules(rules: Union[str, Dict[str, Any]], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Load validation rules from a JSON/YAML string, file path, or dict.
    Args:
        rules: JSON/YAML/dict with rules.
        logger: Optional logger.
    Returns:
        Rules as a dict.
    """
    if isinstance(rules, dict):
        return rules
    if isinstance(rules, str):
        # Try to load as a file path, then as raw JSON/YAML string
        try:
            if rules.endswith(('.json', '.yaml', '.yml')):
                with open(rules, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = rules
            # Try JSON
            try:
                return json.loads(content)
            except Exception:
                if YAML_AVAILABLE:
                    return yaml.safe_load(content)
                else:
                    raise
        except Exception as e:
            if logger:
                logger.error(f"Failed to load rules: {e}")
            raise
    raise ValueError("Rules must be a dict or JSON/YAML string or file path.")

def validate_dataframe(
    df: pd.DataFrame,
    rules: Union[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Validate a DataFrame against declarative rules.

    Args:
        df: The DataFrame to validate.
        rules: Rules dict, JSON, or YAML.
        logger: Logger for detailed logging.
        raise_on_error: Whether to raise on validation error.

    Returns:
        report: Dict with validation results, errors, and warnings.
    """
    logger = logger or logging.getLogger(__name__)
    rules_dict = load_rules(rules, logger=logger)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for col, col_rules in rules_dict.get("columns", {}).items():
        if col not in df.columns:
            msg = f"Column '{col}' not found in DataFrame."
            errors.append({"column": col, "rule": "exists", "error": msg})
            logger.error(msg)
            continue

        # Null constraint
        if col_rules.get("not_null", False):
            n_null = df[col].isnull().sum()
            if n_null > 0:
                msg = f"{n_null} null values found in '{col}' but not_null=True."
                errors.append({"column": col, "rule": "not_null", "error": msg})
                logger.error(msg)

        # Uniqueness
        if col_rules.get("unique", False):
            n_unique = df[col].nunique(dropna=False)
            if n_unique < len(df):
                msg = f"Column '{col}' is not unique. {len(df) - n_unique} duplicates found."
                errors.append({"column": col, "rule": "unique", "error": msg})
                logger.error(msg)

        # Value range
        min_val = col_rules.get("min")
        max_val = col_rules.get("max")
        if min_val is not None or max_val is not None:
            invalid = []
            for idx, val in df[col].items():
                if pd.isnull(val):
                    continue
                if min_val is not None and val < min_val:
                    invalid.append((idx, val, f"less than min {min_val}"))
                if max_val is not None and val > max_val:
                    invalid.append((idx, val, f"greater than max {max_val}"))
            for idx, val, reason in invalid:
                msg = f"Value {val} at row {idx} violates {reason} for '{col}'."
                errors.append({"column": col, "rule": "range", "row": idx, "value": val, "error": msg})
                logger.error(msg)

        # Regex pattern
        pattern = col_rules.get("pattern")
        if pattern:
            try:
                regex = re.compile(pattern)
                for idx, val in df[col].items():
                    if pd.isnull(val):
                        continue
                    if not regex.fullmatch(str(val)):
                        msg = f"Value '{val}' at row {idx} does not match pattern '{pattern}' in '{col}'."
                        errors.append({"column": col, "rule": "pattern", "row": idx, "value": val, "error": msg})
                        logger.error(msg)
            except Exception as e:
                msg = f"Invalid regex pattern '{pattern}' for '{col}': {e}"
                errors.append({"column": col, "rule": "pattern", "error": msg})
                logger.error(msg)

        # Custom rule (warn only)
        custom_warn = col_rules.get("warn_if")
        if custom_warn:
            try:
                failed = df[~df[col].apply(custom_warn)]
                if not failed.empty:
                    for idx, val in failed[col].items():
                        msg = f"Custom warning for '{col}': value '{val}' at row {idx}."
                        warnings.append({"column": col, "rule": "warn_if", "row": idx, "value": val, "warning": msg})
                        logger.warning(msg)
            except Exception as e:
                msg = f"Custom warn_if rule failed for '{col}': {e}"
                warnings.append({"column": col, "rule": "warn_if", "warning": msg})
                logger.warning(msg)

        # Add rule success
        results.append({"column": col, "rules_checked": list(col_rules.keys())})

    report = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "results": results,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
    }

    logger.info(f"Validation completed. Errors: {len(errors)}, Warnings: {len(warnings)}")
    if raise_on_error and errors:
        raise ValidationError(f"Validation failed with {len(errors)} errors.")

    return report

# Example rule definition (YAML or JSON):
"""
columns:
  age:
    min: 0
    max: 120
    not_null: true
    unique: false
  email:
    pattern: "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$"
    not_null: true
    unique: true
"""

# Example:
# logger = logging.getLogger("validation")
# report = validate_dataframe(df, rules, logger=logger)
