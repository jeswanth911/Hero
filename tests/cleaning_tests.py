import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# --- Assume the cleaning functions are imported from cleaning.py ---
# from cleaning import (
#     deduplicate,
#     impute_missing,
#     detect_anomalies,
#     mask_pii,
#     normalize_schema,
# )

# Mock implementations for demonstration/testing
def deduplicate(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)

def impute_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: List[str] = None,
    fill_value: Any = None,
) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.columns.tolist()
    for col in cols:
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif strategy == "constant":
            df[col] = df[col].fillna(fill_value)
        else:
            df[col] = df[col].fillna(fill_value)
    return df

def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    # Simple z-score anomaly detection
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False]*len(df), index=df.index)
    mean = df[column].mean()
    std = df[column].std()
    return abs(df[column] - mean) > threshold * std

def mask_pii(df: pd.DataFrame, pii_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in pii_columns:
        df[col] = df[col].apply(lambda x: "***MASKED***" if pd.notnull(x) else x)
    return df

def normalize_schema(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns=column_map)
    return df

# --- Fixtures and Test Data ---

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 4],
        "name": ["Alice", "Bob", "Bob", "Charlie", "David", "David"],
        "email": ["a@x.com", "b@x.com", "b@x.com", None, "d@x.com", "d@x.com"],
        "age": [25, None, None, 35, 40, 40],
        "ssn": ["123-45-6789", "987-65-4321", "987-65-4321", None, "555-55-5555", "555-55-5555"],
        "income": [50000, 60000, 60000, 70000, 80000, 80000]
    })

@pytest.fixture
def edge_df():
    return pd.DataFrame({
        "id": [],
        "name": [],
        "email": [],
        "age": [],
        "ssn": [],
        "income": []
    })

@pytest.fixture
def missing_df():
    return pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, None, None],
        "c": [1, 2, 3]
    })

@pytest.fixture
def anomaly_df():
    return pd.DataFrame({
        "val": [10, 10, 10, 10, 1000]  # last value is an anomaly
    })

@pytest.fixture
def pii_df():
    return pd.DataFrame({
        "email": ["a@x.com", None, "b@x.com"],
        "ssn": ["123-45-6789", "987-65-4321", None],
        "name": ["Alice", "Bob", "Charlie"]
    })

@pytest.fixture
def schema_map():
    return {"email": "contact_email", "ssn": "social_security_number"}

# --- Deduplication ---

def test_deduplicate_simple(raw_df):
    deduped = deduplicate(raw_df, subset=["id", "email"])
    assert len(deduped) < len(raw_df)
    assert deduped.duplicated(subset=["id", "email"]).sum() == 0

def test_deduplicate_edge(edge_df):
    deduped = deduplicate(edge_df)
    assert deduped.empty

# --- Missing Value Imputation ---

@pytest.mark.parametrize("strategy,expected", [
    ("mean", [1, 2, 3]),
    ("median", [1, 2, 3]),
    ("mode", [1, 1, 1]),
    ("constant", [1, 0, 3]),
])
def test_impute_missing(missing_df, strategy, expected):
    df = impute_missing(missing_df, strategy=strategy, columns=["a"], fill_value=0)
    assert df["a"].tolist() == expected

def test_impute_missing_all_none(missing_df):
    df = impute_missing(missing_df, strategy="constant", columns=["b"], fill_value=-1)
    assert all(x == -1 for x in df["b"])

def test_impute_invalid_column(missing_df):
    with pytest.raises(KeyError):
        impute_missing(missing_df, columns=["not_a_column"])

# --- Anomaly Detection ---

def test_detect_anomalies(anomaly_df):
    result = detect_anomalies(anomaly_df, "val", threshold=3)
    assert result.iloc[-1]  # last is anomaly
    assert result.sum() == 1

def test_detect_anomalies_non_numeric():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    result = detect_anomalies(df, "x")
    assert not result.any()

# --- PII Masking ---

def test_mask_pii(pii_df):
    masked = mask_pii(pii_df, pii_columns=["email", "ssn"])
    assert all(x == "***MASKED***" or pd.isnull(x) for x in masked["email"])
    assert all(x == "***MASKED***" or pd.isnull(x) for x in masked["ssn"])
    assert (masked["name"] == pii_df["name"]).all()

def test_mask_pii_missing_column(pii_df):
    # Should not fail if column is missing
    masked = mask_pii(pii_df, pii_columns=["not_in_df"])
    assert masked.equals(pii_df)

# --- Schema Normalization ---

def test_normalize_schema(pii_df, schema_map):
    normed = normalize_schema(pii_df, schema_map)
    assert "contact_email" in normed.columns
    assert "social_security_number" in normed.columns

def test_normalize_schema_no_map(pii_df):
    normed = normalize_schema(pii_df, {})
    assert normed.equals(pii_df)

# --- Edge Cases and Invalid Inputs ---

def test_functions_on_empty(edge_df, schema_map):
    assert deduplicate(edge_df).empty
    assert impute_missing(edge_df).empty
    assert not detect_anomalies(edge_df, "income").any()
    assert mask_pii(edge_df, ["ssn"]).empty
    assert normalize_schema(edge_df, schema_map).empty

def test_invalid_inputs():
    with pytest.raises(Exception):
        deduplicate(None)
    with pytest.raises(Exception):
        impute_missing(None)
    with pytest.raises(Exception):
        detect_anomalies(None, "col")
    with pytest.raises(Exception):
        mask_pii(None, ["email"])
    with pytest.raises(Exception):
        normalize_schema(None, {})

# --- Test Isolation ---

def test_isolation(raw_df):
    # Ensure cleaning doesn't mutate input
    orig = raw_df.copy(deep=True)
    deduplicate(raw_df)
    impute_missing(raw_df)
    detect_anomalies(raw_df, "income")
    mask_pii(raw_df, ["ssn"])
    normalize_schema(raw_df, {"email": "contact_email"})
    pd.testing.assert_frame_equal(raw_df, orig)
