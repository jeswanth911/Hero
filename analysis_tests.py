import pytest
import pandas as pd
import numpy as np

# Assume the following functions are from analysis.py:
# from analysis import (
#     profile_statistics,
#     compute_correlation_matrix,
#     detect_time_series_trends,
#     extract_kpis,
#     summarize_naturally,
# )

# Mock implementations for demonstration
def profile_statistics(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "missing": df[col].isnull().sum()
            }
        else:
            stats[col] = {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "missing": df[col].isnull().sum()
            }
    return stats

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)

def detect_time_series_trends(ts: pd.Series) -> dict:
    # Dummy: Detect upward or downward trend, naive seasonality
    trend = "upward" if ts.iloc[-1] > ts.iloc[0] else "downward"
    seasonal = any(abs(ts.diff(12).dropna()) > 0.01) if len(ts) > 12 else False
    return {"trend": trend, "seasonal": seasonal}

def extract_kpis(df: pd.DataFrame, kpis: list) -> dict:
    # Dummy: Only supports "mean_sales" and "customer_count"
    result = {}
    if "mean_sales" in kpis and "sales" in df.columns:
        result["mean_sales"] = df["sales"].mean()
    if "customer_count" in kpis and "customer" in df.columns:
        result["customer_count"] = df["customer"].nunique()
    return result

def summarize_naturally(df: pd.DataFrame) -> str:
    # Dummy: just outputs row/col count
    return f"Data has {len(df)} rows and {len(df.columns)} columns."

# ---- Fixtures ----

@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [100, 200, 300, 400, 500]
    })

@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "category": ["x", "y", "x", "z", "y"],
        "value": [100, 200, np.nan, 400, 500],
        "flag": ["yes", "no", "yes", None, "no"]
    })

@pytest.fixture
def timeseries_df():
    dates = pd.date_range("2024-01-01", periods=24, freq="M")
    data = np.linspace(100, 200, 24) + np.sin(np.arange(24) / 2) * 10
    return pd.DataFrame({"date": dates, "sales": data})

@pytest.fixture
def kpi_df():
    return pd.DataFrame({
        "sales": [100, 200, 300, 400, 500],
        "customer": ["A", "B", "A", "C", "B"],
        "region": ["North", "South", "North", "East", "West"]
    })

# ---- Statistical Profiling ----

def test_profile_statistics_numeric(numeric_df):
    stats = profile_statistics(numeric_df)
    for col in numeric_df.columns:
        assert "mean" in stats[col]
        assert abs(stats[col]["mean"] - numeric_df[col].mean()) < 1e-5

def test_profile_statistics_mixed(mixed_df):
    stats = profile_statistics(mixed_df)
    assert "unique" in stats["category"]
    assert stats["value"]["missing"] == 1
    assert "top" in stats["flag"]

# ---- Correlation Matrix ----

def test_correlation_matrix_numeric(numeric_df):
    corr = compute_correlation_matrix(numeric_df)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.shape == (3, 3)
    # Correlation between a and b should be 1.0 for perfect linear data
    assert abs(corr.loc["a", "b"] - 1.0) < 1e-8

def test_correlation_matrix_mixed(mixed_df):
    corr = compute_correlation_matrix(mixed_df)
    assert "id" in corr.columns
    assert "value" in corr.columns
    assert "category" not in corr.columns  # categorical dropped

# ---- Time Series Trend/Seasonality ----

def test_time_series_trend(timeseries_df):
    ts = timeseries_df["sales"]
    res = detect_time_series_trends(ts)
    assert res["trend"] in ["upward", "downward"]
    assert isinstance(res["seasonal"], bool)

def test_time_series_short():
    ts = pd.Series([1, 2, 3, 2, 1])
    res = detect_time_series_trends(ts)
    assert "trend" in res
    assert res["seasonal"] is False

# ---- KPI Extraction ----

def test_extract_kpis(kpi_df):
    kpis = extract_kpis(kpi_df, ["mean_sales", "customer_count"])
    assert "mean_sales" in kpis and abs(kpis["mean_sales"] - kpi_df["sales"].mean()) < 1e-8
    assert "customer_count" in kpis and kpis["customer_count"] == kpi_df["customer"].nunique()

def test_extract_kpis_missing(kpi_df):
    kpis = extract_kpis(kpi_df, ["unknown_kpi"])
    assert kpis == {}

# ---- Natural Language Summaries ----

def test_summarize_naturally(numeric_df):
    summary = summarize_naturally(numeric_df)
    assert isinstance(summary, str)
    assert f"{len(numeric_df)} rows" in summary

def test_summarize_naturally_empty():
    df = pd.DataFrame()
    summary = summarize_naturally(df)
    assert "0 rows" in summary

# ---- Categorical Data Coverage ----

def test_profile_statistics_categorical():
    df = pd.DataFrame({"cat": ["a", "b", "b", "c", None, "a"]})
    stats = profile_statistics(df)
    assert stats["cat"]["unique"] == 3
    assert stats["cat"]["missing"] == 1
    assert stats["cat"]["top"] in ["a", "b", "c"]

# ---- Repeatability (Mocked Output) ----

def test_mocked_repeatability(numeric_df):
    # Same input should yield same stats/output
    s1 = profile_statistics(numeric_df)
    s2 = profile_statistics(numeric_df.copy())
    assert s1 == s2

# ---- Edge Cases ----

def test_profile_statistics_empty():
    df = pd.DataFrame()
    stats = profile_statistics(df)
    assert isinstance(stats, dict)
    assert stats == {}