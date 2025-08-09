import pytest
import numpy as np
import pandas as pd

# For demonstration, we mock model and utility functions.
# In a real codebase, import from modeling.py:
# from modeling import (
#     forecast_prophet,
#     forecast_arima,
#     forecast_lstm,
#     run_classification,
#     run_regression,
#     tune_hyperparameters,
#     feature_engineering,
#     compute_metrics,
#     explain_model,
# )

# --- MOCKED IMPLEMENTATIONS ---
def forecast_prophet(df, periods=5):
    # Returns last value + np.arange(periods) for simplicity
    y = df["y"].values[-1] + np.arange(1, periods + 1)
    return y

def forecast_arima(df, periods=5):
    y = df["y"].values[-1] + np.arange(1, periods + 1)
    return y

def forecast_lstm(df, periods=5):
    y = df["y"].values[-1] + np.arange(1, periods + 1)
    return y

def run_classification(X, y, model="logreg"):
    # Mock: returns accuracy, predictions (random)
    np.random.seed(0)
    preds = np.random.choice(np.unique(y), size=len(y))
    acc = (preds == y).mean()
    return acc, preds

def run_regression(X, y, model="linreg"):
    # Mock: returns RMSE, predictions (random noise)
    np.random.seed(0)
    preds = y + np.random.normal(0, 0.1, size=len(y))
    rmse = np.sqrt(((preds - y) ** 2).mean())
    return rmse, preds

def tune_hyperparameters(model, X, y, param_grid):
    # Mock: returns best_params (first from grid)
    best_params = {k: v[0] for k, v in param_grid.items()}
    return best_params

def feature_engineering(df):
    # Mock: Add a feature
    df = df.copy()
    df["feat_sum"] = df.sum(axis=1)
    return df

def compute_metrics(y_true, y_pred, task="classification"):
    if task == "classification":
        accuracy = (y_true == y_pred).mean()
        return {"accuracy": accuracy}
    elif task == "regression":
        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        return {"rmse": rmse}
    else:
        raise ValueError("Unknown task")

def explain_model(model, X, method="shap"):
    # Mock: returns "shap values"
    return np.ones(X.shape[1])

# --- FIXTURES AND TEST DATA ---

@pytest.fixture
def ts_df():
    # Synthetic time series
    dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
    y = np.linspace(10, 30, 20) + np.random.normal(0, 1, 20)
    return pd.DataFrame({"ds": dates, "y": y})

@pytest.fixture
def regression_df():
    X = np.random.rand(100, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + np.random.normal(0, 0.1, 100)
    return X, y

@pytest.fixture
def classification_df():
    X = np.random.rand(100, 3)
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # imbalanced
    return X, y

@pytest.fixture
def imbalanced_classification_df():
    X = np.random.rand(50, 2)
    y = np.array([0]*45 + [1]*5)  # Strongly imbalanced
    return X, y

@pytest.fixture
def param_grid():
    return {"alpha": [0.01, 0.1, 1], "max_iter": [100, 200]}

# --- FORECASTING TESTS ---

@pytest.mark.parametrize("forecast_func", [forecast_prophet, forecast_arima, forecast_lstm])
def test_time_series_forecasting_accuracy(ts_df, forecast_func):
    # Use last 5 for "ground truth", train on first 15
    train = ts_df.iloc[:15]
    test = ts_df.iloc[15:]
    pred = forecast_func(train, periods=5)
    assert len(pred) == 5
    # Simple error check: should be roughly around test values
    assert np.abs(pred - test["y"].values).mean() < 20  # lenient for mock

# --- CLASSIFICATION TESTS ---

def test_classification_accuracy(classification_df):
    X, y = classification_df
    acc, preds = run_classification(X, y)
    assert 0.0 <= acc <= 1.0
    m = compute_metrics(y, preds, task="classification")
    assert "accuracy" in m

def test_classification_imbalanced(imbalanced_classification_df):
    X, y = imbalanced_classification_df
    acc, preds = run_classification(X, y)
    # Check class coverage
    assert set(preds).issubset({0, 1})
    assert 0.0 <= acc <= 1.0

# --- REGRESSION TESTS ---

def test_regression_rmse(regression_df):
    X, y = regression_df
    rmse, preds = run_regression(X, y)
    assert rmse >= 0
    m = compute_metrics(y, preds, task="regression")
    assert "rmse" in m
    assert rmse == pytest.approx(m["rmse"])

# --- HYPERPARAMETER TUNING ---

def test_hyperparameter_tuning(param_grid, classification_df):
    X, y = classification_df
    best_params = tune_hyperparameters("dummy_model", X, y, param_grid)
    assert best_params["alpha"] == 0.01
    assert best_params["max_iter"] == 100

# --- FEATURE ENGINEERING ---

def test_feature_engineering(regression_df):
    X, y = regression_df
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df_enriched = feature_engineering(df)
    assert "feat_sum" in df_enriched.columns
    assert np.allclose(df_enriched["feat_sum"], df.sum(axis=1))

# --- MODEL EXPLAINABILITY ---

def test_explainability_output(regression_df):
    X, y = regression_df
    shap_values = explain_model("dummy_model", X, method="shap")
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.shape[0] == X.shape[1]

# --- EDGE CASES ---

def test_forecasting_on_short_series():
    df = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=3), "y": [1, 2, 3]})
    pred = forecast_prophet(df, periods=2)
    assert len(pred) == 2

def test_classification_single_class():
    X = np.random.rand(10, 2)
    y = np.zeros(10, dtype=int)
    acc, preds = run_classification(X, y)
    assert all(p == 0 for p in preds)

def test_regression_constant_y():
    X = np.random.rand(10, 2)
    y = np.ones(10)
    rmse, preds = run_regression(X, y)
    assert rmse >= 0
