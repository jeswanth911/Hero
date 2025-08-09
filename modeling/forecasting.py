import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    tf = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

class Forecaster:
    """
    Advanced time series forecasting supporting Prophet, ARIMA, SARIMA, and LSTM.
    """

    def __init__(
        self,
        model_type: str = "prophet",
        logger: Optional[logging.Logger] = None,
        random_state: int = 42,
    ):
        """
        Args:
            model_type: One of 'prophet', 'arima', 'sarima', 'lstm'
            logger: Optional logger.
            random_state: For reproducibility.
        """
        self.model_type = model_type.lower()
        self.logger = logger or logging.getLogger(__name__)
        self.model: Any = None
        self.fitted: bool = False
        self.history: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.random_state = random_state

    def train(
        self,
        df: pd.DataFrame,
        target: str,
        time_col: str,
        exog: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        tune: bool = False,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_iter: int = 10,
        test_size: int = 0,
    ) -> Dict[str, Any]:
        """
        Train chosen model on data.

        Args:
            df: DataFrame containing the time series.
            target: Target variable column name.
            time_col: Time column name.
            exog: List of exogenous variable column names (for SARIMA, LSTM).
            params: Model parameters.
            tune: Whether to perform hyperparameter tuning.
            param_grid: Grid or distributions for tuning.
            n_iter: Number of random search iterations.
            test_size: Number of final points to leave out for validation.

        Returns:
            Dict with best_params and training metrics.
        """
        self.logger.info(f"Training {self.model_type.upper()} model...")
        # Prepare data
        df = df.sort_values(by=time_col)
        if test_size > 0:
            train_df = df.iloc[:-test_size]
            val_df = df.iloc[-test_size:]
        else:
            train_df = df
            val_df = None

        # Tuning
        if tune and param_grid:
            best_score = float("inf")
            best_params = None
            if self.model_type in ["arima", "sarima"]:
                sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=self.random_state))
            else:
                sampler = list(ParameterGrid(param_grid))
            for i, p in enumerate(sampler):
                self.logger.info(f"Tuning: trying params {p} ({i+1}/{len(sampler)})")
                try:
                    model = self._fit_model(train_df, target, time_col, exog, params=p)
                    preds = self._predict_model(model, val_df, target, time_col, exog, len(val_df)) if val_df is not None else None
                    if preds is not None:
                        score = mean_squared_error(val_df[target], preds)
                        if score < best_score:
                            best_score = score
                            best_params = p
                except Exception as e:
                    self.logger.warning(f"Tuning failed for params {p}: {e}")
            params = best_params
            self.best_params = params
            self.logger.info(f"Best params: {params} with score {best_score}")
        else:
            self.best_params = params

        # Fit final model
        self.model = self._fit_model(train_df, target, time_col, exog, params=self.best_params)
        self.fitted = True
        self.history = train_df

        # Metrics on validation if available
        metrics = {}
        if val_df is not None:
            preds = self._predict_model(self.model, val_df, target, time_col, exog, len(val_df))
            y_true = val_df[target].values
            metrics = {
                "MAE": float(mean_absolute_error(y_true, preds)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
                "MAPE": float(mape(y_true, preds))
            }
            self.logger.info(f"Validation metrics: {metrics}")

        return {"best_params": self.best_params, "metrics": metrics}

    def _fit_model(
        self,
        train_df: pd.DataFrame,
        target: str,
        time_col: str,
        exog: Optional[List[str]],
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Internal: Fit model based on type.
        """
        params = params or {}
        if self.model_type == "prophet":
            if Prophet is None:
                raise ImportError("prophet is not installed.")
            dfp = train_df.rename(columns={time_col: "ds", target: "y"})
            model = Prophet(**params)
            model.fit(dfp)
            return model
        elif self.model_type == "arima":
            order = params.get("order", (1, 0, 0))
            model = ARIMA(train_df[target], order=order)
            return model.fit()
        elif self.model_type == "sarima":
            order = params.get("order", (1, 0, 0))
            seasonal_order = params.get("seasonal_order", (0, 0, 0, 0))
            exog_data = train_df[exog] if exog else None
            model = SARIMAX(train_df[target], exog=exog_data, order=order, seasonal_order=seasonal_order)
            return model.fit(disp=False)
        elif self.model_type == "lstm":
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            # Only univariate for simplicity; for multivariate, stack exog features
            lookback = params.get("lookback", 10)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 10)
            X, y = self._create_lstm_data(train_df, target, lookback, exog)
            model = Sequential()
            model.add(LSTM(params.get("units", 50), input_shape=(X.shape[1], X.shape[2])))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer="adam")
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            return (model, lookback, exog)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _predict_model(
        self,
        model: Any,
        df: pd.DataFrame,
        target: str,
        time_col: str,
        exog: Optional[List[str]],
        steps: int
    ) -> np.ndarray:
        """
        Internal: Predict next N steps.
        """
        if self.model_type == "prophet":
            future = df.rename(columns={time_col: "ds"})
            forecast = model.predict(future)
            return forecast["yhat"].values
        elif self.model_type == "arima":
            forecast = model.forecast(steps=steps)
            return forecast.values
        elif self.model_type == "sarima":
            exog_data = df[exog] if exog else None
            forecast = model.forecast(steps=steps, exog=exog_data)
            return forecast.values
        elif self.model_type == "lstm":
            model_obj, lookback, exog_feats = model
            X, _ = self._create_lstm_data(df, target, lookback, exog_feats)
            preds = model_obj.predict(X, verbose=0)
            return preds.flatten()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(
        self,
        future_df: pd.DataFrame,
        target: str,
        time_col: str,
        exog: Optional[List[str]] = None,
        steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict future values.

        Args:
            future_df: DataFrame with future time points.
            target: Target variable column.
            time_col: Time column name.
            exog: List of exogenous variable names.
            steps: Number of time steps to predict.

        Returns:
            Predicted values as numpy array.
        """
        if not self.fitted:
            raise RuntimeError("Model is not trained. Call train() first.")
        steps = steps or len(future_df)
        preds = self._predict_model(self.model, future_df, target, time_col, exog, steps)
        self.logger.info(f"Predicted {steps} future points.")
        return preds

    def validate(
        self,
        val_df: pd.DataFrame,
        target: str,
        time_col: str,
        exog: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Validate the model on a hold-out set.

        Args:
            val_df: Validation DataFrame.
            target: Target variable column name.
            time_col: Time column name.
            exog: Exogenous variable columns.

        Returns:
            Dict of MAE, RMSE, MAPE.
        """
        y_true = val_df[target].values
        preds = self._predict_model(self.model, val_df, target, time_col, exog, len(val_df))
        metrics = {
            "MAE": float(mean_absolute_error(y_true, preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
            "MAPE": float(mape(y_true, preds))
        }
        self.logger.info(f"Validation metrics: {metrics}")
        return metrics

    def _create_lstm_data(
        self,
        df: pd.DataFrame,
        target: str,
        lookback: int,
        exog: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM.

        Args:
            df: DataFrame.
            target: Target column.
            lookback: Number of lags.
            exog: List of exogenous variables.

        Returns:
            X, y arrays for LSTM.
        """
        values = df[target].values
        exog_values = df[exog].values if exog else None
        X, y = [], []
        for i in range(lookback, len(values)):
            x_seq = values[i - lookback:i]
            if exog:
                exog_seq = exog_values[i - lookback:i]
                X.append(np.hstack([x_seq.reshape(-1, 1), exog_seq]))
            else:
                X.append(x_seq.reshape(-1, 1))
            y.append(values[i])
        X = np.array(X)
        y = np.array(y)
        if exog:
            # If multivariate, stack all features
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
        else:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def plot_forecast(
        self,
        actual: pd.Series,
        forecast: Union[pd.Series, np.ndarray],
        time_col: Optional[Union[str, pd.Series]] = None,
        title: str = "Forecast vs Actual",
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot actual vs. forecasted values.

        Args:
            actual: Actual time series.
            forecast: Forecasted values.
            time_col: Time column or values.
            title: Plot title.
            figsize: Figure size.
        """
        if plt is None:
            self.logger.warning("matplotlib not available for plotting.")
            return
        plt.figure(figsize=figsize)
        if time_col is not None:
            plt.plot(time_col, actual, label="Actual")
            plt.plot(time_col, forecast, label="Forecast")
        else:
            plt.plot(actual, label="Actual")
            plt.plot(forecast, label="Forecast")
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.show()

# Example usage:
# logger = logging.getLogger("forecasting")
# forecaster = Forecaster(model_type="prophet", logger=logger)
# forecaster.train(df, target="y", time_col="ds", tune=True, param_grid={"seasonality_mode": ["additive", "multiplicative"]}, test_size=30)
# future = ... # DataFrame with future dates
# preds = forecaster.predict(future, target="y", time_col="ds")
# forecaster.plot_forecast(actual=df["y"], forecast=preds, time_col=df["ds"])
