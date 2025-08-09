import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import zscore

try:
    import ruptures as rpt
except ImportError:
    rpt = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class TimeseriesAnalyzer:
    """
    Advanced time series analysis for Pandas DataFrames.

    Capabilities:
    - Trend, seasonality, and residual extraction (STL)
    - Stationarity tests (ADF, KPSS)
    - Anomaly and change point detection
    - Lag feature and rolling stats generation
    - Visualization of components and detected patterns
    - Designed for large datasets, extensibility, and integration
    """

    def __init__(
        self,
        ts: pd.Series,
        period: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            ts: Univariate time series (Pandas Series with datetime index).
            period: Seasonal period (default: infer from frequency or user input).
            logger: Optional logger.
        """
        if not isinstance(ts, pd.Series):
            raise TypeError("Input ts must be a pandas Series.")
        if not pd.api.types.is_datetime64_any_dtype(ts.index):
            raise ValueError("Series index must be datetime64.")
        self.ts = ts.sort_index()
        self.period = period or self._infer_period()
        self.logger = logger or logging.getLogger(__name__)
        self.components: Optional[Dict[str, pd.Series]] = None
        self.stationarity: Optional[Dict[str, Any]] = None
        self.anomalies: Optional[pd.DataFrame] = None
        self.changepoints: Optional[List[int]] = None

    def _infer_period(self) -> int:
        """Infer seasonal period using Pandas frequency or heuristics."""
        inferred_freq = pd.infer_freq(self.ts.index)
        if inferred_freq:
            freq_map = {
                "D": 7,
                "W": 52,
                "M": 12,
                "Q": 4,
                "A": 1,
                "H": 24,
                "T": 60,
            }
            for k, v in freq_map.items():
                if k in inferred_freq:
                    return v
        # fallback: guess weekly for daily, yearly for monthly, else 12
        n = len(self.ts)
        if n > 365 * 2:
            return 365
        elif n > 60:
            return 12
        else:
            return 7

    def decompose(self, robust: bool = True) -> Dict[str, pd.Series]:
        """
        Decompose the time series into trend, seasonal, and residuals using STL.

        Args:
            robust: Use robust fitting (default: True).

        Returns:
            Dict of components: {'trend', 'seasonal', 'resid'}
        """
        self.logger.info("Starting STL decomposition.")
        stl = STL(self.ts, period=self.period, robust=robust)
        result = stl.fit()
        self.components = {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid
        }
        self.logger.info("STL decomposition complete.")
        return self.components

    def stationarity_tests(self, regression: str = "c") -> Dict[str, Any]:
        """
        Run ADF and KPSS stationarity tests.

        Args:
            regression: 'c' (constant), 'ct' (constant + trend), passed to statsmodels tests.

        Returns:
            Dict with test statistics and p-values.
        """
        self.logger.info("Performing stationarity tests (ADF, KPSS).")
        results = {}
        ts_clean = self.ts.dropna()
        try:
            adf_res = adfuller(ts_clean, regression=regression, autolag='AIC')
            results['adf'] = {
                "statistic": adf_res[0],
                "pvalue": adf_res[1],
                "n_lags": adf_res[2],
                "n_obs": adf_res[3],
                "critical_values": adf_res[4],
                "stationary": adf_res[1] < 0.05
            }
        except Exception as e:
            self.logger.warning(f"ADF test failed: {e}")
            results['adf'] = None
        try:
            kpss_res = kpss(ts_clean, regression=regression, nlags="auto")
            results['kpss'] = {
                "statistic": kpss_res[0],
                "pvalue": kpss_res[1],
                "n_lags": kpss_res[2],
                "critical_values": kpss_res[3],
                "stationary": kpss_res[1] > 0.05
            }
        except Exception as e:
            self.logger.warning(f"KPSS test failed: {e}")
            results['kpss'] = None
        self.stationarity = results
        return results

    def detect_anomalies(self, z_thresh: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in the residual component using z-score.

        Args:
            z_thresh: Z-score threshold for anomaly (default: 3.0).

        Returns:
            DataFrame with anomalies: index, value, zscore
        """
        if self.components is None:
            self.decompose()
        resid = self.components["resid"]
        zs = zscore(resid.dropna())
        anomaly_idx = np.where(np.abs(zs) > z_thresh)[0]
        anomalies = resid.iloc[anomaly_idx]
        df_anom = pd.DataFrame({
            "timestamp": anomalies.index,
            "value": anomalies.values,
            "zscore": zs[anomaly_idx]
        })
        self.anomalies = df_anom
        self.logger.info(f"Detected {len(df_anom)} anomalies using z-score > {z_thresh}.")
        return df_anom

    def detect_changepoints(self, model: str = "l2", pen: float = 10) -> List[int]:
        """
        Detect change points in the trend component using the 'ruptures' package.

        Args:
            model: Cost model for ruptures ("l1", "l2", "rbf", etc.)
            pen: Penalty value for controlling change point sensitivity.

        Returns:
            List of change point indices.
        """
        if rpt is None:
            self.logger.warning("ruptures package not available. Change point detection skipped.")
            return []
        if self.components is None:
            self.decompose()
        trend = self.components["trend"].dropna().values.reshape(-1, 1)
        algo = rpt.Pelt(model=model).fit(trend)
        result = algo.predict(pen=pen)
        self.changepoints = result[:-1]  # exclude endpoint
        self.logger.info(f"Detected {len(self.changepoints)} change points.")
        return self.changepoints

    def generate_lag_features(
        self,
        lags: Union[int, List[int]] = 1,
        rolling_windows: Optional[List[int]] = None,
        stat_funcs: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features and rolling statistics.

        Args:
            lags: Number(s) of lags to compute.
            rolling_windows: List of window sizes for rolling stats.
            stat_funcs: List of statistics to compute (mean, std, min, max).

        Returns:
            DataFrame with new features.
        """
        lags_list = [lags] if isinstance(lags, int) else lags
        rolling_windows = rolling_windows or []
        stat_funcs = stat_funcs or ["mean", "std"]

        df_feat = pd.DataFrame(index=self.ts.index)
        for lag in lags_list:
            df_feat[f"lag_{lag}"] = self.ts.shift(lag)
        for win in rolling_windows:
            for stat in stat_funcs:
                colname = f"roll_{stat}_{win}"
                if stat == "mean":
                    df_feat[colname] = self.ts.rolling(win).mean()
                elif stat == "std":
                    df_feat[colname] = self.ts.rolling(win).std()
                elif stat == "min":
                    df_feat[colname] = self.ts.rolling(win).min()
                elif stat == "max":
                    df_feat[colname] = self.ts.rolling(win).max()
        self.logger.info(f"Generated lag and rolling features: {df_feat.columns.tolist()}")
        return df_feat

    def plot_components(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot STL decomposition components.

        Args:
            figsize: Size of the plot.
        """
        if plt is None:
            self.logger.warning("matplotlib not available for plotting.")
            return
        if self.components is None:
            self.decompose()
        plt.figure(figsize=figsize)
        for i, (name, comp) in enumerate(self.components.items()):
            plt.subplot(3, 1, i + 1)
            plt.plot(comp, label=name)
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_anomalies(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot anomalies on the time series.

        Args:
            figsize: Size of the plot.
        """
        if plt is None:
            self.logger.warning("matplotlib not available for plotting.")
            return
        if self.anomalies is None:
            self.detect_anomalies()
        plt.figure(figsize=figsize)
        plt.plot(self.ts, label="Time Series")
        plt.scatter(self.anomalies["timestamp"], self.anomalies["value"], color="red", label="Anomalies")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_changepoints(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot detected change points on the trend component.

        Args:
            figsize: Size of the plot.
        """
        if plt is None:
            self.logger.warning("matplotlib not available for plotting.")
            return
        if self.changepoints is None:
            self.detect_changepoints()
        if self.components is None:
            self.decompose()
        plt.figure(figsize=figsize)
        plt.plot(self.components["trend"], label="Trend")
        for cp in self.changepoints or []:
            plt.axvline(self.ts.index[cp], color="orange", linestyle="--", label="Change Point" if cp == self.changepoints[0] else "")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage:
# logger = logging.getLogger("ts_analyzer")
# ts = pd.Series(...)  # time series with datetime index
# analyzer = TimeseriesAnalyzer(ts, logger=logger)
# analyzer.decompose()
# analyzer.stationarity_tests()
# analyzer.detect_anomalies()
# analyzer.detect_changepoints()
# feats = analyzer.generate_lag_features(lags=[1,7], rolling_windows=[7,30])
# analyzer.plot_components()
# analyzer.plot_anomalies()
# analyzer.plot_changepoints()
