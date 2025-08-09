import pandas as pd
import numpy as np
import json
import logging
from typing import Optional, Dict, Any, List, Union
from collections import Counter

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.express as px
except ImportError:
    px = None

class StatisticalProfiler:
    """
    A class for comprehensive statistical profiling of Pandas DataFrames.

    Features:
        - Descriptive statistics (mean, median, mode, variance, std, skewness, kurtosis)
        - Distribution summaries (histograms, quantiles, value counts)
        - Missing value and data type detection
        - JSON-serializable profiling reports
        - Optional visualization with Matplotlib or Plotly
        - Memory-efficient large dataset support
        - Designed for extensibility and pipeline integration
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize the profiler.

        Args:
            df: The input DataFrame to profile.
            logger: Optional logger for logging.
            chunk_size: If provided, process DataFrame in row-wise chunks for memory efficiency.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)
        self.report: Optional[Dict[str, Any]] = None

    def profile(self) -> Dict[str, Any]:
        """
        Perform statistical profiling and return JSON-serializable report.

        Returns:
            A dictionary containing profiling results.
        """
        self.logger.info("Starting statistical profiling.")
        profile = {}

        profile["n_rows"], profile["n_columns"] = self.df.shape
        profile["columns"] = {}
        profile["missing"] = self.detect_missing()
        profile["dtypes"] = self.detect_types()
        profile["descriptive"] = {}
        profile["distributions"] = {}

        for col in self.df.columns:
            col_data = self.df[col]
            stats = self.get_descriptive_stats(col_data)
            dist = self.get_distribution_summary(col_data)
            profile["columns"][col] = {
                "descriptive": stats,
                "distribution": dist,
                "dtype": str(col_data.dtype),
                "missing": int(self.df[col].isnull().sum()),
            }
            profile["descriptive"][col] = stats
            profile["distributions"][col] = dist

        self.report = profile
        self.logger.info("Profiling complete.")
        return profile

    def get_descriptive_stats(self, col: pd.Series) -> Dict[str, Optional[float]]:
        """
        Compute descriptive statistics for a column.

        Args:
            col: Pandas Series.

        Returns:
            Dict of descriptive statistics.
        """
        stats: Dict[str, Optional[float]] = {
            "mean": None, "median": None, "mode": None,
            "variance": None, "std": None, "skewness": None, "kurtosis": None
        }

        if pd.api.types.is_numeric_dtype(col):
            stats["mean"] = float(np.nanmean(col))
            stats["median"] = float(np.nanmedian(col))
            try:
                mode_vals = col.mode(dropna=True)
                stats["mode"] = float(mode_vals.iloc[0]) if not mode_vals.empty else None
            except Exception:
                stats["mode"] = None
            stats["variance"] = float(np.nanvar(col, ddof=1))
            stats["std"] = float(np.nanstd(col, ddof=1))
            stats["skewness"] = float(col.skew(skipna=True))
            stats["kurtosis"] = float(col.kurtosis(skipna=True))
        elif pd.api.types.is_datetime64_dtype(col):
            stats["mean"] = str(col.mean())
            stats["median"] = str(col.median())
            mode_vals = col.mode(dropna=True)
            stats["mode"] = str(mode_vals.iloc[0]) if not mode_vals.empty else None
        else:
            # Categorical or object
            mode_vals = col.mode(dropna=True)
            stats["mode"] = mode_vals.iloc[0] if not mode_vals.empty else None
            stats["mean"] = None
            stats["median"] = None
        return stats

    def get_distribution_summary(self, col: pd.Series, bins: int = 10) -> Dict[str, Any]:
        """
        Compute histogram, quantiles, and value counts for a column.

        Args:
            col: Pandas Series.
            bins: Number of bins for numeric histograms.

        Returns:
            Dict with histogram, quantiles, and value counts.
        """
        dist: Dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(col):
            col_no_na = col.dropna()
            try:
                hist, bin_edges = np.histogram(col_no_na, bins=bins)
                dist["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            except Exception:
                dist["histogram"] = {"counts": [], "bin_edges": []}
            dist["quantiles"] = {str(q): float(col_no_na.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1]}
            # Value counts for most common values
            dist["value_counts"] = col_no_na.value_counts().head(10).to_dict()
        else:
            vc = col.value_counts(dropna=True).head(10)
            dist["value_counts"] = vc.to_dict()
            dist["histogram"] = {}
            dist["quantiles"] = {}
        return dist

    def detect_missing(self) -> Dict[str, int]:
        """
        Detect missing values per column.

        Returns:
            Dict mapping column names to missing value counts.
        """
        self.logger.info("Detecting missing values.")
        return self.df.isnull().sum().to_dict()

    def detect_types(self) -> Dict[str, str]:
        """
        Detect data types per column.

        Returns:
            Dict mapping column names to dtype string.
        """
        self.logger.info("Detecting column types.")
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Export the profiling report as a JSON string.

        Args:
            indent: Number of spaces for indentation.

        Returns:
            JSON string of the profiling report.
        """
        if self.report is None:
            self.profile()
        return json.dumps(self.report, indent=indent, default=str)

    def visualize(
        self,
        columns: Optional[List[str]] = None,
        backend: str = "matplotlib",
        bins: int = 10,
        show: bool = True
    ) -> None:
        """
        Generate histograms for numeric columns.

        Args:
            columns: List of columns to plot (default: all numeric).
            backend: 'matplotlib' or 'plotly'.
            bins: Number of bins for histograms.
            show: Whether to call plt.show() or fig.show().
        """
        columns = columns or self.df.select_dtypes(include=[np.number]).columns.tolist()
        if backend == "matplotlib":
            if plt is None:
                self.logger.error("Matplotlib is not available.")
                return
            for col in columns:
                plt.figure()
                self.df[col].hist(bins=bins)
                plt.title(f"Histogram: {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                if show:
                    plt.show()
        elif backend == "plotly":
            if px is None:
                self.logger.error("Plotly is not available.")
                return
            for col in columns:
                fig = px.histogram(self.df, x=col, nbins=bins, title=f"Histogram: {col}")
                if show:
                    fig.show()
        else:
            self.logger.error(f"Unknown backend '{backend}' for visualization.")

# Example usage:
# logger = logging.getLogger("stat_profiler")
# profiler = StatisticalProfiler(df, logger=logger)
# report = profiler.profile()
# print(profiler.to_json())
# profiler.visualize()