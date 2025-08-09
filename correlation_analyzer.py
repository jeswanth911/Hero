import pandas as pd
import numpy as np
import logging
import json
from typing import Optional, Dict, Any, List, Tuple, Union

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    sns = None
    plt = None

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V statistic for categorical-categorical association.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = (
        ((confusion_matrix - confusion_matrix.mean()) ** 2) / confusion_matrix.mean()
    ).values.sum()
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if n == 0 or min_dim == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))

class CorrelationAnalyzer:
    """
    Analyzes feature correlations in datasets:
    - Computes Pearson, Spearman, Kendall, and Cramér's V for categorical
    - Identifies multicollinearity and redundant features
    - Visualizes correlation heatmaps
    - Ranks feature dependencies
    - Outputs JSON-friendly summaries
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
        corr_threshold: float = 0.8,
        top_n: Optional[int] = 20,
        ignore_features: Optional[List[str]] = None,
    ):
        """
        Args:
            df: Input DataFrame
            logger: Optional logger
            corr_threshold: Threshold for multicollinearity flagging
            top_n: Number of top dependencies to report
            ignore_features: Features to exclude from analysis
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df
        self.corr_threshold = corr_threshold
        self.top_n = top_n
        self.ignore_features = set(ignore_features or [])
        self.logger = logger or logging.getLogger(__name__)
        self.numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in self.ignore_features
        ]
        self.categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c not in self.ignore_features
        ]
        self.corr_matrices: Dict[str, pd.DataFrame] = {}
        self.cramers_v_matrix: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict[str, Any]] = None

    def compute_correlations(self) -> Dict[str, pd.DataFrame]:
        """
        Compute correlation matrices for Pearson, Spearman, Kendall, and Cramér's V.
        Returns:
            Dict of DataFrames for each correlation type.
        """
        corr_types = ["pearson", "spearman", "kendall"]
        result = {}
        for corr in corr_types:
            try:
                mat = self.df[self.numeric_cols].corr(method=corr)
                result[corr] = mat
                self.logger.info(f"{corr.title()} correlation matrix computed.")
            except Exception as e:
                self.logger.error(f"Could not compute {corr} correlation: {e}")
        # Cramér's V for categorical-categorical
        if len(self.categorical_cols) > 1:
            cv_mat = pd.DataFrame(
                np.nan, index=self.categorical_cols, columns=self.categorical_cols
            )
            for i, col1 in enumerate(self.categorical_cols):
                for col2 in self.categorical_cols[i:]:
                    try:
                        v = cramers_v(self.df[col1], self.df[col2])
                        cv_mat.loc[col1, col2] = v
                        cv_mat.loc[col2, col1] = v
                    except Exception as e:
                        self.logger.warning(f"Could not compute Cramér's V for {col1}, {col2}: {e}")
            result["cramers_v"] = cv_mat
            self.cramers_v_matrix = cv_mat
        self.corr_matrices = result
        return result

    def find_multicollinearity(self) -> Dict[str, List[str]]:
        """
        Identify multicollinear and redundant features in numeric data.
        Returns:
            Dict mapping features to lists of highly correlated features.
        """
        if "pearson" not in self.corr_matrices:
            self.compute_correlations()
        mat = self.corr_matrices["pearson"].copy()
        np.fill_diagonal(mat.values, 0)
        multicollinear = {}
        for col in mat.columns:
            high_corr = mat.index[(np.abs(mat[col]) >= self.corr_threshold)].tolist()
            if high_corr:
                multicollinear[col] = high_corr
                self.logger.info(f"Feature '{col}' is highly correlated with: {high_corr}")
        return multicollinear

    def rank_feature_dependencies(self) -> List[Tuple[str, str, float, str]]:
        """
        Generate a ranked list of feature dependencies and their strength.
        Returns:
            List of (feature1, feature2, score, method) tuples, sorted descending by score.
        """
        rankings = []
        # Numeric: Pearson, Spearman, Kendall
        for method, mat in self.corr_matrices.items():
            if method == "cramers_v":
                continue
            for i, col1 in enumerate(mat.columns):
                for col2 in mat.columns[i+1:]:
                    score = abs(mat.loc[col1, col2])
                    rankings.append((col1, col2, score, method))
        # Categorical: Cramér's V
        if self.cramers_v_matrix is not None:
            for i, col1 in enumerate(self.cramers_v_matrix.columns):
                for col2 in self.cramers_v_matrix.columns[i+1:]:
                    val = self.cramers_v_matrix.loc[col1, col2]
                    if pd.notnull(val):
                        rankings.append((col1, col2, abs(val), "cramers_v"))
        rankings.sort(key=lambda x: x[2], reverse=True)
        if self.top_n:
            rankings = rankings[: self.top_n]
        self.logger.info(f"Top {self.top_n or len(rankings)} feature dependencies ranked.")
        return rankings

    def correlation_summary(self) -> Dict[str, Any]:
        """
        Generate a JSON-friendly summary including correlation matrices,
        multicollinearity, and ranked dependencies.
        """
        if not self.corr_matrices:
            self.compute_correlations()
        multi = self.find_multicollinearity()
        ranked = self.rank_feature_dependencies()
        summary = {
            "correlation_matrices": {k: v.round(4).to_dict() for k, v in self.corr_matrices.items()},
            "multicollinearity": multi,
            "ranked_dependencies": [
                {
                    "feature_1": f1, "feature_2": f2, "score": float(score), "method": method
                }
                for (f1, f2, score, method) in ranked
            ]
        }
        self.summary = summary
        return summary

    def to_json(self, indent: int = 2) -> str:
        """
        Output the summary as a JSON string.
        """
        if self.summary is None:
            self.correlation_summary()
        return json.dumps(self.summary, indent=indent, default=str)

    def plot_heatmap(
        self, 
        method: str = "pearson", 
        figsize: Tuple[int, int] = (10, 8), 
        mask_upper: bool = True, 
        annot: bool = False, 
        cmap: str = "coolwarm"
    ) -> None:
        """
        Visualize correlation matrix as a heatmap.

        Args:
            method: Correlation method to visualize.
            figsize: Figure size.
            mask_upper: Whether to mask upper triangle for clarity.
            annot: Whether to annotate each cell with value.
            cmap: Colormap.
        """
        if sns is None or plt is None:
            self.logger.warning("Seaborn/matplotlib not available for visualization.")
            return
        if not self.corr_matrices:
            self.compute_correlations()
        if method not in self.corr_matrices:
            self.logger.error(f"Method '{method}' not found in computed correlation matrices.")
            return
        mat = self.corr_matrices[method]
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(mat, dtype=bool))
        plt.figure(figsize=figsize)
        sns.heatmap(mat, mask=mask, cmap=cmap, annot=annot, fmt=".2f", square=True, cbar=True)
        plt.title(f"{method.title()} Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_cramers_v_heatmap(self, figsize: Tuple[int, int] = (10, 8), annot: bool = False, cmap: str = "YlGnBu") -> None:
        """
        Visualize Cramér's V matrix as a heatmap.

        Args:
            figsize: Figure size.
            annot: Annotate each cell with value.
            cmap: Colormap.
        """
        if sns is None or plt is None:
            self.logger.warning("Seaborn/matplotlib not available for visualization.")
            return
        if self.cramers_v_matrix is None:
            self.compute_correlations()
        if self.cramers_v_matrix is None:
            self.logger.error("Cramér's V matrix not computed.")
            return
        plt.figure(figsize=figsize)
        sns.heatmap(self.cramers_v_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True, cbar=True)
        plt.title("Cramér's V Correlation Heatmap (Categorical)")
        plt.tight_layout()
        plt.show()

# Example usage:
# logger = logging.getLogger("correlation_analyzer")
# analyzer = CorrelationAnalyzer(df, logger=logger, corr_threshold=0.85, top_n=10)
# analyzer.compute_correlations()
# summary = analyzer.correlation_summary()
# print(analyzer.to_json())
# analyzer.plot_heatmap(method="pearson")
# analyzer.plot_cramers_v_heatmap()