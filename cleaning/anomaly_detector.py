import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

class AnomalyDetector:
    """
    Detects anomalies in Pandas DataFrames using statistical and ML-based methods.

    Features:
    - Statistical: z-score, IQR for univariate
    - ML-based: Isolation Forest, Local Outlier Factor for multivariate
    - Per-column or combined feature modes
    - Detailed anomaly reports (row indices, severity scores)
    - Configurable thresholds and model parameters
    - Logging and robust error handling
    """

    SUPPORTED_METHODS = ['zscore', 'iqr', 'isolation_forest', 'lof']

    def __init__(
        self,
        method: str = 'zscore',
        threshold: Optional[Union[float, Dict[str, float]]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        mode: str = 'combined',
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            method: 'zscore', 'iqr', 'isolation_forest', or 'lof'
            threshold: threshold for anomaly detection (float or dict per column)
            model_params: parameters for ML models
            mode: 'per_column' or 'combined'
            logger: logging.Logger instance
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {self.SUPPORTED_METHODS}")
        if mode not in ['per_column', 'combined']:
            raise ValueError("mode must be 'per_column' or 'combined'")
        self.method = method
        self.threshold = threshold
        self.model_params = model_params or {}
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.fitted_ = False
        self.cols_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detector (for ML-based methods).

        Args:
            df: DataFrame to fit on
            columns: Columns to use (defaults to all numeric)
        Returns:
            self
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cols_ = columns

        if self.method == 'isolation_forest':
            params = {'n_estimators': 100, 'contamination': 'auto', 'random_state': 42}
            params.update(self.model_params)
            self.model = IsolationForest(**params)
            self.model.fit(df[columns].values)
            self.fitted_ = True
            self.logger.info(f"Fitted Isolation Forest on columns: {columns}")
        elif self.method == 'lof':
            params = {'n_neighbors': 20, 'contamination': 'auto'}
            params.update(self.model_params)
            self.model = LocalOutlierFactor(**params, novelty=True)
            self.model.fit(df[columns].values)
            self.fitted_ = True
            self.logger.info(f"Fitted Local Outlier Factor on columns: {columns}")
        else:
            # For statistical methods, fitting is not needed
            self.fitted_ = True
            self.logger.info(f"Statistical method '{self.method}' does not require fitting.")
        return self

    def predict(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in the DataFrame.

        Args:
            df: DataFrame to predict on
            columns: Columns to use (defaults to those from fit or all numeric)
        Returns:
            report: Dict with anomalies: 
                {
                    "anomalies": List[Dict],
                    "method": str,
                    "mode": str,
                    "columns": list,
                }
                Each anomaly dict: {"row": idx, "column": col, "score": float, "severity": float}
        """
        if not self.fitted_:
            raise RuntimeError("AnomalyDetector must be fitted before calling predict().")
        if columns is None:
            columns = self.cols_ or df.select_dtypes(include=[np.number]).columns.tolist()
        report = {
            "anomalies": [],
            "method": self.method,
            "mode": self.mode,
            "columns": columns,
        }
        try:
            if self.method == 'zscore':
                report["anomalies"] = self._zscore_anomalies(df, columns)
            elif self.method == 'iqr':
                report["anomalies"] = self._iqr_anomalies(df, columns)
            elif self.method == 'isolation_forest':
                report["anomalies"] = self._isoforest_anomalies(df, columns)
            elif self.method == 'lof':
                report["anomalies"] = self._lof_anomalies(df, columns)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            raise
        self.logger.info(f"Detected {len(report['anomalies'])} anomalies using {self.method} ({self.mode})")
        return report

    def _zscore_anomalies(self, df, columns):
        anomalies = []
        thresh = self.threshold or 3.0
        if self.mode == 'per_column':
            for col in columns:
                if col not in df.columns:
                    continue
                try:
                    scores = np.abs(zscore(df[col].astype(float), nan_policy='omit'))
                    for idx, score in enumerate(scores):
                        if np.isnan(score):
                            continue
                        t = thresh[col] if isinstance(thresh, dict) and col in thresh else thresh
                        if score > t:
                            anomalies.append({"row": idx, "column": col, "score": float(score), "severity": float(score)})
                except Exception as e:
                    self.logger.warning(f"Skipping column '{col}' for z-score anomaly: {e}")
        else:  # combined
            data = df[columns].astype(float)
            scores = np.abs(zscore(data, nan_policy='omit', axis=0)).max(axis=1)
            for idx, score in enumerate(scores):
                if np.isnan(score):
                    continue
                if score > thresh:
                    # Mark as anomaly for all contributing columns
                    for col in columns:
                        anomalies.append({"row": idx, "column": col, "score": float(score), "severity": float(score)})
        return anomalies

    def _iqr_anomalies(self, df, columns):
        anomalies = []
        thresh = self.threshold or 1.5
        if self.mode == 'per_column':
            for col in columns:
                if col not in df.columns:
                    continue
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - thresh * iqr
                    upper = q3 + thresh * iqr
                    for idx, val in df[col].items():
                        if pd.isnull(val):
                            continue
                        if val < lower or val > upper:
                            severity = max(abs(val - lower), abs(val - upper)) / (iqr + 1e-9)
                            anomalies.append({"row": idx, "column": col, "score": float(val), "severity": float(severity)})
                except Exception as e:
                    self.logger.warning(f"Skipping column '{col}' for IQR anomaly: {e}")
        else:  # combined
            # Use Mahalanobis or sum IQR scores (for simplicity, use sum)
            iqrs = {col: df[col].quantile(0.75) - df[col].quantile(0.25) for col in columns}
            q1s = {col: df[col].quantile(0.25) for col in columns}
            q3s = {col: df[col].quantile(0.75) for col in columns}
            for idx, row in df[columns].iterrows():
                score = 0
                for col in columns:
                    iqr = iqrs[col]
                    lower = q1s[col] - thresh * iqr
                    upper = q3s[col] + thresh * iqr
                    val = row[col]
                    if pd.isnull(val):
                        continue
                    if val < lower or val > upper:
                        score += 1
                if score > 0:
                    for col in columns:
                        anomalies.append({"row": idx, "column": col, "score": float(score), "severity": float(score / len(columns))})
        return anomalies

    def _isoforest_anomalies(self, df, columns):
        if not self.model:
            raise RuntimeError("IsolationForest model is not fitted.")
        X = df[columns].values
        try:
            anomaly_scores = -self.model.score_samples(X)  # higher is more anomalous
            preds = self.model.predict(X)
            anomalies = []
            for idx, (pred, score) in enumerate(zip(preds, anomaly_scores)):
                if pred == -1:
                    anomalies.append({
                        "row": idx,
                        "column": "combined",
                        "score": float(score),
                        "severity": float(score)
                    })
            return anomalies
        except Exception as e:
            self.logger.error(f"IsolationForest anomaly scoring failed: {e}")
            raise

    def _lof_anomalies(self, df, columns):
        if not self.model:
            raise RuntimeError("LocalOutlierFactor model is not fitted.")
        X = df[columns].values
        try:
            anomaly_scores = -self.model.score_samples(X)  # higher is more anomalous
            preds = self.model.predict(X)
            anomalies = []
            for idx, (pred, score) in enumerate(zip(preds, anomaly_scores)):
                if pred == -1:
                    anomalies.append({
                        "row": idx,
                        "column": "combined",
                        "score": float(score),
                        "severity": float(score)
                    })
            return anomalies
        except Exception as e:
            self.logger.error(f"LOF anomaly scoring failed: {e}")
            raise

    def get_supported_methods(self) -> List[str]:
        return self.SUPPORTED_METHODS

    def get_params(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "model_params": self.model_params,
            "mode": self.mode,
            "columns": self.cols_
        }
