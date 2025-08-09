import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, List, Union, Callable
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, train_test_split, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, mean_squared_error, mean_absolute_error, r2_score
)
import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class ModelEvaluator:
    """
    Comprehensive model validation and performance tracking for classification and regression.
    """

    def __init__(
        self,
        model: Any,
        task: str = "classification",
        logger: Optional[logging.Logger] = None,
        experiment_name: Optional[str] = None,
        track_with_mlflow: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            model: The estimator (sklearn-like API: fit/predict/predict_proba).
            task: 'classification' or 'regression'.
            logger: Optional logger.
            experiment_name: MLflow experiment name.
            track_with_mlflow: Enable MLflow tracking.
            random_state: Seed for reproducibility.
        """
        self.model = model
        self.task = task
        self.logger = logger or logging.getLogger(__name__)
        self.random_state = random_state
        self.track_with_mlflow = track_with_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name or "DefaultExperiment"
        self.metrics_history: List[Dict[str, Any]] = []

        if self.track_with_mlflow:
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow tracking enabled for experiment '{self.experiment_name}'.")

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        stratified: bool = True,
        time_series: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation and return mean/standard deviation for metrics.

        Args:
            X: Features.
            y: Targets.
            cv: Number of folds.
            scoring: Scoring metric(s).
            stratified: Use StratifiedKFold for classification.
            time_series: Use TimeSeriesSplit for backtesting.

        Returns:
            Dict with scores.
        """
        if time_series:
            splitter = TimeSeriesSplit(n_splits=cv)
        elif self.task == "classification" and stratified:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        if scoring is None:
            scoring = "accuracy" if self.task == "classification" else "neg_mean_squared_error"

        self.logger.info(f"Starting cross-validation: {cv} folds, scoring={scoring}")
        scores = cross_val_score(self.model, X, y, cv=splitter, scoring=scoring)
        result = {"mean_score": np.mean(scores), "std_score": np.std(scores), "all_scores": scores.tolist()}
        self._log_metrics({"cross_val_" + scoring: result})
        return result

    def holdout_validate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform hold-out validation.

        Args:
            X: Features.
            y: Targets.
            test_size: Fraction for holdout.
            metrics: List of metrics to compute.

        Returns:
            Dict with computed metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if self.task == "classification" else None
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        if self.task == "classification":
            y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
            result = self._classification_metrics(y_test, y_pred, y_proba, metrics)
        else:
            result = self._regression_metrics(y_test, y_pred, metrics)
        self._log_metrics({"holdout": result})
        return result

    def backtest(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform time series backtesting using expanding window.

        Args:
            X: Features.
            y: Targets.
            n_splits: Number of splits.
            metrics: List of metrics to compute.

        Returns:
            List of metric dicts for each split.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        split_idx = 1
        for train_idx, test_idx in tscv.split(X):
            self.logger.info(f"Backtest split {split_idx}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            if self.task == "classification":
                y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
                res = self._classification_metrics(y_test, y_pred, y_proba, metrics)
            else:
                res = self._regression_metrics(y_test, y_pred, metrics)
            res["split"] = split_idx
            results.append(res)
            split_idx += 1
        self._log_metrics({"backtest": results})
        return results

    def _classification_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)
        if "precision" in metrics:
            results["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        if "recall" in metrics:
            results["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        if "f1" in metrics:
            results["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        if "roc_auc" in metrics and y_proba is not None:
            try:
                results["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                results["roc_auc"] = None
                self.logger.warning(f"roc_auc calculation failed: {e}")
        return results

    def _regression_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute regression metrics."""
        metrics = metrics or ["mae", "rmse", "r2"]
        results = {}
        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y_true, y_pred)
        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        if "r2" in metrics:
            results["r2"] = r2_score(y_true, y_pred)
        return results

    def track_metrics_over_time(
        self,
        metrics: Dict[str, Any],
        timestamp: Optional[Union[str, datetime.datetime]] = None
    ):
        """
        Track and log metrics for drift or trend detection.

        Args:
            metrics: Metrics dictionary.
            timestamp: Timestamp (default: now).
        """
        timestamp = timestamp or datetime.datetime.utcnow().isoformat()
        entry = {"timestamp": timestamp, "metrics": metrics}
        self.metrics_history.append(entry)
        self.logger.info(f"Tracking metrics at {timestamp}: {metrics}")
        if self.track_with_mlflow:
            with mlflow.start_run():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                    elif isinstance(v, dict):
                        for subk, subv in v.items():
                            if isinstance(subv, (int, float)):
                                mlflow.log_metric(f"{k}_{subk}", subv)
                mlflow.log_param("timestamp", timestamp)

    def plot_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        normalize: bool = False,
        labels: Optional[List[str]] = None,
        show: bool = True
    ):
        """
        Plot confusion matrix.

        Args:
            X: Features.
            y: Target.
            normalize: Normalize confusion matrix.
            labels: Class labels.
            show: Whether to call plt.show().
        """
        if self.task != "classification":
            self.logger.warning("Confusion matrix is only valid for classification tasks.")
            return
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=labels, normalize="true" if normalize else None)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels)) if labels else np.arange(cm.shape[0])
        plt.xticks(tick_marks, labels or tick_marks)
        plt.yticks(tick_marks, labels or tick_marks)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], ".2f"), ha="center", va="center", color="red")
        plt.tight_layout()
        if show:
            plt.show()

    def plot_roc_curve(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        show: bool = True
    ):
        """
        Plot ROC curve.

        Args:
            X: Features.
            y: Target.
            show: Whether to call plt.show().
        """
        if self.task != "classification":
            self.logger.warning("ROC curve is only valid for classification tasks.")
            return
        if not hasattr(self.model, "predict_proba"):
            self.logger.warning("Model does not support probability prediction.")
            return
        y_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label="ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def plot_residuals(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        show: bool = True
    ):
        """
        Plot residuals (for regression).

        Args:
            X: Features.
            y: Target.
            show: Whether to call plt.show().
        """
        if self.task != "regression":
            self.logger.warning("Residual plot is only valid for regression tasks.")
            return
        y_pred = self.model.predict(X)
        residuals = y - y_pred
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.tight_layout()
        if show:
            plt.show()

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Internal: Log metrics to history and MLflow if enabled."""
        self.track_metrics_over_time(metrics)

# Example usage:
# logger = logging.getLogger("model_evaluator")
# evaluator = ModelEvaluator(model, task="classification", logger=logger, track_with_mlflow=True)
# cv_scores = evaluator.cross_validate(X, y, cv=5)
# holdout_metrics = evaluator.holdout_validate(X, y)
# backtest_results = evaluator.backtest(X, y)
# evaluator.plot_confusion_matrix(X, y)
# evaluator.plot_roc_curve(X, y)
# evaluator.plot_residuals(X, y)