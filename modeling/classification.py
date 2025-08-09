import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    tf = None

class FeatureEngineering:
    """Feature engineering utilities for lag and rolling statistics."""

    @staticmethod
    def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Add lag features for specified columns."""
        for col in columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    @staticmethod
    def add_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int], stats: List[str] = ["mean", "std"]) -> pd.DataFrame:
        """Add rolling statistics (mean, std, min, max) for specified columns."""
        for col in columns:
            for window in windows:
                for stat in stats:
                    if stat == "mean":
                        df[f"{col}_roll_{window}_mean"] = df[col].rolling(window).mean()
                    elif stat == "std":
                        df[f"{col}_roll_{window}_std"] = df[col].rolling(window).std()
                    elif stat == "min":
                        df[f"{col}_roll_{window}_min"] = df[col].rolling(window).min()
                    elif stat == "max":
                        df[f"{col}_roll_{window}_max"] = df[col].rolling(window).max()
        return df

class TrendPredictor:
    """
    Classification and regression for trend detection and prediction.
    Supports Random Forest, Gradient Boosting, XGBoost, and simple Neural Networks.
    """

    def __init__(
        self,
        model_type: str = "rf",
        task: str = "classification",
        logger: Optional[logging.Logger] = None,
        random_state: int = 42,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        imbalance_method: Optional[str] = None # e.g., 'class_weight', 'smote'
    ):
        """
        Args:
            model_type: 'rf', 'gb', 'xgboost', 'nn'
            task: 'classification' or 'regression'
            logger: Logger instance
            random_state: Seed for reproducibility
            class_weight: 'balanced' or custom dict for classification
            imbalance_method: Optional - 'class_weight', 'smote', etc.
        """
        self.model_type = model_type.lower()
        self.task = task
        self.logger = logger or logging.getLogger(__name__)
        self.random_state = random_state
        self.class_weight = class_weight
        self.imbalance_method = imbalance_method
        self.model = None
        self.selected_features: Optional[List[str]] = None

    def _get_model(self, params: Optional[Dict[str, Any]] = None):
        """Instantiate the model according to type and task."""
        params = params or {}
        if self.task == "classification":
            if self.model_type == "rf":
                return RandomForestClassifier(
                    random_state=self.random_state, class_weight=self.class_weight, **params)
            elif self.model_type == "gb":
                return GradientBoostingClassifier(random_state=self.random_state, **params)
            elif self.model_type == "xgboost":
                if XGBClassifier is None:
                    raise ImportError("xgboost is not installed.")
                return XGBClassifier(random_state=self.random_state, **params)
            elif self.model_type == "nn":
                if tf is None:
                    raise ImportError("TensorFlow is not installed.")
                # NN will be built in fit()
                return "nn"
            else:
                raise ValueError("Unsupported model_type for classification.")
        else:
            # Regression
            if self.model_type == "rf":
                return RandomForestRegressor(random_state=self.random_state, **params)
            elif self.model_type == "gb":
                return GradientBoostingRegressor(random_state=self.random_state, **params)
            elif self.model_type == "xgboost":
                if XGBRegressor is None:
                    raise ImportError("xgboost is not installed.")
                return XGBRegressor(random_state=self.random_state, **params)
            elif self.model_type == "nn":
                if tf is None:
                    raise ImportError("TensorFlow is not installed.")
                # NN will be built in fit()
                return "nn"
            else:
                raise ValueError("Unsupported model_type for regression.")

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        feature_selection: bool = False,
        selection_model: Optional[str] = None,
        selection_threshold: float = 1e-5,
        params: Optional[Dict[str, Any]] = None,
        hyperparam_tuning: bool = False,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_iter: int = 10,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Fit the model. Includes optional feature selection and hyperparameter tuning.

        Args:
            X: Features DataFrame.
            y: Target.
            feature_selection: Whether to perform automated feature selection.
            selection_model: Model to use for feature selection ('rf', 'gb', 'xgboost').
            selection_threshold: Threshold for feature selection.
            params: Model parameters.
            hyperparam_tuning: Whether to run tuning.
            param_grid: Grid for hyperparameter search.
            n_iter: Number of random search iterations.
            validation_split: For NN, portion for validation.
            early_stopping: For NN, use early stopping.
            epochs: NN epochs.
            batch_size: NN batch size.

        Returns:
            Dict with model, selected_features, and (if tuning) best_params.
        """
        self.logger.info("Starting fit process for TrendPredictor.")
        # Basic: drop NA from X, y
        X = X.copy()
        y = pd.Series(y).copy()
        X = X.loc[y.notna()]
        y = y[y.notna()]
        X = X.fillna(X.mean())  # Basic imputation
        if self.task == "classification" and self.class_weight == "balanced":
            # Compute class weights
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weight = dict(zip(classes, weights))
            self.logger.info(f"Computed class weights: {self.class_weight}")

        # Feature selection
        if feature_selection:
            fs_model_type = selection_model or self.model_type
            fs_model = self._get_model()
            fs_model.fit(X, y)
            selector = SelectFromModel(fs_model, threshold=selection_threshold, prefit=True)
            mask = selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            X = X[self.selected_features]
            self.logger.info(f"Selected features: {self.selected_features}")
        else:
            self.selected_features = list(X.columns)

        # Hyperparameter tuning
        model = self._get_model(params)
        best_params = params
        if hyperparam_tuning and param_grid:
            if self.model_type == "nn":
                self.logger.warning("Hyperparameter tuning for neural networks is not supported in this version.")
            else:
                search_cls = RandomizedSearchCV if n_iter < len(ParameterGrid(param_grid)) else GridSearchCV
                search = search_cls(
                    estimator=model,
                    param_distributions=param_grid if search_cls == RandomizedSearchCV else param_grid,
                    n_iter=n_iter if search_cls == RandomizedSearchCV else None,
                    scoring="accuracy" if self.task == "classification" else "neg_mean_squared_error",
                    cv=3,
                    random_state=self.random_state,
                    verbose=0
                )
                search.fit(X, y)
                model = search.best_estimator_
                best_params = search.best_params_
                self.logger.info(f"Best params from tuning: {best_params}")

        # Fit final model
        if self.model_type == "nn":
            input_dim = X.shape[1]
            out_dim = len(np.unique(y)) if self.task == "classification" else 1
            model = Sequential()
            model.add(Dense(64, activation="relu", input_dim=input_dim))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))
            model.add(Dropout(0.1))
            final_activation = "softmax" if self.task == "classification" and out_dim > 2 else "sigmoid"
            model.add(Dense(out_dim, activation=final_activation))
            loss = "sparse_categorical_crossentropy" if self.task == "classification" and out_dim > 2 else \
                ("binary_crossentropy" if self.task == "classification" else "mse")
            model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
            callbacks = []
            if early_stopping:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
            model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            self.logger.info("Neural network trained.")
        else:
            model.fit(X, y)
            self.logger.info("Model trained.")

        self.model = model
        return {
            "model": model,
            "selected_features": self.selected_features,
            "best_params": best_params
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target for given features.

        Args:
            X: DataFrame of features.

        Returns:
            np.ndarray: Predictions.
        """
        if not self.model:
            raise RuntimeError("Model not fitted.")
        X = X[self.selected_features].fillna(X.mean())
        if self.model_type == "nn":
            preds = self.model.predict(X)
            if self.task == "classification":
                return np.argmax(preds, axis=1)
            else:
                return preds.flatten()
        else:
            return self.model.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X: Features.
            y_true: Ground-truth labels.

        Returns:
            Dict with performance metrics.
        """
        y_pred = self.predict(X)
        metrics = {}
        if self.task == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred, multi_class="ovr" if len(np.unique(y_true)) > 2 else "raise")
            except Exception as e:
                metrics["roc_auc"] = None
                self.logger.warning(f"ROC-AUC could not be computed: {e}")
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
            metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
        else:
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y_true, y_pred)
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    @staticmethod
    def handle_imbalance(X: pd.DataFrame, y: pd.Series, method: str = "class_weight") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced datasets.

        Args:
            X: Features.
            y: Target.
            method: 'class_weight', 'smote', etc.

        Returns:
            Tuple of (X, y)
        """
        if method == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE()
                X_res, y_res = sm.fit_resample(X, y)
                return X_res, y_res
            except ImportError:
                raise ImportError("imblearn is required for SMOTE.")
        # Default: return as is (class_weight handled in model)
        return X, y

# Example usage:
# logger = logging.getLogger("trend_predictor")
# fe = FeatureEngineering()
# df = fe.add_lag_features(df, ["feature1"], lags=[1,2])
# df = fe.add_rolling_features(df, ["feature1"], windows=[3,7])
# X = df.drop("target", axis=1)
# y = df["target"]
# predictor = TrendPredictor(model_type="xgboost", task="classification", logger=logger)
# predictor.fit(X, y, feature_selection=True, hyperparam_tuning=True, param_grid={"n_estimators": [50, 100]}, n_iter=2)
# preds = predictor.predict(X)
# metrics = predictor.evaluate(X, y)
