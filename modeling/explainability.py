import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    lime = None
    LimeTabularExplainer = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class ModelExplainer:
    """
    Provides model interpretability using SHAP and LIME for tree-based and neural models.
    
    Supports global and local explanation, human-readable reports, and visualizations.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        mode: str = "classification",
        logger: Optional[logging.Logger] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Trained model (tree-based, neural, or any compatible with SHAP/LIME)
            X_train: Training data used for explainer fitting
            mode: 'classification' or 'regression'
            logger: Optional logger
            feature_names: Column names of input features
            class_names: Optional class names for classification
        """
        self.model = model
        self.X_train = X_train
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        self.feature_names = feature_names or list(X_train.columns)
        self.class_names = class_names
        self.shap_explainer = None
        self.lime_explainer = None
        self._init_explainers()

    def _init_explainers(self):
        """Initialize SHAP and LIME explainers based on model type."""
        # SHAP
        if shap is not None:
            try:
                if hasattr(self.model, "predict_proba") or hasattr(self.model, "predict"):
                    # Tree-based models
                    if "tree" in str(type(self.model)).lower():
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    # Neural networks (keras, tensorflow)
                    elif "keras" in str(type(self.model)).lower() or "tensorflow" in str(type(self.model)).lower():
                        self.shap_explainer = shap.DeepExplainer(self.model, self.X_train)
                    # Generic
                    else:
                        self.shap_explainer = shap.Explainer(self.model, self.X_train)
                else:
                    self.logger.warning("Model may not be compatible with SHAP.")
            except Exception as e:
                self.logger.warning(f"SHAP explainer initialization failed: {e}")

        # LIME
        if LimeTabularExplainer is not None:
            try:
                self.lime_explainer = LimeTabularExplainer(
                    training_data=np.array(self.X_train),
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode=self.mode
                )
            except Exception as e:
                self.logger.warning(f"LIME explainer initialization failed: {e}")

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_idx: Optional[int] = None,
        method: str = "shap",
        nsamples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate explanations for the model.

        Args:
            X: DataFrame or np.ndarray for explanation (one or more samples)
            instance_idx: Index of instance for local explanation (None for global)
            method: 'shap' or 'lime'
            nsamples: Number of samples for SHAP summary (global)

        Returns:
            Dict with explanation results.
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X

        results = {}

        if method == "shap" and shap is not None and self.shap_explainer is not None:
            try:
                if instance_idx is not None:
                    shap_values = self.shap_explainer(X_np[instance_idx:instance_idx+1])
                    results["local"] = {
                        "values": shap_values.values[0].tolist(),
                        "base_value": float(shap_values.base_values[0]),
                        "feature_names": self.feature_names
                    }
                else:
                    shap_values = self.shap_explainer(X_np[:nsamples])
                    results["global"] = {
                        "values": shap_values.values.tolist(),
                        "base_values": [float(b) for b in np.array(shap_values.base_values).flatten()],
                        "feature_names": self.feature_names
                    }
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {e}")
        elif method == "lime" and LimeTabularExplainer is not None and self.lime_explainer is not None:
            try:
                if instance_idx is None:
                    instance_idx = 0
                exp = self.lime_explainer.explain_instance(
                    X_np[instance_idx],
                    self.model.predict_proba if self.mode == "classification" else self.model.predict,
                    num_features=len(self.feature_names)
                )
                results["local"] = {
                    "explanation": exp.as_list(),
                    "instance": X_np[instance_idx].tolist(),
                    "feature_names": self.feature_names
                }
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {e}")
        else:
            self.logger.warning("No valid explanation method or explainer available.")
        return results

    def plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_idx: Optional[int] = None,
        method: str = "shap",
        nsamples: int = 100,
        show: bool = True
    ) -> None:
        """
        Visualize model explanations.

        Args:
            X: DataFrame or np.ndarray for explanation (one or more samples)
            instance_idx: Index of instance for local explanation (None for global)
            method: 'shap' or 'lime'
            nsamples: Number of samples for global SHAP summary
            show: Whether to call plt.show()
        """
        if plt is None:
            self.logger.warning("matplotlib not available for plotting.")
            return

        if method == "shap" and shap is not None and self.shap_explainer is not None:
            try:
                if instance_idx is not None:
                    shap_values = self.shap_explainer(X.values[instance_idx:instance_idx+1] if isinstance(X, pd.DataFrame) else X[instance_idx:instance_idx+1])
                    shap.plots.waterfall(shap_values[0], show=show)
                else:
                    shap_values = self.shap_explainer(X.values[:nsamples] if isinstance(X, pd.DataFrame) else X[:nsamples])
                    shap.summary_plot(shap_values, X.values[:nsamples] if isinstance(X, pd.DataFrame) else X[:nsamples], feature_names=self.feature_names, show=show)
            except Exception as e:
                self.logger.warning(f"SHAP plot failed: {e}")
        elif method == "lime" and LimeTabularExplainer is not None and self.lime_explainer is not None:
            try:
                if instance_idx is None:
                    instance_idx = 0
                exp = self.lime_explainer.explain_instance(
                    X.values[instance_idx] if isinstance(X, pd.DataFrame) else X[instance_idx],
                    self.model.predict_proba if self.mode == "classification" else self.model.predict,
                    num_features=len(self.feature_names)
                )
                fig = exp.as_pyplot_figure()
                if show:
                    plt.show()
            except Exception as e:
                self.logger.warning(f"LIME plot failed: {e}")
        else:
            self.logger.warning("No valid explanation method or explainer available.")

    def human_readable_report(
        self,
        explanation: Dict[str, Any],
        instance_idx: Optional[int] = None
    ) -> str:
        """
        Generate a human-readable explanation report.

        Args:
            explanation: Output from explain()
            instance_idx: Index of explained instance (if local)

        Returns:
            String report.
        """
        report = []
        if "local" in explanation:
            report.append(f"Local explanation for instance {instance_idx if instance_idx is not None else 0}:")
            for fname, val in zip(explanation["local"].get("feature_names", []), explanation["local"].get("values", [])):
                report.append(f"  Feature '{fname}': contribution {val:.4f}")
            if "base_value" in explanation["local"]:
                report.append(f"  Base value: {explanation['local']['base_value']:.4f}")
            if "explanation" in explanation["local"]:
                for feat, contrib in explanation["local"]["explanation"]:
                    report.append(f"  {feat}: {contrib:+.4f}")
        elif "global" in explanation:
            report.append("Global feature importance summary:")
            vals = np.abs(np.mean(explanation["global"]["values"], axis=0))
            sorted_idx = np.argsort(vals)[::-1]
            for idx in sorted_idx:
                report.append(f"  Feature '{explanation['global']['feature_names'][idx]}': mean(|contribution|) = {vals[idx]:.4f}")
        else:
            report.append("No explanation data available.")
        return "\n".join(report)

# Example usage:
# logger = logging.getLogger("explainer")
# explainer = ModelExplainer(model, X_train, mode="classification", logger=logger)
# explanation = explainer.explain(X_test, instance_idx=0, method="shap")
# print(explainer.human_readable_report(explanation, instance_idx=0))
# explainer.plot(X_test, instance_idx=0, method="shap")
