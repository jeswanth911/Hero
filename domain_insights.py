import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import json

class KPIExtractor:
    """
    Extracts and computes KPIs using domain-specific templates.
    """

    DEFAULT_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
        "finance": {
            "Total Revenue": {"columns": ["revenue", "total_revenue"], "func": "sum"},
            "Net Profit": {"columns": ["net_profit", "profit"], "func": "sum"},
            "Profit Margin": {
                "columns": ["net_profit", "revenue"],
                "func": lambda df: (df["net_profit"].sum() / df["revenue"].sum()) if "net_profit" in df and "revenue" in df else None,
            },
            "Return on Assets": {
                "columns": ["net_income", "total_assets"],
                "func": lambda df: (df["net_income"].sum() / df["total_assets"].sum()) if "net_income" in df and "total_assets" in df else None,
            },
        },
        "healthcare": {
            "Patient Count": {"columns": ["patient_id", "patient"], "func": "nunique"},
            "Average Stay": {"columns": ["length_of_stay", "stay_duration"], "func": "mean"},
            "Readmission Rate": {
                "columns": ["readmitted", "patient_id"],
                "func": lambda df: (
                    df["readmitted"].sum() / df["patient_id"].nunique()
                ) if "readmitted" in df and "patient_id" in df else None,
            },
        },
        "retail": {
            "Total Sales": {"columns": ["sales", "total_sales", "amount"], "func": "sum"},
            "Average Basket Size": {
                "columns": ["basket_size", "items_purchased"],
                "func": "mean",
            },
            "Customer Count": {"columns": ["customer_id", "customer"], "func": "nunique"},
        },
        "logistics": {
            "Total Shipments": {"columns": ["shipment_id", "tracking_id"], "func": "nunique"},
            "Average Delivery Time": {
                "columns": ["delivery_time", "transit_days"], "func": "mean",
            },
            "On-Time Delivery Rate": {
                "columns": ["on_time_flag", "shipment_id"],
                "func": lambda df: (
                    df["on_time_flag"].sum() / df["shipment_id"].nunique()
                ) if "on_time_flag" in df and "shipment_id" in df else None,
            },
        },
    }

    def __init__(
        self,
        industry: str,
        custom_templates: Optional[Dict[str, Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            industry: Industry domain key (e.g., 'finance', 'healthcare', ...)
            custom_templates: Optionally supply/override KPI templates.
            logger: Logger instance.
        """
        self.industry = industry.lower()
        self.templates = self.DEFAULT_TEMPLATES.get(self.industry, {}).copy()
        if custom_templates:
            self.templates.update(custom_templates)
        self.logger = logger or logging.getLogger(__name__)

    def detect_and_compute_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects relevant KPI columns and computes KPI values.

        Args:
            df: Input DataFrame.

        Returns:
            Dict of computed KPIs.
        """
        kpi_results = {}
        lower_cols = [c.lower() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        for kpi_name, kpi in self.templates.items():
            found_cols = []
            for alt_col in kpi["columns"]:
                if alt_col in lower_cols:
                    found_cols.append(col_map[alt_col])
            if not found_cols:
                self.logger.info(f"KPI '{kpi_name}' skipped (columns not found).")
                continue

            try:
                if callable(kpi["func"]):
                    value = kpi["func"](df)
                else:
                    col = found_cols[0]
                    if kpi["func"] == "sum":
                        value = df[col].sum()
                    elif kpi["func"] == "mean":
                        value = df[col].mean()
                    elif kpi["func"] == "nunique":
                        value = df[col].nunique()
                    else:
                        value = None
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    kpi_results[kpi_name] = value
                    self.logger.info(f"Computed KPI '{kpi_name}': {value}")
            except Exception as e:
                self.logger.warning(f"Failed to compute KPI '{kpi_name}': {e}")

        return kpi_results


class DomainInsights:
    """
    Extracts industry-specific KPIs and actionable insights from datasets.
    """

    def __init__(
        self,
        industry: str,
        kpi_templates: Optional[Dict[str, Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            industry: Industry domain.
            kpi_templates: Optional custom/extended KPI templates.
            logger: Logger instance.
        """
        self.industry = industry
        self.kpi_extractor = KPIExtractor(industry, custom_templates=kpi_templates, logger=logger)
        self.logger = logger or logging.getLogger(__name__)

    def extract_insights(
        self,
        df: pd.DataFrame,
        statistics: Optional[Dict[str, Any]] = None,
        correlations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main method: Extract KPIs, trends, risks, and opportunities.

        Args:
            df: Input DataFrame.
            statistics: Output from statistical profiling (optional).
            correlations: Output from correlation analysis (optional).

        Returns:
            JSON-serializable dictionary of insights.
        """
        self.logger.info(f"Extracting KPIs for industry '{self.industry}'.")
        kpis = self.kpi_extractor.detect_and_compute_kpis(df)

        summary_nl, risks, opportunities = self._summarize(
            kpis, statistics or {}, correlations or {}
        )

        insights = {
            "industry": self.industry,
            "kpis": kpis,
            "summary": summary_nl,
            "risks": risks,
            "opportunities": opportunities,
        }
        return insights

    def _summarize(
        self,
        kpis: Dict[str, Any],
        statistics: Dict[str, Any],
        correlations: Dict[str, Any],
    ) -> Tuple[str, List[str], List[str]]:
        """
        Generate a natural language summary, risks, and opportunities.

        Args:
            kpis: Dict of computed KPIs.
            statistics: Statistical profiler output.
            correlations: Correlation analyzer output.

        Returns:
            (summary, risks, opportunities)
        """
        lines = []
        risks = []
        opportunities = []

        if not kpis:
            lines.append("No key performance indicators (KPIs) could be computed for this dataset.")
        else:
            lines.append("Key KPIs identified:")
            for k, v in kpis.items():
                lines.append(f"- {k}: {v:,.2f}" if isinstance(v, (int, float)) else f"- {k}: {v}")

        # Example integration of statistics/correlations for insights
        if statistics:
            for col, stats in (statistics.get("descriptive") or {}).items():
                if isinstance(stats, dict) and "mean" in stats and stats["mean"] is not None:
                    if stats.get("std", 0) > 2 * abs(stats["mean"]):
                        risks.append(f"High variance detected in '{col}'.")
                    if stats.get("skewness", 0) > 1:
                        risks.append(f"Significant right skew in '{col}'.")
                    elif stats.get("skewness", 0) < -1:
                        risks.append(f"Significant left skew in '{col}'.")
        if correlations:
            ranked = (correlations.get("ranked_dependencies") or [])
            for dep in ranked[:3]:
                f1 = dep.get("feature_1")
                f2 = dep.get("feature_2")
                score = dep.get("score", 0)
                if score > 0.9:
                    risks.append(f"Strong dependency between '{f1}' and '{f2}' (correlation: {score:.2f}). Consider reducing redundancy.")
                elif score > 0.7:
                    opportunities.append(f"Moderate correlation between '{f1}' and '{f2}' ({score:.2f}) may yield actionable insight.")

        if not risks:
            risks.append("No significant risks detected based on available data.")
        if not opportunities:
            opportunities.append("No clear opportunities identified; further domain analysis recommended.")

        summary = "\n".join(lines)
        return summary, risks, opportunities

    def to_json(self, insights: Dict[str, Any], indent: int = 2) -> str:
        """
        Output insights as JSON string.

        Args:
            insights: Insights dictionary.
            indent: Indent for JSON.

        Returns:
            JSON string.
        """
        try:
            return json.dumps(insights, indent=indent, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to serialize insights to JSON: {e}")
            return "{}"

# Example usage:
# logger = logging.getLogger("domain_insights")
# insights_engine = DomainInsights("finance", logger=logger)
# insights = insights_engine.extract_insights(df, statistics=stat_report, correlations=corr_summary)
# print(insights_engine.to_json(insights))