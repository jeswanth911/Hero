import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
import plotly.express as px
import plotly.graph_objects as go

class QueryResultVisualizer:
    """
    Generate visual representations of SQL query results.
    Supports Plotly interactive charts for web embedding.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        default_palette: Optional[List[str]] = None,
        max_points: int = 1000,
        max_categories: int = 20,
    ):
        """
        Args:
            logger: Optional logger for process logging.
            default_palette: Optional color palette for charts.
            max_points: Max points to plot before sampling.
            max_categories: Max categories for bar/pie before grouping 'Other'.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.default_palette = default_palette or px.colors.qualitative.Plotly
        self.max_points = max_points
        self.max_categories = max_categories

    def visualize(
        self,
        df: pd.DataFrame,
        chart_type: Optional[str] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        color: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        tooltips: Optional[List[str]] = None,
        layout_kwargs: Optional[Dict[str, Any]] = None,
        paginate: bool = False,
        page: int = 1,
        page_size: int = 100,
    ) -> go.Figure:
        """
        Main entrypoint: returns a Plotly Figure object.

        Args:
            df: DataFrame of query results.
            chart_type: Override chart type ('bar', 'line', 'pie', 'scatter', 'table').
            x, y, color: Column names for axes/colors.
            labels: Dict for axis/legend labels.
            title: Chart title.
            tooltips: List of columns to show as tooltips.
            layout_kwargs: Additional Plotly layout settings.
            paginate: If True, return only a page of data.
            page: Page number (1-based).
            page_size: Rows per page for pagination.

        Returns:
            Plotly Figure.
        """
        self.logger.info(f"Preparing to visualize {len(df)} rows, chart_type={chart_type}")

        # Sample or paginate for large datasets
        if paginate:
            df = self._paginate(df, page, page_size)
        elif len(df) > self.max_points:
            self.logger.info(f"Sampling {self.max_points} points out of {len(df)}")
            df = df.sample(n=self.max_points, random_state=42)

        # Auto chart type if not specified
        ctype = chart_type or self._infer_chart_type(df, x, y)

        # Choose columns for axes if not specified
        x, y = self._suggest_xy(df, x, y, ctype)

        # Labels and tooltips
        labels = labels or {}
        tooltips = tooltips or (df.columns.tolist() if ctype != "table" else None)

        # Generate the chart
        if ctype == "bar":
            fig = self._bar_chart(df, x, y, color, labels, title, tooltips)
        elif ctype == "line":
            fig = self._line_chart(df, x, y, color, labels, title, tooltips)
        elif ctype == "pie":
            fig = self._pie_chart(df, x, y, labels, title, tooltips)
        elif ctype == "scatter":
            fig = self._scatter_chart(df, x, y, color, labels, title, tooltips)
        elif ctype == "table":
            fig = self._table(df, title)
        else:
            raise ValueError(f"Unknown chart_type: {ctype}")

        # Layout
        layout_kwargs = layout_kwargs or {}
        fig.update_layout(
            title=title or "",
            colorway=self.default_palette,
            **layout_kwargs
        )
        return fig

    def _infer_chart_type(self, df: pd.DataFrame, x: Optional[str], y: Optional[str]) -> str:
        """
        Infer an appropriate chart type based on DataFrame structure.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Prefer table for many columns
        if len(df.columns) > 6:
            return "table"
        # Pie: one categorical, one numeric, few categories
        if len(categorical_cols) == 1 and len(numeric_cols) == 1:
            col = categorical_cols[0]
            if df[col].nunique() <= self.max_categories:
                return "pie"
        # Bar: categorical + numeric
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            return "bar"
        # Line: time series or 2+ numeric
        if len(numeric_cols) >= 2:
            return "line"
        # Scatter: 2+ numeric
        if len(numeric_cols) >= 2:
            return "scatter"
        return "table"

    def _suggest_xy(self, df: pd.DataFrame, x: Optional[str], y: Optional[str], chart_type: str) -> (str, str):
        """
        Suggest default x, y columns based on chart type and DataFrame.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if not x:
            if chart_type in ["bar", "pie"]:
                x = categorical_cols[0] if categorical_cols else df.columns[0]
            elif chart_type in ["line", "scatter"]:
                x = numeric_cols[0] if numeric_cols else df.columns[0]
            else:
                x = df.columns[0]
        if not y:
            if chart_type in ["bar", "pie"]:
                y = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            elif chart_type in ["line", "scatter"]:
                y = numeric_cols[1] if len(numeric_cols) > 1 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            else:
                y = None
        return x, y

    def _bar_chart(
        self, df, x, y, color, labels, title, tooltips
    ) -> go.Figure:
        # Group rare categories if too many
        if df[x].nunique() > self.max_categories:
            top_cats = df[x].value_counts().nlargest(self.max_categories).index
            df = df[df[x].isin(top_cats)].copy()
            df[x] = df[x].where(df[x].isin(top_cats), "Other")
        fig = px.bar(
            df, x=x, y=y, color=color,
            labels=labels, title=title, hover_data=tooltips,
            color_discrete_sequence=self.default_palette
        )
        return fig

    def _line_chart(
        self, df, x, y, color, labels, title, tooltips
    ) -> go.Figure:
        fig = px.line(
            df, x=x, y=y, color=color,
            labels=labels, title=title, hover_data=tooltips,
            color_discrete_sequence=self.default_palette
        )
        return fig

    def _pie_chart(
        self, df, x, y, labels, title, tooltips
    ) -> go.Figure:
        # Pie: x=categorical (labels), y=numeric (values)
        if df[x].nunique() > self.max_categories:
            top_cats = df[x].value_counts().nlargest(self.max_categories).index
            df = df[df[x].isin(top_cats)].copy()
            df[x] = df[x].where(df[x].isin(top_cats), "Other")
        fig = px.pie(
            df, names=x, values=y,
            labels=labels, title=title, hover_data=tooltips,
            color_discrete_sequence=self.default_palette
        )
        return fig

    def _scatter_chart(
        self, df, x, y, color, labels, title, tooltips
    ) -> go.Figure:
        fig = px.scatter(
            df, x=x, y=y, color=color,
            labels=labels, title=title, hover_data=tooltips,
            color_discrete_sequence=self.default_palette
        )
        return fig

    def _table(self, df, title) -> go.Figure:
        # Show first max_points rows
        display_df = df.head(self.max_points)
        header = list(display_df.columns)
        values = [display_df[col].tolist() for col in header]
        fig = go.Figure(
            data=[go.Table(
                header=dict(values=header, fill_color='paleturquoise', align='left'),
                cells=dict(values=values, fill_color='lavender', align='left')
            )]
        )
        fig.update_layout(title=title or "Query Results Table")
        return fig

    def _paginate(self, df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
        """
        Return a DataFrame page.
        """
        start = (page - 1) * page_size
        end = start + page_size
        self.logger.info(f"Paginating rows {start} to {end}")
        return df.iloc[start:end]

# Example usage:
# import plotly.io as pio
# vis = QueryResultVisualizer()
# fig = vis.visualize(df, chart_type="bar", x="country", y="sales", title="Sales by Country")
# pio.show(fig)