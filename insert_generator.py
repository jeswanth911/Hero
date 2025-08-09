import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, Generator

class InsertGenerator:
    """
    Generates efficient SQL INSERT statements from Pandas DataFrames.

    Features:
    - Batch insert for large datasets
    - Handles NULLs, special characters, and type conversion
    - Supports PostgreSQL (COPY), MySQL (LOAD DATA), SQLite, and standard SQL
    - Parameterized query support for prepared statements
    - Logging and error handling
    """

    DIALECTS = ("postgresql", "mysql", "sqlite", "standard")
    DEFAULT_BATCH_SIZE = 1000

    def __init__(
        self,
        dialect: str = "postgresql",
        logger: Optional[logging.Logger] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Args:
            dialect: Target SQL dialect ('postgresql', 'mysql', 'sqlite', 'standard')
            logger: Logger instance
            batch_size: Number of rows per INSERT
        """
        self.dialect = dialect.lower()
        if self.dialect not in self.DIALECTS:
            raise ValueError(f"Unsupported dialect: {dialect}. Supported: {self.DIALECTS}")
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size

    def _escape_value(self, val: Any) -> str:
        """Escapes and formats a value for SQL."""
        if pd.isnull(val):
            return "NULL"
        if isinstance(val, (int, float, np.integer, np.floating)):
            return str(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return f"'{str(val)}'"
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        # String types
        s = str(val)
        s = s.replace("'", "''")  # Escape single quotes
        if self.dialect == "mysql":
            s = s.replace("\\", "\\\\")  # Escape backslashes for MySQL
        return f"'{s}'"

    def _row_to_values(self, row: pd.Series) -> List[str]:
        """Converts a DataFrame row to a list of SQL-safe values."""
        return [self._escape_value(val) for val in row]

    def _row_to_placeholders(self, row: pd.Series) -> List[str]:
        """
        Returns a list of %s placeholders for parameterized queries.
        """
        return ["%s"] * len(row)

    def _dialect_bulk_insert(
        self,
        df: pd.DataFrame,
        table: str,
        columns: Optional[List[str]] = None,
        file_path: Optional[str] = None
    ) -> str:
        """
        Generates dialect-specific optimized bulk insert (COPY, LOAD DATA).
        """
        if self.dialect == "postgresql":
            # COPY FROM STDIN (assume CSV format)
            cols = columns or list(df.columns)
            csv_data = df.to_csv(index=False, header=False, na_rep='\\N')
            statement = f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N');\n"
            statement += csv_data
            return statement
        elif self.dialect == "mysql":
            # LOAD DATA INFILE (needs a CSV file)
            if not file_path:
                raise ValueError("file_path required for MySQL LOAD DATA INFILE")
            cols = columns or list(df.columns)
            statement = (
                f"LOAD DATA INFILE '{file_path}' INTO TABLE {table} "
                f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
                f"LINES TERMINATED BY '\\n' "
                f"({', '.join(cols)});"
            )
            return statement
        else:
            raise NotImplementedError("Bulk insert only supported for PostgreSQL (COPY) and MySQL (LOAD DATA)")

    def generate_insert_statements(
        self,
        df: pd.DataFrame,
        table: str,
        columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        parameterized: bool = False
    ) -> Generator[Tuple[str, Optional[List[Tuple[Any, ...]]]], None, None]:
        """
        Yields batched INSERT statements (optionally parameterized).

        Args:
            df: DataFrame to insert
            table: SQL table name
            columns: List of columns to insert (default: all columns)
            batch_size: Number of rows per statement
            parameterized: If True, generates parameterized query with parameters

        Yields:
            Tuple of (sql statement, params for parameterized) for each batch
        """
        batch_size = batch_size or self.batch_size
        cols = columns or list(df.columns)
        n_rows = len(df)

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch = df.iloc[start:end]
            if parameterized:
                # Prepared statement: INSERT ... VALUES (%s, %s, ...)
                placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
                values_clause = ", ".join([placeholders] * len(batch))
                sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES {values_clause};"
                params: List[Tuple[Any, ...]] = [tuple(row) for row in batch[cols].values]
                # Flatten params for executemany-like APIs
                params_flat = [item for sublist in params for item in sublist]
                yield sql, params_flat
            else:
                # Inline values: INSERT ... VALUES (v1, v2, ...), ...
                value_rows = []
                for _, row in batch.iterrows():
                    vals = self._row_to_values(row[cols])
                    value_rows.append("(" + ", ".join(vals) + ")")
                sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES\n  " + ",\n  ".join(value_rows) + ";"
                yield sql, None

    def generate(
        self,
        df: pd.DataFrame,
        table: str,
        columns: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        bulk: bool = False,
        parameterized: bool = False
    ) -> Union[str, Generator[Tuple[str, Optional[List[Tuple[Any, ...]]]], None, None]]:
        """
        Main entry point for generating INSERT scripts.

        Args:
            df: DataFrame to convert
            table: SQL table name
            columns: List of columns (default: all)
            file_path: Path to use for bulk (LOAD DATA) if needed
            bulk: Use bulk insert optimization (COPY/LOAD DATA)
            parameterized: Generate parameterized statements

        Returns:
            String (for bulk) or generator of (sql, params) tuples for batched inserts
        """
        try:
            if bulk and self.dialect in ("postgresql", "mysql"):
                return self._dialect_bulk_insert(df, table, columns, file_path)
            else:
                return self.generate_insert_statements(df, table, columns, parameterized=parameterized)
        except Exception as e:
            self.logger.error(f"Insert statement generation failed: {e}")
            raise

# Example usage:
# logger = logging.getLogger("insert_generator")
# gen = InsertGenerator(dialect="postgresql", logger=logger)
# for sql, params in gen.generate(df, "my_table"):
#     print(sql)
#     # Optionally pass params to DB API if parameterized
#
# For bulk COPY (Postgres):
# script = gen.generate(df, "my_table", bulk=True)
# print(script)