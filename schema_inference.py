import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

# Default mapping from Pandas/numpy dtypes to SQL standard types
DEFAULT_TYPE_MAP = {
    "int64": "INTEGER",
    "int32": "INTEGER",
    "int16": "INTEGER",
    "int8": "INTEGER",
    "uint8": "INTEGER",
    "uint16": "INTEGER",
    "uint32": "INTEGER",
    "uint64": "INTEGER",
    "float64": "FLOAT",
    "float32": "FLOAT",
    "float16": "FLOAT",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "object": "TEXT",
    "string": "TEXT",
    "datetime64[ns]": "DATE",
    "datetime64[ns, UTC]": "DATE",
    "timedelta[ns]": "TEXT",
    "category": "TEXT",
}

def infer_sql_type(
    dtype: Union[str, np.dtype], 
    column: pd.Series, 
    custom_type_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Infer SQL data type from a pandas dtype and series content.
    Allows custom mapping override.
    """
    type_map = DEFAULT_TYPE_MAP.copy()
    if custom_type_map:
        type_map.update(custom_type_map)
    dtype_str = str(dtype)

    # Special case: check if object is all numbers, or all dates
    if dtype_str == "object":
        if pd.api.types.is_datetime64_any_dtype(column):
            return "DATE"
        elif pd.api.types.is_integer_dtype(column):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(column):
            return "FLOAT"
        # Try detecting boolean columns in object type
        unique_vals = set(column.dropna().unique())
        if unique_vals <= {0, 1, True, False}:
            return "BOOLEAN"
    return type_map.get(dtype_str, "TEXT")

def detect_primary_keys(
    df: pd.DataFrame,
    user_keys: Optional[List[str]] = None,
    max_cardinality: float = 0.98
) -> List[str]:
    """
    Detect likely primary keys.
    Args:
        df: DataFrame
        user_keys: User-specified primary keys (if any)
        max_cardinality: Minimum uniqueness ratio to consider a column as PK
    Returns:
        List of column names for primary keys
    """
    if user_keys:
        return user_keys
    pk_candidates = []
    n_rows = len(df)
    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        if n_unique == n_rows and not df[col].isnull().any():
            pk_candidates.append(col)
        # Optionally, allow near-unique if > threshold (for poor data)
        elif n_unique / n_rows >= max_cardinality and not df[col].isnull().any():
            pk_candidates.append(col)
    return pk_candidates

def infer_schema(
    df: pd.DataFrame,
    table_name: str = "my_table",
    type_overrides: Optional[Dict[str, str]] = None,
    key_overrides: Optional[List[str]] = None,
    custom_type_map: Optional[Dict[str, str]] = None,
    sample_size: int = 10000
) -> Dict[str, Any]:
    """
    Infer SQL table schema from a DataFrame.

    Args:
        df: DataFrame
        table_name: Name of the table
        type_overrides: Dict of column:type for forced typing
        key_overrides: List of primary key columns
        custom_type_map: Dict for custom pandas->SQL type mapping
        sample_size: Number of rows to sample for inference (for large sets)

    Returns:
        Dict with table_name, columns, primary_keys
    """
    # For big datasets, sample for speed
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    schema = {
        "table_name": table_name,
        "columns": [],
        "primary_keys": [],
    }

    # Detect PKs
    primary_keys = detect_primary_keys(df_sample, user_keys=key_overrides)
    schema["primary_keys"] = primary_keys

    for col in df.columns:
        # Type override
        col_type = (
            type_overrides.get(col)
            if type_overrides and col in type_overrides
            else infer_sql_type(df[col].dtype, df[col], custom_type_map)
        )
        nullable = df[col].isnull().any()
        col_def = {
            "name": col,
            "type": col_type,
            "nullable": nullable,
            "primary_key": col in primary_keys,
        }
        schema["columns"].append(col_def)
    return schema

def generate_column_definitions(schema: Dict[str, Any]) -> List[str]:
    """
    Generate SQL column definitions from schema dict.

    Args:
        schema: Output from infer_schema

    Returns:
        List of column definition strings
    """
    defs = []
    for col in schema["columns"]:
        col_str = f"{col['name']} {col['type']}"
        if not col["nullable"]:
            col_str += " NOT NULL"
        if col["primary_key"]:
            col_str += " PRIMARY KEY"
        defs.append(col_str)
    return defs

def generate_create_table_statement(schema: Dict[str, Any]) -> str:
    """
    Generate full CREATE TABLE statement.

    Args:
        schema: Output from infer_schema

    Returns:
        CREATE TABLE statement as string
    """
    col_defs = []
    for col in schema["columns"]:
        col_str = f"{col['name']} {col['type']}"
        if not col["nullable"]:
            col_str += " NOT NULL"
        col_defs.append(col_str)
    pk = schema["primary_keys"]
    pk_str = f", PRIMARY KEY ({', '.join(pk)})" if pk else ""
    return f"CREATE TABLE {schema['table_name']} (\n  " + ",\n  ".join(col_defs) + pk_str + "\n);"

# Example usage:
# df = pd.DataFrame(...)
# schema = infer_schema(df, table_name="users", type_overrides={"age": "INTEGER"}, key_overrides=["user_id"])
# create_stmt = generate_create_table_statement(schema)
# print(create_stmt)