import pytest
import pandas as pd
import numpy as np

# Mocked functions (replace with real imports in project)
# from sql_export import infer_sql_schema, generate_create_table, generate_bulk_insert

def infer_sql_schema(df: pd.DataFrame, dialect: str = "postgresql") -> dict:
    """Infer column types from a DataFrame for SQL dialect."""
    mapping = {
        "int64": "INTEGER",
        "float64": "FLOAT",
        "object": "TEXT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }
    if dialect == "mysql":
        mapping["bool"] = "TINYINT(1)"
        mapping["datetime64[ns]"] = "DATETIME"
    elif dialect == "sqlite":
        mapping["int64"] = "INTEGER"
        mapping["float64"] = "REAL"
        mapping["datetime64[ns]"] = "TEXT"
    return {col: mapping[str(dtype)] for col, dtype in df.dtypes.items()}

def generate_create_table(table_name: str, schema: dict, dialect: str = "postgresql") -> str:
    """Generate CREATE TABLE statement for a given schema."""
    cols = []
    for col, typ in schema.items():
        safe_col = f'"{col}"' if dialect in {"postgresql", "sqlite"} else f'`{col}`'
        cols.append(f"{safe_col} {typ}")
    if dialect == "postgresql":
        return f'CREATE TABLE "{table_name}" (\n  {",\n  ".join(cols)}\n);'
    elif dialect == "mysql":
        return f'CREATE TABLE `{table_name}` (\n  {",\n  ".join(cols)}\n);'
    else:
        return f'CREATE TABLE "{table_name}" (\n  {",\n  ".join(cols)}\n);'

def generate_bulk_insert(table_name: str, df: pd.DataFrame, batch_size: int = 2, dialect: str = "postgresql") -> list:
    """Generate bulk INSERT statements with batching."""
    stmts = []
    col_list = ', '.join([f'"{c}"' if dialect in {"postgresql", "sqlite"} else f'`{c}`' for c in df.columns])
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        values = []
        for _, row in batch.iterrows():
            vals = []
            for v in row:
                if pd.isnull(v):
                    vals.append("NULL")
                elif isinstance(v, str):
                    vals.append(f"'{v.replace(\"'\", \"''\")}'")
                else:
                    vals.append(str(v))
            values.append(f"({', '.join(vals)})")
        stmt = f'INSERT INTO {"`" if dialect == "mysql" else "\""}{table_name}{"`" if dialect == "mysql" else "\""} ({col_list}) VALUES ' + ', '.join(values) + ";"
        stmts.append(stmt)
    return stmts

# ---- Fixtures ----

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "O'Reilly"],
        "amount": [10.5, 20, np.nan],
        "is_active": [True, False, None],
        "created_at": pd.to_datetime(["2023-01-01", "2023-02-01", None])
    })

@pytest.fixture
def expected_schema_pg():
    return {
        "id": "INTEGER",
        "name": "TEXT",
        "amount": "FLOAT",
        "is_active": "BOOLEAN",
        "created_at": "TIMESTAMP"
    }

@pytest.fixture
def expected_schema_mysql():
    return {
        "id": "INTEGER",
        "name": "TEXT",
        "amount": "FLOAT",
        "is_active": "TINYINT(1)",
        "created_at": "DATETIME"
    }

@pytest.fixture
def expected_schema_sqlite():
    return {
        "id": "INTEGER",
        "name": "TEXT",
        "amount": "REAL",
        "is_active": "BOOLEAN",
        "created_at": "TEXT"
    }

# ---- SQL Type Mapping ----

@pytest.mark.parametrize("dialect,expected_fixture", [
    ("postgresql", "expected_schema_pg"),
    ("mysql", "expected_schema_mysql"),
    ("sqlite", "expected_schema_sqlite"),
])
def test_infer_sql_schema(sample_df, dialect, expected_fixture, request):
    expected = request.getfixturevalue(expected_fixture)
    schema = infer_sql_schema(sample_df, dialect)
    assert schema == expected

# ---- CREATE TABLE Generation ----

@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "sqlite"])
def test_generate_create_table(sample_df, dialect):
    table_name = "test_table"
    schema = infer_sql_schema(sample_df, dialect)
    ddl = generate_create_table(table_name, schema, dialect)
    # Check syntactic start/end and that all columns are present
    assert f"CREATE TABLE" in ddl
    for col in sample_df.columns:
        assert col in ddl
    # Should have appropriate quote chars
    if dialect == "mysql":
        assert "`id`" in ddl
    else:
        assert f'"id"' in ddl
    # Should end with );
    assert ddl.strip().endswith(");")

# ---- Bulk Insert Statement Generation ----

@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "sqlite"])
def test_generate_bulk_insert_statements(sample_df, dialect):
    stmts = generate_bulk_insert("test_table", sample_df, batch_size=2, dialect=dialect)
    # Should create 2 batches (3 rows, batch_size=2)
    assert len(stmts) == 2
    for i, stmt in enumerate(stmts):
        assert "INSERT INTO" in stmt
        assert "test_table" in stmt
        # Check string escaping (O'Reilly)
        assert "O''Reilly" in stmt
        # Check NULLs for missing values
        assert "NULL" in stmt
        # Check correct quote chars per dialect
        if dialect == "mysql":
            assert "`name`" in stmt
        else:
            assert '"name"' in stmt

# ---- Edge Cases ----

def test_special_characters_and_nulls():
    df = pd.DataFrame({
        "col1": ["a", "b", "c,d", "e\"f", "g'h"],
        "col2": [None, 2, 3, np.nan, 5]
    })
    stmts = generate_bulk_insert("mytable", df, batch_size=2)
    for stmt in stmts:
        assert "NULL" in stmt
        # Escaping for single quote
        assert "'g''h'" in stmt

def test_create_table_with_keywords():
    df = pd.DataFrame({
        "select": [1, 2],
        "from": ["a", "b"]
    })
    schema = infer_sql_schema(df)
    ddl = generate_create_table("keywords", schema)
    # Should quote keywords
    assert '"select"' in ddl
    assert '"from"' in ddl

def test_sql_syntax_validity(sample_df):
    schema = infer_sql_schema(sample_df)
    ddl = generate_create_table("syntest", schema)
    # Quick syntax check (not full SQL parse)
    assert ddl.count("(") == ddl.count(")")
    assert ddl.endswith(");")

def test_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    schema = infer_sql_schema(df)
    ddl = generate_create_table("empty_table", schema)
    assert "a" in ddl and "b" in ddl
    stmts = generate_bulk_insert("empty_table", df)
    assert stmts == []

# ---- Batched Inserts ----

def test_insert_batching(sample_df):
    stmts = generate_bulk_insert("batch_table", sample_df, batch_size=1)
    assert len(stmts) == len(sample_df)
    stmts2 = generate_bulk_insert("batch_table", sample_df, batch_size=5)
    assert len(stmts2) == 1