import logging
from typing import Dict, Any, List, Optional, Tuple

class DDLGenerator:
    """
    Generates SQL CREATE TABLE statements for various SQL dialects
    from schema definitions (e.g., from schema_inference.py).
    """

    DIALECTS = ('postgresql', 'mysql', 'sqlite')

    def __init__(
        self,
        dialect: str = "postgresql",
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            dialect: Target SQL dialect ('postgresql', 'mysql', 'sqlite').
            logger: Optional logger for messages.
        """
        self.dialect = dialect.lower()
        if self.dialect not in self.DIALECTS:
            raise ValueError(f"Unsupported dialect '{dialect}'. Supported: {self.DIALECTS}")
        self.logger = logger or logging.getLogger(__name__)

    def generate_create_table(
        self,
        schema: Dict[str, Any],
        if_not_exists: bool = True,
        tablespace: Optional[str] = None,
        indexes: Optional[List[Dict[str, Any]]] = None,
        foreign_keys: Optional[List[Dict[str, Any]]] = None,
        unique_constraints: Optional[List[List[str]]] = None,
    ) -> str:
        """
        Generate a CREATE TABLE statement.

        Args:
            schema: Schema dictionary (from schema_inference.py).
            if_not_exists: Add IF NOT EXISTS to CREATE TABLE.
            tablespace: Optional tablespace for PostgreSQL/MySQL.
            indexes: List of index definitions.
            foreign_keys: List of foreign key definitions.
            unique_constraints: List of unique constraints (each is a list of column names).
        Returns:
            SQL DDL CREATE TABLE statement.
        """
        table = schema["table_name"]
        cols = schema["columns"]
        pks = schema.get("primary_keys", [])

        self.logger.info(f"Generating CREATE TABLE for '{table}' ({self.dialect})")

        col_defs = []
        for col in cols:
            col_sql = self._column_definition(col)
            col_defs.append(col_sql)

        constraint_defs = []

        # Primary Key
        if pks and not any(col.get("primary_key", False) for col in cols):
            constraint_defs.append(self._primary_key_constraint(pks))

        # Unique constraints
        if unique_constraints:
            for uq_cols in unique_constraints:
                constraint_defs.append(self._unique_constraint(uq_cols))

        # Foreign keys
        if foreign_keys:
            for fk in foreign_keys:
                constraint_defs.append(self._foreign_key_constraint(fk))

        # Assemble all column and constraint definitions
        all_defs = col_defs + constraint_defs

        # CREATE TABLE line
        if_clause = "IF NOT EXISTS " if if_not_exists and self.dialect != "sqlite" else ""
        create_stmt = f"CREATE TABLE {if_clause}{self._quote_ident(table)} (\n  "
        create_stmt += ",\n  ".join(all_defs)
        create_stmt += "\n)"

        # Tablespace
        if tablespace and self.dialect in ("postgresql", "mysql"):
            if self.dialect == "postgresql":
                create_stmt += f" TABLESPACE {tablespace}"
            elif self.dialect == "mysql":
                create_stmt += f" TABLESPACE `{tablespace}`"

        create_stmt += ";"

        # Indexes
        index_stmts = []
        if indexes:
            for idx in indexes:
                index_stmts.append(self._index_statement(table, idx))

        script = create_stmt
        if index_stmts:
            script += "\n" + "\n".join(index_stmts)

        return script

    def _quote_ident(self, ident: str) -> str:
        if self.dialect == "postgresql":
            return f'"{ident}"'
        elif self.dialect == "mysql":
            return f'`{ident}`'
        elif self.dialect == "sqlite":
            return f'"{ident}"'
        return ident

    def _column_definition(self, col: Dict[str, Any]) -> str:
        """
        Build a column definition SQL string for the target dialect.
        """
        name = self._quote_ident(col["name"])
        sql_type = self._map_type(col["type"])
        col_def = f"{name} {sql_type}"

        # NOT NULL
        if not col.get("nullable", True):
            col_def += " NOT NULL"

        # Primary Key (inline, if single column and dialect supports)
        if col.get("primary_key", False):
            if self.dialect == "sqlite":
                col_def += " PRIMARY KEY"
            elif self.dialect == "mysql" and sql_type.lower() == "integer":
                col_def += " AUTO_INCREMENT PRIMARY KEY"
            elif self.dialect == "postgresql" and sql_type.lower() in ("integer", "bigint"):
                # For SERIAL, but prefer explicit for enterprise
                col_def += " PRIMARY KEY"
        return col_def

    def _primary_key_constraint(self, pk_cols: List[str]) -> str:
        """
        Multi-column PK constraint.
        """
        cols = ", ".join(self._quote_ident(c) for c in pk_cols)
        return f"PRIMARY KEY ({cols})"

    def _unique_constraint(self, uq_cols: List[str]) -> str:
        cols = ", ".join(self._quote_ident(c) for c in uq_cols)
        return f"UNIQUE ({cols})"

    def _foreign_key_constraint(self, fk: Dict[str, Any]) -> str:
        """
        fk: Dict with keys:
            columns: List[str]
            ref_table: str
            ref_columns: List[str]
            on_delete: Optional[str]
            on_update: Optional[str]
        """
        cols = ", ".join(self._quote_ident(c) for c in fk["columns"])
        ref_cols = ", ".join(self._quote_ident(c) for c in fk["ref_columns"])
        stmt = f"FOREIGN KEY ({cols}) REFERENCES {self._quote_ident(fk['ref_table'])} ({ref_cols})"
        if fk.get("on_delete"):
            stmt += f" ON DELETE {fk['on_delete']}"
        if fk.get("on_update"):
            stmt += f" ON UPDATE {fk['on_update']}"
        return stmt

    def _index_statement(self, table: str, idx: Dict[str, Any]) -> str:
        """
        idx: Dict with keys:
            name: str
            columns: List[str]
            unique: Optional[bool]
            tablespace: Optional[str]
        """
        name = idx.get("name") or f"{table}_{'_'.join(idx['columns'])}_idx"
        unique = "UNIQUE " if idx.get("unique") else ""
        cols = ", ".join(self._quote_ident(c) for c in idx["columns"])
        stmt = f"CREATE {unique}INDEX {self._quote_ident(name)} ON {self._quote_ident(table)} ({cols})"
        if idx.get("tablespace") and self.dialect in ("postgresql", "mysql"):
            if self.dialect == "postgresql":
                stmt += f" TABLESPACE {idx['tablespace']}"
            elif self.dialect == "mysql":
                stmt += f" TABLESPACE `{idx['tablespace']}`"
        stmt += ";"
        return stmt

    def _map_type(self, sql_type: str) -> str:
        """
        Map generic SQL types to dialect-specific types.
        """
        t = sql_type.upper()

        # SQLite: no strict typing
        if self.dialect == "sqlite":
            if t in ("INTEGER", "BIGINT"):
                return "INTEGER"
            elif t == "FLOAT":
                return "REAL"
            elif t == "BOOLEAN":
                return "INTEGER"
            elif t == "DATE":
                return "TEXT"
            elif t == "TEXT":
                return "TEXT"
            else:
                return t

        # MySQL
        if self.dialect == "mysql":
            if t == "BOOLEAN":
                return "TINYINT(1)"
            elif t == "DATE":
                return "DATETIME"
            else:
                return t

        # PostgreSQL: default, just return
        return t

# Example usage:
# from schema_inference import infer_schema
# schema = infer_schema(df, table_name="users")
# ddlgen = DDLGenerator(dialect="postgresql")
# sql = ddlgen.generate_create_table(schema, if_not_exists=True)
# print(sql)