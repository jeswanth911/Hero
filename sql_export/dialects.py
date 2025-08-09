from typing import Dict, Any, List, Optional, Type

class SQLDialect:
    """
    Base class for SQL dialect-specific logic.
    """

    name: str = "standard"
    type_map: Dict[str, str] = {
        "INTEGER": "INTEGER",
        "FLOAT": "FLOAT",
        "TEXT": "TEXT",
        "DATE": "DATE",
        "BOOLEAN": "BOOLEAN"
    }

    def quote_identifier(self, ident: str) -> str:
        """
        Return the SQL-quoted version of an identifier.
        """
        return f'"{ident}"'

    def map_type(self, generic_type: str) -> str:
        """
        Map a generic SQL type to the dialect's specific SQL type.
        """
        return self.type_map.get(generic_type.upper(), generic_type.upper())

    def if_not_exists(self) -> str:
        """
        Return the IF NOT EXISTS clause for CREATE TABLE, or "" if unsupported.
        """
        return "IF NOT EXISTS "

    def null_str(self) -> str:
        """
        Return the standard NULL string for the dialect.
        """
        return "NULL"

    def bulk_insert_optimization(self) -> Optional[str]:
        """
        Return the name of the optimized bulk insert method, if any.
        """
        return None

    def supports_tablespace(self) -> bool:
        """
        Does the dialect support TABLESPACE in CREATE TABLE?
        """
        return False

    def autoincrement_keyword(self) -> Optional[str]:
        """
        Keyword for autoincrementing primary keys, if any.
        """
        return None

    def __str__(self):
        return self.name


class PostgreSQLDialect(SQLDialect):
    name = "postgresql"
    type_map = {
        "INTEGER": "INTEGER",
        "FLOAT": "DOUBLE PRECISION",
        "TEXT": "TEXT",
        "DATE": "DATE",
        "BOOLEAN": "BOOLEAN"
    }

    def quote_identifier(self, ident: str) -> str:
        return f'"{ident}"'

    def map_type(self, generic_type: str) -> str:
        # SERIAL for autoincrement can be handled at DDL level
        return self.type_map.get(generic_type.upper(), generic_type.upper())

    def if_not_exists(self) -> str:
        return "IF NOT EXISTS "

    def bulk_insert_optimization(self) -> Optional[str]:
        return "COPY"

    def supports_tablespace(self) -> bool:
        return True

    def autoincrement_keyword(self) -> Optional[str]:
        # SERIAL is handled as a column type, not a keyword
        return None


class MySQLDialect(SQLDialect):
    name = "mysql"
    type_map = {
        "INTEGER": "INT",
        "FLOAT": "DOUBLE",
        "TEXT": "TEXT",
        "DATE": "DATETIME",
        "BOOLEAN": "TINYINT(1)"
    }

    def quote_identifier(self, ident: str) -> str:
        return f'`{ident}`'

    def map_type(self, generic_type: str) -> str:
        return self.type_map.get(generic_type.upper(), generic_type.upper())

    def if_not_exists(self) -> str:
        return "IF NOT EXISTS "

    def bulk_insert_optimization(self) -> Optional[str]:
        return "LOAD DATA"

    def supports_tablespace(self) -> bool:
        return True

    def autoincrement_keyword(self) -> Optional[str]:
        return "AUTO_INCREMENT"


class SQLiteDialect(SQLDialect):
    name = "sqlite"
    type_map = {
        "INTEGER": "INTEGER",
        "FLOAT": "REAL",
        "TEXT": "TEXT",
        "DATE": "TEXT",
        "BOOLEAN": "INTEGER"
    }

    def quote_identifier(self, ident: str) -> str:
        return f'"{ident}"'

    def map_type(self, generic_type: str) -> str:
        return self.type_map.get(generic_type.upper(), generic_type.upper())

    def if_not_exists(self) -> str:
        return "IF NOT EXISTS "

    def bulk_insert_optimization(self) -> Optional[str]:
        return None  # No special optimization

    def supports_tablespace(self) -> bool:
        return False

    def autoincrement_keyword(self) -> Optional[str]:
        return "AUTOINCREMENT"

# Registry for easy extension
DIALECT_REGISTRY: Dict[str, Type[SQLDialect]] = {
    "postgresql": PostgreSQLDialect,
    "mysql": MySQLDialect,
    "sqlite": SQLiteDialect,
    "standard": SQLDialect,
}

def get_dialect(name: str) -> SQLDialect:
    """
    Factory to get a dialect instance by name.
    """
    name = name.lower()
    cls = DIALECT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown SQL dialect: {name}")
    return cls()

# Example usage:
# dialect = get_dialect("postgresql")
# quoted = dialect.quote_identifier("my_table")
# sql_type = dialect.map_type("BOOLEAN")
