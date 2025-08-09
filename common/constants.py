from enum import Enum, unique, auto
from typing import Final, Dict

# ===============================
# Database Table Names & Columns
# ===============================

#: Name of the user table in the main database.
USER_TABLE_NAME: Final[str] = "users"
#: Name of the audit log table.
AUDIT_LOG_TABLE_NAME: Final[str] = "audit_logs"
#: Name of the data ingestion table.
INGESTION_TABLE_NAME: Final[str] = "ingestion_records"
#: Name of the roles/permissions table.
ROLE_TABLE_NAME: Final[str] = "roles"

#: Standardized column name mappings for ETL/cleaning
COLUMN_MAPPINGS: Final[Dict[str, str]] = {
    "userId": "user_id",
    "UserID": "user_id",
    "createdAt": "created_at",
    "CreatedAt": "created_at",
    "emailAddress": "email",
    "Email": "email",
    "FullName": "full_name",
    "fullName": "full_name",
}

# ===============================
# Default Configuration Values
# ===============================

#: Default page size for paginated API responses.
DEFAULT_PAGE_SIZE: Final[int] = 100
#: Maximum allowed page size.
MAX_PAGE_SIZE: Final[int] = 1000
#: Default timezone for date/time operations.
DEFAULT_TIMEZONE: Final[str] = "UTC"
#: Default retry count for workflows.
DEFAULT_RETRY_COUNT: Final[int] = 3

# ===============================
# Enumerations
# ===============================

@unique
class LogLevel(Enum):
    """
    Enum for application log levels.
    Usage: logger.setLevel(LogLevel.INFO.value)
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@unique
class DataType(Enum):
    """
    Enum for supported data types in schema inference and validation.
    Usage: DataType.STRING, DataType.INTEGER, etc.
    """
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    JSON = "json"
    BINARY = "binary"

@unique
class ErrorCode(Enum):
    """
    Error codes used across the application for standardized error responses.
    Usage: ErrorCode.DB_CONN_ERROR, ErrorCode.AUTH_FAILED, etc.
    """
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    VALIDATION_ERROR = 100
    DB_CONN_ERROR = 200
    SQL_EXEC_ERROR = 201
    AUTH_FAILED = 300
    PERMISSION_DENIED = 301
    NOT_FOUND = 404
    RATE_LIMITED = 429
    SERVICE_UNAVAILABLE = 503

@unique
class UserRole(Enum):
    """
    Enum for user roles in the system.
    Usage: UserRole.ADMIN, UserRole.USER, etc.
    """
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    GUEST = "guest"

@unique
class Permission(Enum):
    """
    Enum for granular permissions.
    Usage: Permission.READ, Permission.WRITE, etc.
    """
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE_USERS = "manage_users"
    EXPORT = "export"
    EXECUTE = "execute"

# ===============================
# Miscellaneous Immutable Constants
# ===============================

#: List of all supported SQL dialects.
SUPPORTED_SQL_DIALECTS: Final[frozenset] = frozenset({"postgresql", "mysql", "sqlite"})

#: Application-wide identifier for request correlation.
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

# ===============================
# Usage Example (documentation only)
# ===============================
"""
# Example of using constants and enums:

from constants import USER_TABLE_NAME, LogLevel, ErrorCode, UserRole

print(USER_TABLE_NAME)  # "users"
logger.setLevel(LogLevel.INFO.value)
if errcode == ErrorCode.AUTH_FAILED:
    print("Authentication failed")
if current_user.role == UserRole.ADMIN:
    # grant admin permissions
"""
