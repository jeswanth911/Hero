"""
exceptions.py

Shared exception classes for the data platform application.

Usage:
    from exceptions import IngestionError, CleaningError, ValidationError, SQLGenerationError, APIError

    try:
        ...
    except IngestionError as e:
        log.error(e.serialize())
        return JSONResponse(e.serialize(), status_code=400)
"""

from typing import Any, Dict, Optional, Union
from common.constants import ErrorCode

class DataPipelineError(Exception):
    """
    Base exception for all application errors.

    Args:
        message: Human-readable error message.
        code: Standardized error code (see constants.ErrorCode).
        context: Optional dict with additional context for debugging.
    """

    def __init__(
        self,
        message: str = "An error occurred in the data pipeline.",
        code: Union[int, "ErrorCode"] = ErrorCode.UNKNOWN_ERROR.value,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code.value if hasattr(code, "value") else code
        self.context = context or {}

    def __str__(self):
        return f"[Error {self.code}] {self.message}"

    def serialize(self) -> Dict[str, Any]:
        """Return a dict for API responses (safe for JSON serialization)."""
        data = {
            "error": {
                "message": self.message,
                "code": self.code,
            }
        }
        if self.context:
            data["error"]["context"] = self.context
        return data

# --- Granular Exception Classes ---

class IngestionError(DataPipelineError):
    """Raised for data ingestion errors."""
    def __init__(self, message="Ingestion failed.", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, context)

class CleaningError(DataPipelineError):
    """Raised for data cleaning/processing errors."""
    def __init__(self, message="Cleaning failed.", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, context)

class ValidationError(DataPipelineError):
    """Raised for schema/data validation failures."""
    def __init__(self, message="Validation failed.", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, context)

class SQLGenerationError(DataPipelineError):
    """Raised for SQL/DDL generation errors."""
    def __init__(self, message="SQL generation failed.", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.SQL_EXEC_ERROR, context)

class APIError(DataPipelineError):
    """Raised for API endpoint errors (bad request, forbidden, etc)."""
    def __init__(
        self,
        message="API error.",
        code: Union[int, "ErrorCode"] = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, context)

# --- Usage Patterns and Best Practices ---
"""
- Always raise a specific exception type (not just DataPipelineError) for clarity.
- Use the 'context' attribute for debug info (never include secrets).
- In API code, catch DataPipelineError and return .serialize() in the response body.
- You can extend exceptions for new modules by inheriting from DataPipelineError.

Example:
    try:
        ingest_data(file)
    except IngestionError as e:
        logger.error(e)
        return JSONResponse(e.serialize(), status_code=400)
"""
