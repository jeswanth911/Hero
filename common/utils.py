import re
import logging
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union, Callable
from datetime import datetime, date, timezone, timedelta
import pytz
import unicodedata
import uuid

T = TypeVar('T')

# =========================
# Date/Time Parsing Utilities
# =========================

COMMON_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%m/%d/%y",
    "%Y.%m.%d",
    "%d-%m-%Y",
]

def parse_datetime(
    value: Union[str, int, float, datetime, date],
    tz: Optional[Union[str, timezone]] = None,
    formats: Optional[List[str]] = None,
    strict: bool = False,
) -> Optional[datetime]:
    """
    Parse a value into a timezone-aware datetime object.
    Supports multiple formats and Unix timestamps.

    Args:
        value: String, int, float, datetime, or date to parse.
        tz: Target timezone (name or tzinfo); defaults to UTC if None.
        formats: List of datetime formats to try; defaults to COMMON_DATE_FORMATS.
        strict: If True, raise error on failure; else return None.

    Returns:
        Timezone-aware datetime if parseable, else None or raises if strict.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            if strict:
                raise ValueError(f"Invalid timestamp: {value}")
            return None
    elif isinstance(value, str):
        dt = None
        fmts = formats or COMMON_DATE_FORMATS
        for fmt in fmts:
            try:
                dt = datetime.strptime(value.strip(), fmt)
                break
            except Exception:
                continue
        if dt is None:
            # Try ISO8601
            try:
                dt = datetime.fromisoformat(value.strip())
            except Exception:
                if strict:
                    raise ValueError(f"Could not parse date: {value}")
                return None
    else:
        if strict:
            raise TypeError(f"Unsupported type for parse_datetime: {type(value)}")
        return None

    # Set timezone
    if dt.tzinfo is None:
        if tz:
            if isinstance(tz, str):
                try:
                    tzinfo = pytz.timezone(tz)
                except Exception:
                    if strict:
                        raise
                    tzinfo = pytz.UTC
            else:
                tzinfo = tz
        else:
            tzinfo = pytz.UTC
        dt = dt.replace(tzinfo=tzinfo)
    return dt

# =========================
# Data Sanitization Utilities
# =========================

def normalize_string(val: str, lower: bool = True, strip: bool = True, ascii_only: bool = False) -> str:
    """
    Normalize a string: Unicode normalize, optional lower, strip, ascii.

    Args:
        val: Input string.
        lower: Convert to lowercase.
        strip: Strip whitespace.
        ascii_only: Convert non-ascii to closest ascii.

    Returns:
        Normalized string.
    """
    if not isinstance(val, str):
        val = str(val)
    val = unicodedata.normalize("NFKC", val)
    if strip:
        val = val.strip()
    if lower:
        val = val.lower()
    if ascii_only:
        val = val.encode("ascii", "ignore").decode()
    return val

def trim_whitespace(val: str) -> str:
    """
    Trim leading and trailing whitespace, and collapse internal whitespace.
    """
    return re.sub(r"\s+", " ", val.strip())

def prevent_injection(val: str) -> str:
    """
    Very basic SQL/script injection prevention.
    Escapes single quotes and strips dangerous characters.
    """
    if not isinstance(val, str):
        val = str(val)
    # Remove dangerous patterns (expand in real usage)
    val = re.sub(r"(--|;|/\*|\*/|xp_|\bDROP\b|\bTABLE\b)", "", val, flags=re.IGNORECASE)
    val = val.replace("'", "''")
    return val

def sanitize_dict(d: Dict[str, Any], sanitizer: Callable[[Any], Any] = normalize_string) -> Dict[str, Any]:
    """
    Apply sanitizer to all string values in dictionary.
    """
    return {k: sanitizer(v) if isinstance(v, str) else v for k, v in d.items()}

# =========================
# Type Conversion Utilities
# =========================

def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default

def to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1", "y"}
    try:
        return bool(int(val))
    except Exception:
        return default

# =========================
# Safe Dictionary Access
# =========================

def get_safe(d: Dict[str, Any], keys: Union[str, Sequence[str]], default: Any = None) -> Any:
    """
    Safely access a dictionary by a single key or sequence of keys (path).
    Returns default if key/path does not exist.
    """
    if isinstance(keys, str):
        return d.get(keys, default)
    val = d
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

# =========================
# Deep Merge Dicts
# =========================

def deep_merge(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merge dict b into dict a. Returns new dict.
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# =========================
# Logging Context Enrichment
# =========================

def enrich_log_context(
    logger: logging.Logger,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra
) -> logging.LoggerAdapter:
    """
    Returns a LoggerAdapter with contextual info for structured logging.
    """
    context = {"request_id": request_id, "correlation_id": correlation_id, "user_id": user_id}
    context.update(extra)
    return logging.LoggerAdapter(logger, {k: v for k, v in context.items() if v is not None})

def generate_request_id() -> str:
    """Generate a unique request ID (UUID4)."""
    return str(uuid.uuid4())

def generate_correlation_id() -> str:
    """Generate a unique correlation ID (UUID4)."""
    return str(uuid.uuid4())

# =========================
# General Purpose Utilities
# =========================

def is_empty(val: Any) -> bool:
    """Returns True if value is None or empty."""
    return val is None or (hasattr(val, '__len__') and len(val) == 0)

def chunk_list(lst: List[T], size: int) -> List[List[T]]:
    """Chunk a list into sublists of given size."""
    return [lst[i:i+size] for i in range(0, len(lst), size)]

# =========================
# __all__ Export
# =========================

__all__ = [
    "parse_datetime",
    "normalize_string",
    "trim_whitespace",
    "prevent_injection",
    "sanitize_dict",
    "safe_int",
    "safe_float",
    "to_bool",
    "get_safe",
    "deep_merge",
    "enrich_log_context",
    "generate_request_id",
    "generate_correlation_id",
    "is_empty",
    "chunk_list",
]
