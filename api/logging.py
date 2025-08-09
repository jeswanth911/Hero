import logging
import sys
import json
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None

# ========== JSON Formatter ==========

class JsonLogFormatter(logging.Formatter):
    """
    Custom log formatter that outputs logs as JSON strings.
    Suitable for ingestion by ELK stack and similar systems.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Add optional contextual info
        for attr in ("request_id", "user_id", "workflow_id", "event_type"):
            val = getattr(record, attr, None)
            if val is not None:
                log_record[attr] = val
        # Extra fields from user
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_record.update(record.extra_fields)
        # Remove known sensitive fields if present
        for sensitive in ["password", "token", "secret", "api_key"]:
            log_record.pop(sensitive, None)
        return json.dumps(log_record)

# ========== Logger Setup ==========

def setup_logging(
    log_level: str = "INFO",
    sentry_dsn: Optional[str] = None,
    enable_console: bool = True,
    enable_sentry: bool = False,
    service_name: Optional[str] = None,
) -> None:
    """
    Set up structured (JSON) logging and integrate with Sentry if requested.
    """
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Remove default handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console handler with JSON formatter
    if enable_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        root.addHandler(handler)

    # Optionally integrate with Sentry for error/exception monitoring
    if enable_sentry and sentry_dsn and sentry_sdk is not None:
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=1.0,
            environment=service_name or "default"
        )
        root.info("Sentry integration enabled.")

# ========== Logging Utilities ==========

def get_logger(name: str = "app") -> logging.Logger:
    """
    Get a logger instance for the given name.
    """
    return logging.getLogger(name)

def log_api_call(
    logger: logging.Logger,
    path: str,
    method: str,
    status_code: int,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an API call event.
    """
    logger.info(
        f"API {method} {path} status={status_code}",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "event_type": "api_call",
            "extra_fields": extra or {},
        }
    )

def log_error(
    logger: logging.Logger,
    err: Exception,
    context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """
    Log an error event, with context and optional Sentry integration.
    """
    logger.error(
        f"Error: {str(err)}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "event_type": "error",
            "extra_fields": context or {},
        }
    )
    # Optionally: send to Sentry (integration happens via setup_logging)

def log_workflow_event(
    logger: logging.Logger,
    workflow_id: str,
    stage: str,
    status: str,
    message: Optional[str] = None,
    user_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log workflow-related events (e.g., start, progress, complete, fail).
    """
    logger.info(
        f"Workflow {workflow_id} - {stage} - {status}: {message or ''}",
        extra={
            "workflow_id": workflow_id,
            "user_id": user_id,
            "event_type": "workflow_event",
            "extra_fields": extra or {},
        }
    )

# ========== Prometheus Metrics Stub ==========

# Real Prometheus integration should use prometheus_client library.
# Here is a stub for metrics increment.

def increment_metric(metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
    """
    Stub for Prometheus metric increment.
    Integrate with prometheus_client.Counter in production.
    """
    # Example: prometheus_counter.labels(**labels).inc(value)
    pass

# ========== Request ID Utility ==========

def generate_request_id() -> str:
    """Generate a unique request ID (UUID4)."""
    return str(uuid.uuid4())

# ========== Example Usage ==========

# setup_logging(log_level="DEBUG", sentry_dsn=None, service_name="data-platform")
# logger = get_logger("data-platform")
# log_api_call(logger, "/api/data", "POST", 200, request_id=generate_request_id(), user_id="user123")
# try:
#     x = 1 / 0
# except Exception as e:
#     log_error(logger, e)
