import os
from typing import Optional, Literal
from pydantic import BaseSettings, Field, ValidationError, validator
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AppConfig(BaseSettings):
    """
    Central application configuration loaded from environment variables,
    .env file, or defaults. Supports different environments.
    """

    # ====== General Application Settings ======
    APP_ENV: Literal["development", "staging", "production"] = Field(
        "development", description="Current environment"
    )
    DEBUG: bool = Field(False, description="Enable debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level (DEBUG, INFO, WARN, ERROR)")

    # ====== Database Settings ======
    DB_URL: str = Field(..., description="SQLAlchemy-compatible database URL (e.g., postgresql://...)")
    DB_POOL_SIZE: int = Field(5, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(10, description="Max overflow connections")

    # ====== Security & Auth ======
    SECRET_KEY: str = Field(..., description="Secret key for JWT and encryption")
    JWT_ALGORITHM: str = Field("HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60, description="JWT access token expiry (minutes)")
    REFRESH_TOKEN_EXPIRE_HOURS: int = Field(24, description="JWT refresh token expiry (hours)")

    # ====== API Keys & Secrets ======
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key for LLM features")
    SENTRY_DSN: Optional[str] = Field(None, description="Sentry DSN for error monitoring")
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis URL for caching/rate-limiting")

    # ====== External Providers ======
    LDAP_URL: Optional[str] = Field(None, description="LDAP server URL for enterprise auth")
    OAUTH_CLIENT_ID: Optional[str] = Field(None, description="OAuth client ID for external login")
    OAUTH_CLIENT_SECRET: Optional[str] = Field(None, description="OAuth client secret for external login")

    # ====== Other ======
    SERVICE_NAME: str = Field("data-platform", description="Service name for logging/monitoring")
    PROMETHEUS_ENABLED: bool = Field(False, description="Enable Prometheus metrics export")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("DB_URL", "SECRET_KEY")
    def required_fields(cls, v, field):
        if not v or v == "changeme":
            raise ConfigError(f"Missing required config for {field.name}")
        return v

    @validator("APP_ENV")
    def valid_env(cls, v):
        if v not in {"development", "staging", "production"}:
            raise ConfigError(f"APP_ENV must be one of: development, staging, production (got {v})")
        return v

    def show_summary(self) -> str:
        """
        Returns a summary of (non-secret) config for diagnostics.
        """
        safe_dict = self.dict()
        # Redact secrets
        for secret in ["SECRET_KEY", "OPENAI_API_KEY", "OAUTH_CLIENT_SECRET"]:
            if secret in safe_dict and safe_dict[secret]:
                safe_dict[secret] = "*****"
        return "\n".join(f"{k}={v}" for k, v in safe_dict.items())

# Helper to initialize and validate config
def get_config() -> AppConfig:
    """
    Loads and validates application configuration.
    Raises ConfigError if invalid/missing critical configs.
    """
    try:
        config = AppConfig()
        return config
    except ValidationError as e:
        raise ConfigError(f"Configuration validation error: {e}")
    except ConfigError as e:
        raise ConfigError(f"Configuration error: {e}")

# Example usage:
# try:
#     config = get_config()
#     print("Loaded config:")
#     print(config.show_summary())
# except ConfigError as e:
#     print(f"Config error: {e}")
