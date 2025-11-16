"""Centralized configuration management for Model Chat CLI.

Supports environment variable overrides for all settings.
Environment variables should be prefixed with MODEL_CHAT_
"""
import os
from pathlib import Path
from typing import List


class Config:
    """Application configuration with environment override support."""

    # Network Scanner Configuration
    COMMON_PORTS: List[int] = [
        int(p) for p in os.getenv(
            "MODEL_CHAT_COMMON_PORTS",
            "11434,1234,5000,8000,8080"
        ).split(",")
    ]

    SCAN_TIMEOUT: float = float(os.getenv("MODEL_CHAT_SCAN_TIMEOUT", "30.0"))
    SCAN_SEMAPHORE_LIMIT: int = int(os.getenv("MODEL_CHAT_SCAN_SEMAPHORE", "50"))
    PROBE_TIMEOUT: float = float(os.getenv("MODEL_CHAT_PROBE_TIMEOUT", "2.0"))
    PROBE_CONNECT_TIMEOUT: float = float(os.getenv("MODEL_CHAT_PROBE_CONNECT_TIMEOUT", "1.0"))

    # Cache Configuration
    CACHE_FILE: Path = Path(
        os.getenv(
            "MODEL_CHAT_CACHE_FILE",
            str(Path.home() / ".model_chat_cache.json")
        )
    )

    THEME_FILE: Path = Path(
        os.getenv(
            "MODEL_CHAT_THEME_FILE",
            str(Path.home() / ".model_chat_theme.json")
        )
    )

    FAVORITES_FILE: Path = Path(
        os.getenv(
            "MODEL_CHAT_FAVORITES_FILE",
            str(Path.home() / ".model_chat_favorites.json")
        )
    )

    # Memory Limits
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MODEL_CHAT_MAX_HISTORY", "1000"))
    MAX_DISPLAY_MESSAGES: int = int(os.getenv("MODEL_CHAT_MAX_DISPLAY", "100"))

    # Retry Configuration
    MAX_RETRIES: int = int(os.getenv("MODEL_CHAT_MAX_RETRIES", "3"))
    RETRY_BASE_DELAY: float = float(os.getenv("MODEL_CHAT_RETRY_BASE_DELAY", "1.0"))
    RETRY_MAX_DELAY: float = float(os.getenv("MODEL_CHAT_RETRY_MAX_DELAY", "8.0"))

    # Stress Test Configuration
    DEFAULT_TOKEN_SIZES: List[int] = [
        int(s) for s in os.getenv(
            "MODEL_CHAT_TOKEN_SIZES",
            "500,1000,2000,5000"
        ).split(",")
    ]

    # UI Configuration
    ENABLE_BANNER: bool = os.getenv("MODEL_CHAT_ENABLE_BANNER", "true").lower() == "true"
    BANNER_FONT: str = os.getenv("MODEL_CHAT_BANNER_FONT", "slant")
    BANNER_DURATION: float = float(os.getenv("MODEL_CHAT_BANNER_DURATION", "0.8"))

    DEFAULT_THEME: str = os.getenv("MODEL_CHAT_DEFAULT_THEME", "default")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("MODEL_CHAT_LOG_LEVEL", "INFO")
    LOG_FILE: Path = Path(
        os.getenv(
            "MODEL_CHAT_LOG_FILE",
            str(Path.home() / ".model_chat.log")
        )
    )

    # Model Configuration
    DEFAULT_TEMPERATURE: float = float(os.getenv("MODEL_CHAT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("MODEL_CHAT_MAX_TOKENS", "2048"))

    @classmethod
    def get_config_summary(cls) -> dict:
        """Get summary of all configuration values.

        Returns:
            Dictionary of configuration keys and values
        """
        return {
            "network": {
                "common_ports": cls.COMMON_PORTS,
                "scan_timeout": cls.SCAN_TIMEOUT,
                "scan_semaphore_limit": cls.SCAN_SEMAPHORE_LIMIT,
                "probe_timeout": cls.PROBE_TIMEOUT,
            },
            "cache": {
                "cache_file": str(cls.CACHE_FILE),
                "theme_file": str(cls.THEME_FILE),
                "favorites_file": str(cls.FAVORITES_FILE),
            },
            "memory": {
                "max_history_messages": cls.MAX_HISTORY_MESSAGES,
                "max_display_messages": cls.MAX_DISPLAY_MESSAGES,
            },
            "retry": {
                "max_retries": cls.MAX_RETRIES,
                "retry_base_delay": cls.RETRY_BASE_DELAY,
                "retry_max_delay": cls.RETRY_MAX_DELAY,
            },
            "ui": {
                "enable_banner": cls.ENABLE_BANNER,
                "banner_font": cls.BANNER_FONT,
                "default_theme": cls.DEFAULT_THEME,
            },
            "logging": {
                "log_level": cls.LOG_LEVEL,
                "log_file": str(cls.LOG_FILE),
            },
        }

    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration values.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not cls.COMMON_PORTS:
            errors.append("COMMON_PORTS cannot be empty")

        if cls.SCAN_TIMEOUT <= 0:
            errors.append("SCAN_TIMEOUT must be positive")

        if cls.MAX_HISTORY_MESSAGES < cls.MAX_DISPLAY_MESSAGES:
            errors.append("MAX_HISTORY_MESSAGES must be >= MAX_DISPLAY_MESSAGES")

        if cls.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES cannot be negative")

        if cls.RETRY_BASE_DELAY <= 0 or cls.RETRY_MAX_DELAY <= 0:
            errors.append("Retry delays must be positive")

        if cls.RETRY_MAX_DELAY < cls.RETRY_BASE_DELAY:
            errors.append("RETRY_MAX_DELAY must be >= RETRY_BASE_DELAY")

        return errors


# Create singleton instance
config = Config()

# Validate on import
validation_errors = config.validate()
if validation_errors:
    import warnings
    for error in validation_errors:
        warnings.warn(f"Configuration validation error: {error}")
