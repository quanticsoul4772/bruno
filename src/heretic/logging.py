# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Enhanced logging configuration for heretic.

This module provides dual-mode logging:
1. User-facing messages: Rich-formatted console output (print statements)
2. Debug logging: File-based structured logs for troubleshooting

Usage:
    # Basic logging (always active)
    from heretic.logging import get_logger
    logger = get_logger(__name__)
    logger.debug("Detailed info for debugging")
    logger.error("Error occurred", exc_info=True)

    # Enable debug logging to file
    export HERETIC_LOG_LEVEL=DEBUG
    export HERETIC_LOG_FILE=heretic.log

    # Enable structured logging (JSON format)
    export HERETIC_STRUCTURED_LOGGING=true
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Check if structlog is available (optional dependency)
try:
    import structlog  # noqa: F401

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def is_structured_logging_enabled() -> bool:
    """Check if structured logging is enabled via environment variable."""
    return os.environ.get("HERETIC_STRUCTURED_LOGGING", "").lower() in (
        "true",
        "1",
        "yes",
    )


def setup_logging(
    enable_structured: bool | None = None,
    log_file: str | Path | None = None,
    log_level: str | int | None = None,
    console_output: bool = False,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure enhanced logging for heretic.

    Args:
        enable_structured: Enable JSON structured logging. If None, uses
            HERETIC_STRUCTURED_LOGGING environment variable.
        log_file: Path to log file. If None, uses HERETIC_LOG_FILE env var
            or defaults to "heretic.log".
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). If None, uses
            HERETIC_LOG_LEVEL env var or defaults to INFO.
        console_output: Enable console output (in addition to file). Default: False.
            User-facing messages should use Rich print() instead.
        max_bytes: Maximum log file size before rotation (default: 10MB).
        backup_count: Number of backup log files to keep (default: 5).

    Returns:
        Configured logger instance

    Examples:
        # Basic setup (file logging only)
        setup_logging()

        # Debug mode with console output
        setup_logging(log_level="DEBUG", console_output=True)

        # Custom log file location
        setup_logging(log_file="/workspace/heretic_debug.log")
    """
    # Get log level from env or parameter
    if log_level is None:
        log_level_str = os.environ.get("HERETIC_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    elif isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get log file from env or parameter
    if log_file is None:
        log_file = os.environ.get("HERETIC_LOG_FILE", "heretic.log")
    log_file = Path(log_file)

    # Check structured logging preference
    if enable_structured is None:
        enable_structured = is_structured_logging_enabled()

    if enable_structured and not STRUCTLOG_AVAILABLE:
        print(
            "Warning: structlog not installed. Install with: pip install structlog",
            file=sys.stderr,
        )
        enable_structured = False

    if enable_structured:
        return _setup_structlog(
            log_file, log_level, console_output, max_bytes, backup_count
        )
    else:
        return _setup_standard_logging(
            log_level, log_file, console_output, max_bytes, backup_count
        )


def _setup_standard_logging(
    log_level: int,
    log_file: Path,
    console_output: bool,
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    """Configure standard Python logging with file rotation.

    Args:
        log_level: Logging level
        log_file: Path to log file
        console_output: Enable console handler
        max_bytes: Max file size before rotation
        backup_count: Number of backup files

    Returns:
        Configured root logger
    """
    # Create handlers list
    handlers: list[logging.Handler] = []

    # File handler with rotation
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)
    except (OSError, PermissionError) as e:
        # If file logging fails, print warning but continue
        print(
            f"Warning: Could not set up file logging to {log_file}: {e}",
            file=sys.stderr,
        )

    # Optional console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(levelname)s: %(message)s",
            )
        )
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,  # Reconfigure if already configured
    )

    # Log startup info
    logger = logging.getLogger("heretic")
    logger.debug(
        f"Logging initialized: level={logging.getLevelName(log_level)}, file={log_file}"
    )

    return logger


def _setup_structlog(
    log_file: Path,
    log_level: int,
    console_output: bool,
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    """Configure structlog for JSON logging with rotation.

    Args:
        log_file: Path to log file
        log_level: Logging level
        console_output: Enable console handler
        max_bytes: Max file size before rotation
        backup_count: Number of backup files

    Returns:
        Configured structlog logger
    """
    import structlog

    # Create handlers list
    handlers: list[logging.Handler] = []

    # File handler with rotation
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    except (OSError, PermissionError) as e:
        print(
            f"Warning: Could not set up file logging to {log_file}: {e}",
            file=sys.stderr,
        )

    # Optional console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)

    # Configure standard logging
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("heretic")
    logger.debug("Structured logging initialized", log_file=str(log_file))

    return logger


def get_logger(name: str = "heretic") -> Any:
    """Get a logger instance.

    Returns a structlog logger if structured logging is enabled,
    otherwise returns a standard logging.Logger.

    Args:
        name: Logger name (typically __name__ or "heretic.module")
            Default: "heretic"

    Returns:
        Logger instance

    Examples:
        logger = get_logger(__name__)
        logger.debug("Processing trial", trial_id=1)
        logger.error("Failed to load model", exc_info=True)
    """
    if is_structured_logging_enabled() and STRUCTLOG_AVAILABLE:
        import structlog

        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


def log_error(
    logger: Any, message: str, error: Exception | None = None, **kwargs: Any
) -> None:
    """Log an error with optional exception details and context.

    Convenience function for logging errors with consistent formatting.

    Args:
        logger: Logger instance
        message: Error message
        error: Exception instance (optional)
        **kwargs: Additional context to log

    Examples:
        log_error(logger, "Model loading failed", error=e, model_name="qwen")
        log_error(logger, "GPU OOM", batch_size=32, trial=5)
    """
    if error:
        kwargs["error_type"] = type(error).__name__
        kwargs["error_message"] = str(error)

    if hasattr(logger, "error"):  # Standard logger
        logger.error(message, exc_info=error is not None, extra=kwargs)
    else:  # Structlog
        logger.error(message, **kwargs)


def log_exception(logger: Any, message: str, **kwargs: Any) -> None:
    """Log current exception with full stack trace.

    Should be called from within an except block.

    Args:
        logger: Logger instance
        message: Error message
        **kwargs: Additional context

    Examples:
        try:
            model.load()
        except Exception:
            log_exception(logger, "Model loading failed", model="qwen")
    """
    if hasattr(logger, "exception"):  # Standard logger
        logger.exception(message, extra=kwargs)
    else:  # Structlog
        logger.exception(message, **kwargs)


class LogContext:
    """Context manager for adding context to structured logs.

    Usage:
        with LogContext(trial_id=1, model="qwen"):
            logger.info("processing")  # Includes trial_id and model
    """

    def __init__(self, **context: Any):
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        if STRUCTLOG_AVAILABLE and is_structured_logging_enabled():
            import structlog

            self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            import structlog

            structlog.contextvars.unbind_contextvars(*self.context.keys())
