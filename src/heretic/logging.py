# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Structured logging configuration for heretic.

This module provides optional structured logging using structlog.
When enabled, logs are written in JSON format to heretic.log for
better observability during long experiments.

Usage:
    # Enable structured logging via environment variable
    export HERETIC_STRUCTURED_LOGGING=true
    
    # Or programmatically
    from heretic.logging import setup_logging, get_logger
    setup_logging(enable_structured=True)
    logger = get_logger("heretic.main")
    logger.info("trial_completed", trial_id=1, kl_divergence=0.5)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Check if structlog is available (optional dependency)
try:
    import structlog  # noqa: F401
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def is_structured_logging_enabled() -> bool:
    """Check if structured logging is enabled via environment variable."""
    return os.environ.get("HERETIC_STRUCTURED_LOGGING", "").lower() in ("true", "1", "yes")


def setup_logging(
    enable_structured: bool | None = None,
    log_file: str | Path = "heretic.log",
    log_level: int = logging.INFO,
) -> None:
    """Configure logging for heretic.
    
    Args:
        enable_structured: Enable JSON structured logging. If None, uses
            HERETIC_STRUCTURED_LOGGING environment variable.
        log_file: Path to log file for structured logs.
        log_level: Logging level (default: INFO)
    """
    if enable_structured is None:
        enable_structured = is_structured_logging_enabled()
    
    if enable_structured and not STRUCTLOG_AVAILABLE:
        print(
            "Warning: structlog not installed. "
            "Install with: pip install structlog",
            file=sys.stderr,
        )
        enable_structured = False
    
    if enable_structured:
        _setup_structlog(log_file, log_level)
    else:
        _setup_standard_logging(log_level)


def _setup_standard_logging(log_level: int) -> None:
    """Configure standard Python logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _setup_structlog(log_file: str | Path, log_level: int) -> None:
    """Configure structlog for JSON logging."""
    import structlog
    
    # Configure standard logging to write to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler],
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


def get_logger(name: str) -> Any:
    """Get a logger instance.
    
    Returns a structlog logger if structured logging is enabled,
    otherwise returns a standard logging.Logger.
    
    Args:
        name: Logger name (typically __name__ or "heretic.module")
        
    Returns:
        Logger instance
    """
    if is_structured_logging_enabled() and STRUCTLOG_AVAILABLE:
        import structlog
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


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
