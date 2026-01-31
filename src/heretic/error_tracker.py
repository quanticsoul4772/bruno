# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Centralized error tracking for suppressed exceptions.

This module provides a mechanism to track exceptions that are caught and
suppressed (not re-raised) throughout the codebase. This enables:
- Debugging: See all errors that occurred during a run
- Monitoring: Track error patterns over time
- Reporting: Generate summaries of suppressed errors

Usage:
    from heretic.error_tracker import error_tracker, record_suppressed_error

    try:
        risky_operation()
    except Exception as e:
        record_suppressed_error(
            error=e,
            context="model_reload",
            module="main",
            severity="warning",
            details={"trial_id": 5, "batch_size": 32},
        )
        # Continue with fallback behavior...

    # Later, get a summary:
    print(error_tracker.get_summary())
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for suppressed errors."""

    DEBUG = "debug"  # Minor issues, logged for completeness
    INFO = "info"  # Informational, expected fallbacks
    WARNING = "warning"  # Potentially problematic, but handled
    ERROR = "error"  # Significant issue, may affect results
    CRITICAL = "critical"  # Severe issue, likely affects correctness


@dataclass
class SuppressedError:
    """Record of a suppressed exception.

    Attributes:
        error_type: The exception class name (e.g., "ValueError")
        error_message: The exception message string
        context: What operation was being attempted (e.g., "model_reload")
        module: Which module/file the error occurred in
        severity: How serious the error is
        timestamp: When the error occurred (Unix timestamp)
        details: Additional context as key-value pairs
        traceback_str: Optional stack trace for debugging
    """

    error_type: str
    error_message: str
    context: str
    module: str
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    traceback_str: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "context": self.context,
            "module": self.module,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details,
            "traceback": self.traceback_str,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"[{self.severity.value.upper()}] {self.module}/{self.context}: "
            f"{self.error_type}: {self.error_message}"
        )


class ErrorTracker:
    """Thread-safe singleton for tracking suppressed exceptions.

    This class maintains a log of all suppressed exceptions that occurred
    during program execution, enabling post-hoc analysis and debugging.

    The tracker is a singleton - use `error_tracker` instance or
    `get_error_tracker()` function to access it.
    """

    _instance: "ErrorTracker | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ErrorTracker":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the error tracker (only runs once due to singleton)."""
        if self._initialized:
            return

        self._errors: list[SuppressedError] = []
        self._errors_lock = threading.Lock()
        self._max_errors = 1000  # Prevent unbounded memory growth
        self._initialized = True

    def record(
        self,
        error: Exception | None,
        context: str,
        module: str,
        severity: ErrorSeverity | str = ErrorSeverity.WARNING,
        details: dict[str, Any] | None = None,
        include_traceback: bool = False,
    ) -> SuppressedError:
        """Record a suppressed exception.

        Args:
            error: The exception that was caught (can be None for non-exception errors)
            context: What operation was being attempted
            module: Which module/file the error occurred in
            severity: How serious the error is (string or ErrorSeverity)
            details: Additional context as key-value pairs
            include_traceback: Whether to capture the full stack trace

        Returns:
            The SuppressedError record that was created
        """
        # Convert string severity to enum
        if isinstance(severity, str):
            try:
                severity = ErrorSeverity(severity.lower())
            except ValueError:
                severity = ErrorSeverity.WARNING

        # Extract error info
        if error is not None:
            error_type = type(error).__name__
            error_message = str(error)
        else:
            error_type = "Unknown"
            error_message = "No exception provided"

        # Capture traceback if requested
        traceback_str = None
        if include_traceback and error is not None:
            traceback_str = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )

        # Create the record
        record = SuppressedError(
            error_type=error_type,
            error_message=error_message,
            context=context,
            module=module,
            severity=severity,
            details=details or {},
            traceback_str=traceback_str,
        )

        # Store it thread-safely
        with self._errors_lock:
            self._errors.append(record)

            # Prevent unbounded growth by removing oldest errors
            if len(self._errors) > self._max_errors:
                self._errors = self._errors[-self._max_errors :]

        # Also log to the structured logger
        # Note: We use extra={} for standard logging compatibility
        # 'module' is a reserved field in Python's logging, so we use 'source_module'
        log_method = getattr(logger, severity.value, logger.warning)
        try:
            # Try structured logging style first (for structlog)
            log_method(
                f"Suppressed {error_type}: {error_message}",
                extra={
                    "context": context,
                    "source_module": module,  # 'module' is reserved in logging
                    **(details or {}),
                },
            )
        except (TypeError, KeyError):
            # Fallback for loggers that don't accept extra or have reserved field conflicts
            log_method(
                f"Suppressed {error_type}: {error_message} (context={context}, module={module})"
            )

        return record

    def get_all(self) -> list[SuppressedError]:
        """Get all recorded suppressed errors.

        Returns:
            List of all SuppressedError records (copy to prevent mutation)
        """
        with self._errors_lock:
            return list(self._errors)

    def get_by_severity(self, severity: ErrorSeverity | str) -> list[SuppressedError]:
        """Get errors filtered by severity.

        Args:
            severity: The severity level to filter by

        Returns:
            List of matching SuppressedError records
        """
        if isinstance(severity, str):
            severity = ErrorSeverity(severity.lower())

        with self._errors_lock:
            return [e for e in self._errors if e.severity == severity]

    def get_by_module(self, module: str) -> list[SuppressedError]:
        """Get errors filtered by module.

        Args:
            module: The module name to filter by

        Returns:
            List of matching SuppressedError records
        """
        with self._errors_lock:
            return [e for e in self._errors if e.module == module]

    def get_by_context(self, context: str) -> list[SuppressedError]:
        """Get errors filtered by context.

        Args:
            context: The context string to filter by

        Returns:
            List of matching SuppressedError records
        """
        with self._errors_lock:
            return [e for e in self._errors if e.context == context]

    def get_recent(self, n: int = 10) -> list[SuppressedError]:
        """Get the N most recent errors.

        Args:
            n: Number of errors to return

        Returns:
            List of most recent SuppressedError records
        """
        with self._errors_lock:
            return list(self._errors[-n:])

    def count(self) -> int:
        """Get the total count of suppressed errors."""
        with self._errors_lock:
            return len(self._errors)

    def count_by_severity(self) -> dict[str, int]:
        """Get counts grouped by severity.

        Returns:
            Dictionary mapping severity names to counts
        """
        counts: dict[str, int] = {s.value: 0 for s in ErrorSeverity}
        with self._errors_lock:
            for error in self._errors:
                counts[error.severity.value] += 1
        return counts

    def get_summary(self) -> str:
        """Get a human-readable summary of all suppressed errors.

        Returns:
            Multi-line string summarizing all errors
        """
        with self._errors_lock:
            if not self._errors:
                return "No suppressed errors recorded."

            lines = [
                f"Suppressed Errors Summary ({len(self._errors)} total)",
                "=" * 50,
            ]

            # Group by severity (inline to avoid deadlock from calling count_by_severity)
            by_severity: dict[str, int] = {s.value: 0 for s in ErrorSeverity}
            for error in self._errors:
                by_severity[error.severity.value] += 1

            lines.append("\nBy Severity:")
            for severity, count in by_severity.items():
                if count > 0:
                    lines.append(f"  {severity.upper()}: {count}")

            # Group by module
            by_module: dict[str, int] = {}
            for error in self._errors:
                by_module[error.module] = by_module.get(error.module, 0) + 1

            lines.append("\nBy Module:")
            for module, count in sorted(by_module.items(), key=lambda x: -x[1]):
                lines.append(f"  {module}: {count}")

            # Recent errors
            lines.append("\nMost Recent Errors:")
            for error in self._errors[-5:]:
                lines.append(f"  - {error}")

            return "\n".join(lines)

    def clear(self) -> int:
        """Clear all recorded errors.

        Returns:
            Number of errors that were cleared
        """
        with self._errors_lock:
            count = len(self._errors)
            self._errors.clear()
            return count

    def export_json(self) -> list[dict[str, Any]]:
        """Export all errors as a list of dictionaries for JSON serialization.

        Returns:
            List of error dictionaries
        """
        with self._errors_lock:
            return [e.to_dict() for e in self._errors]


# Global singleton instance
error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance.

    Returns:
        The singleton ErrorTracker instance
    """
    return error_tracker


def record_suppressed_error(
    error: Exception | None,
    context: str,
    module: str,
    severity: ErrorSeverity | str = ErrorSeverity.WARNING,
    details: dict[str, Any] | None = None,
    include_traceback: bool = False,
) -> SuppressedError:
    """Convenience function to record a suppressed error.

    This is a shortcut for `error_tracker.record()` that is easier to import
    and use throughout the codebase.

    Args:
        error: The exception that was caught
        context: What operation was being attempted
        module: Which module/file the error occurred in
        severity: How serious the error is
        details: Additional context as key-value pairs
        include_traceback: Whether to capture the full stack trace

    Returns:
        The SuppressedError record that was created

    Example:
        try:
            model.reload_model()
        except Exception as e:
            record_suppressed_error(
                error=e,
                context="oom_recovery",
                module="main",
                severity="warning",
                details={"oom_count": oom_count},
            )
    """
    return error_tracker.record(
        error=error,
        context=context,
        module=module,
        severity=severity,
        details=details,
        include_traceback=include_traceback,
    )
