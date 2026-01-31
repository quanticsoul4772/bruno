# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Heretic: LLM abliteration toolkit.

This package provides tools for removing refusal behavior from large language models
through abliteration (orthogonalization of refusal directions).
"""

from .error_tracker import (
    ErrorSeverity,
    ErrorTracker,
    SuppressedError,
    error_tracker,
    get_error_tracker,
    record_suppressed_error,
)
from .exceptions import SacredDirectionError
from .model import SacredDirectionResult
from .phases import DatasetBundle, DirectionExtractionResult

__all__ = [
    # Error tracking
    "ErrorSeverity",
    "ErrorTracker",
    "SuppressedError",
    "error_tracker",
    "get_error_tracker",
    "record_suppressed_error",
    # Exceptions
    "SacredDirectionError",
    # Model
    "SacredDirectionResult",
    # Phases
    "DatasetBundle",
    "DirectionExtractionResult",
]
