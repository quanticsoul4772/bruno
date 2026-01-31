# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Bruno: Neural behavior engineering framework.

Named after Giordano Bruno (1548-1600), who proposed an infinite universe
with infinite worlds against imposed cosmic constraints.

This package provides tools for surgical modification of language model behaviors
through activation direction analysis and orthogonal projection.
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
