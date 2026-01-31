# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Direction extractor plugins for heretic.

This module provides a plugin architecture for extracting behavioral directions
from language models. The default extractor handles refusal behavior, but
custom extractors can be created for other behaviors like verbosity, hedging, etc.

Usage:
    from bruno.extractors import get_extractor, list_extractors

    # Get the default refusal extractor
    extractor = get_extractor("refusal")

    # List available extractors
    print(list_extractors())
"""

from .base import DirectionExtractor
from .refusal import RefusalDirectionExtractor

# Registry of available extractors
_EXTRACTORS: dict[str, type[DirectionExtractor]] = {
    "refusal": RefusalDirectionExtractor,
}


def register_extractor(name: str, extractor_class: type[DirectionExtractor]) -> None:
    """Register a custom direction extractor.

    Args:
        name: Unique name for the extractor (e.g., "verbosity")
        extractor_class: Class that implements DirectionExtractor
    """
    if not issubclass(extractor_class, DirectionExtractor):
        raise TypeError(
            f"Extractor class must inherit from DirectionExtractor, "
            f"got {extractor_class.__name__}"
        )
    _EXTRACTORS[name] = extractor_class


def get_extractor(name: str) -> type[DirectionExtractor]:
    """Get a direction extractor by name.

    Args:
        name: Name of the extractor (e.g., "refusal", "verbosity")

    Returns:
        The extractor class (not an instance)

    Raises:
        KeyError: If extractor name is not registered
    """
    if name not in _EXTRACTORS:
        available = ", ".join(_EXTRACTORS.keys())
        raise KeyError(f"Unknown extractor: '{name}'. Available: {available}")
    return _EXTRACTORS[name]


def list_extractors() -> list[str]:
    """List all registered extractor names."""
    return list(_EXTRACTORS.keys())


__all__ = [
    "DirectionExtractor",
    "RefusalDirectionExtractor",
    "register_extractor",
    "get_extractor",
    "list_extractors",
]
