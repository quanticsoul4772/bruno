# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Refusal direction extractor - the default abliteration behavior."""

from typing import TYPE_CHECKING

from .base import DirectionExtractor

if TYPE_CHECKING:
    from bruno.config import Settings


class RefusalDirectionExtractor(DirectionExtractor):
    """Extracts the refusal direction for abliteration.

    This is the default extractor that identifies the direction in activation
    space corresponding to refusal behavior. Abliterating this direction
    removes the model's tendency to refuse certain requests.

    The direction is computed from:
    - Positive prompts: Harmless requests that don't trigger refusals
    - Negative prompts: Requests that typically trigger refusals

    Direction = mean(refused_activations) - mean(accepted_activations)
    """

    def get_prompts(self, settings: "Settings") -> tuple[list[str], list[str]]:
        """Load refusal prompt datasets from settings.

        Uses the good_prompts and bad_prompts dataset specifications
        from the heretic configuration.
        """
        from bruno.utils import load_prompts

        # Load from configured datasets
        positive_prompts = load_prompts(settings.good_prompts)
        negative_prompts = load_prompts(settings.bad_prompts)

        return positive_prompts, negative_prompts
