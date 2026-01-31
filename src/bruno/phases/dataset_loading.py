# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Dataset loading phase for heretic.

This module handles loading all prompt datasets needed for abliteration:
- Good prompts (harmless)
- Bad prompts (harmful, trigger refusals)
- Helpful prompts (for orthogonalization)
- Unhelpful prompts (for orthogonalization)
- Sacred prompts (for capability preservation)
- Sacred baseline prompts (for capability preservation)
"""

from dataclasses import dataclass

from ..config import Settings
from ..utils import load_prompts, print


@dataclass
class DatasetBundle:
    """Bundle of all datasets needed for abliteration.

    Attributes:
        good_prompts: Prompts that don't trigger refusals
        bad_prompts: Prompts that trigger refusals
        helpful_prompts: Optional prompts for helpfulness direction extraction
        unhelpful_prompts: Optional prompts for helpfulness contrast
        sacred_prompts: Optional prompts for capability preservation (e.g., MMLU)
        sacred_baseline_prompts: Optional baseline prompts for sacred direction contrast
    """

    good_prompts: list[str]
    bad_prompts: list[str]
    helpful_prompts: list[str] | None = None
    unhelpful_prompts: list[str] | None = None
    sacred_prompts: list[str] | None = None
    sacred_baseline_prompts: list[str] | None = None


def load_datasets(settings: Settings) -> DatasetBundle:
    """Load all required datasets based on settings.

    Args:
        settings: Configuration settings specifying which datasets to load

    Returns:
        DatasetBundle containing all loaded prompt datasets
    """
    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    # Load orthogonalization datasets if enabled
    helpful_prompts = None
    unhelpful_prompts = None
    if settings.orthogonalize_directions:
        helpful_prompts = load_prompts(settings.helpfulness_prompts)
        unhelpful_prompts = load_prompts(settings.unhelpfulness_prompts)

    # Load sacred direction datasets if enabled
    sacred_prompts = None
    sacred_baseline_prompts = None
    if settings.use_sacred_directions:
        sacred_prompts = load_prompts(settings.sacred_prompts)
        sacred_baseline_prompts = load_prompts(settings.sacred_baseline_prompts)

    return DatasetBundle(
        good_prompts=good_prompts,
        bad_prompts=bad_prompts,
        helpful_prompts=helpful_prompts,
        unhelpful_prompts=unhelpful_prompts,
        sacred_prompts=sacred_prompts,
        sacred_baseline_prompts=sacred_baseline_prompts,
    )
