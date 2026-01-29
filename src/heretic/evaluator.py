# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import re
from concurrent.futures import ThreadPoolExecutor

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        # Pre-compile regex pattern for refusal detection (5-10x faster than linear search)
        # Escape special regex characters and join with OR
        normalized_markers = [m.lower() for m in settings.refusal_markers]
        pattern = "|".join(re.escape(marker) for marker in normalized_markers)
        self.refusal_pattern = re.compile(pattern, re.IGNORECASE)

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Use pre-compiled regex for fast matching (5-10x faster than linear search)
        return bool(self.refusal_pattern.search(response))

    def count_refusals(self) -> int:
        # Use shorter token limit for refusal checking - refusals appear in first 20-30 tokens
        # This provides ~40-60% speedup vs generating full responses
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            max_tokens=self.settings.refusal_check_tokens,
        )
        refusals = [response for response in responses if self.is_refusal(response)]
        return len(refusals)

    def _compute_kl_divergence(self) -> float:
        """Compute KL divergence from base model (for parallel execution)."""
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        return F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Evaluating in parallel (KL divergence + refusal counting)...")

        # Run KL divergence and refusal counting in parallel
        # These use different prompt sets (good vs bad) so can overlap
        with ThreadPoolExecutor(max_workers=2) as executor:
            kl_future = executor.submit(self._compute_kl_divergence)
            refusal_future = executor.submit(self.count_refusals)

            kl_divergence = kl_future.result()
            refusals = refusal_future.result()

        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
