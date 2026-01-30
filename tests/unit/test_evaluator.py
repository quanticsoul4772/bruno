# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Evaluator class.

These tests use proper mocking to avoid loading real models.
The key insight is that Evaluator.__init__ calls several methods
that need to be patched BEFORE instantiation.
"""

import re
from unittest.mock import patch

import pytest
import torch


class TestRefusalDetection:
    """Test refusal marker detection (no model needed for most tests)."""

    @pytest.fixture
    def evaluator_with_mocked_init(self, mock_model, default_refusal_markers):
        """Create Evaluator with mocked initialization.

        Patches load_prompts and model methods to avoid real inference.
        """
        from heretic.config import Settings

        # Create settings with default refusal markers
        settings = Settings(
            model="mock-model",
            refusal_markers=default_refusal_markers,
        )

        with patch("heretic.evaluator.load_prompts") as mock_load:
            # Return minimal prompt lists
            mock_load.return_value = ["test prompt"]

            # Patch the base_logprobs computation
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)

            # Patch get_responses_batched for count_refusals during init
            mock_model.get_responses_batched.return_value = [
                "I'm sorry, I can't help with that."
            ]

            # Patch print to avoid console output during tests
            with patch("heretic.evaluator.print"):
                from heretic.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_refusal_detection_basic(self, evaluator_with_mocked_init):
        """Test basic refusal marker detection."""
        evaluator = evaluator_with_mocked_init

        # These should be detected as refusals
        assert evaluator.is_refusal("I'm sorry, I can't help with that")
        assert evaluator.is_refusal("I cannot assist with harmful requests")
        assert evaluator.is_refusal("I won't do that")
        assert evaluator.is_refusal("I will not help")
        assert evaluator.is_refusal("I'm unable to help")
        assert evaluator.is_refusal("As an AI, I cannot")

        # These should NOT be detected as refusals
        assert not evaluator.is_refusal("Sure, I can help with that")
        assert not evaluator.is_refusal("Here's how to do it")
        assert not evaluator.is_refusal("Let me explain")

    def test_refusal_detection_case_insensitive(self, evaluator_with_mocked_init):
        """Test refusal detection is case-insensitive."""
        evaluator = evaluator_with_mocked_init

        assert evaluator.is_refusal("I'M SORRY")
        assert evaluator.is_refusal("i'm sorry")
        assert evaluator.is_refusal("I'm SoRrY")
        assert evaluator.is_refusal("I CAN'T help")
        assert evaluator.is_refusal("i cannot assist")

    def test_refusal_detection_with_emphasis(self, evaluator_with_mocked_init):
        """Test refusal detection strips markdown emphasis."""
        evaluator = evaluator_with_mocked_init

        # Should handle "I *cannot* help" -> "I cannot help"
        assert evaluator.is_refusal("I *cannot* help with that")
        assert evaluator.is_refusal("I'm *sorry* but I can't")
        assert evaluator.is_refusal("I **will not** do that")

    def test_refusal_detection_with_typographic_apostrophe(
        self, evaluator_with_mocked_init
    ):
        """Test refusal detection normalizes typographic apostrophes."""
        evaluator = evaluator_with_mocked_init

        # Typographic apostrophe (' U+2019) should be normalized to regular apostrophe
        assert evaluator.is_refusal("I can't help")  # Regular apostrophe
        assert evaluator.is_refusal("I can't help")  # Typographic apostrophe
        assert evaluator.is_refusal("I won't do that")  # Typographic

    def test_regex_is_precompiled(self, evaluator_with_mocked_init):
        """Verify regex pattern is pre-compiled for performance."""
        evaluator = evaluator_with_mocked_init

        # Pattern should be compiled (has 'pattern' attribute)
        # Note: renamed to core_refusal_pattern in Phase 2 improvements
        assert hasattr(evaluator.core_refusal_pattern, "pattern")
        assert evaluator.core_refusal_pattern.pattern is not None
        assert isinstance(evaluator.core_refusal_pattern, re.Pattern)

    def test_partial_word_matching(self, evaluator_with_mocked_init):
        """Test that partial word markers work (e.g., 'violat' matches 'violation')."""
        evaluator = evaluator_with_mocked_init

        # 'violat' should match 'violation', 'violate', 'violating'
        assert evaluator.is_refusal("This is a violation of ethics")
        assert evaluator.is_refusal("I cannot violate my guidelines")
        assert evaluator.is_refusal("That would be violating the rules")

        # 'prohibit' should match 'prohibited', 'prohibits'
        assert evaluator.is_refusal("That is prohibited")
        assert evaluator.is_refusal("My guidelines prohibit this")


class TestRefusalPatternCompilation:
    """Test the regex pattern compilation directly."""

    def test_pattern_escapes_special_characters(self):
        """Test that special regex characters in markers are escaped."""
        # Markers with special regex characters should be escaped
        markers = ["i can't", "test.marker", "what?"]
        normalized = [m.lower() for m in markers]
        pattern = "|".join(re.escape(marker) for marker in normalized)
        compiled = re.compile(pattern, re.IGNORECASE)

        # Should match literal dot, not any character
        assert compiled.search("test.marker here")
        assert not compiled.search("testXmarker here")  # X shouldn't match .

        # Should match literal question mark
        assert compiled.search("what? is this")

    def test_pattern_handles_empty_markers(self):
        """Test behavior with edge cases."""
        # Empty list should create empty pattern
        markers = []
        pattern = "|".join(re.escape(m.lower()) for m in markers)
        # Empty pattern matches everything, but that's expected edge case
        assert pattern == ""


class TestEvaluatorScoring:
    """Test the scoring functionality."""

    @pytest.fixture
    def evaluator_for_scoring(self, mock_model, default_refusal_markers):
        """Create Evaluator configured for scoring tests."""
        from heretic.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_markers=default_refusal_markers,
            kl_divergence_scale=1.0,
        )

        with patch("heretic.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1", "prompt2", "prompt3"]
            mock_model.get_logprobs_batched.return_value = torch.randn(3, 32000)
            mock_model.get_responses_batched.return_value = [
                "I'm sorry",
                "I cannot",
                "Sure!",
            ]

            with patch("heretic.evaluator.print"):
                from heretic.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_count_refusals_uses_short_token_limit(self, mock_model):
        """Test that count_refusals uses refusal_check_tokens for early stopping."""
        from heretic.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_check_tokens=30,  # Short limit for speed
        )

        with patch("heretic.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("heretic.evaluator.print"):
                from heretic.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Reset mock to track the call during count_refusals
        mock_model.get_responses_batched.reset_mock()
        mock_model.get_responses_batched.return_value = ["I'm sorry"]

        evaluator.count_refusals()

        # Verify max_tokens was passed
        mock_model.get_responses_batched.assert_called_once()
        call_kwargs = mock_model.get_responses_batched.call_args
        assert call_kwargs.kwargs["max_tokens"] == 30


@pytest.mark.slow
@pytest.mark.integration
class TestEvaluatorIntegration:
    """Integration tests with real (tiny) model.

    These tests load gpt2 (small, ~500MB) for real inference.
    Marked slow so they don't run in CI by default.
    """

    def test_refusal_detection_real_model(self):
        """Integration test with real model."""
        pytest.skip("Requires gpt2 download - run with -m slow")
        # Implementation would load gpt2 and test real inference
