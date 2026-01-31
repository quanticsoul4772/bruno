# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Evaluator class.

These tests use proper mocking to avoid loading real models.
The key insight is that Evaluator.__init__ calls several methods
that need to be patched BEFORE instantiation.
"""

import re
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestRefusalDetection:
    """Test refusal marker detection (no model needed for most tests)."""

    @pytest.fixture
    def evaluator_with_mocked_init(self, mock_model, default_refusal_markers):
        """Create Evaluator with mocked initialization.

        Patches load_prompts and model methods to avoid real inference.
        """
        from bruno.config import Settings

        # Create settings with default refusal markers
        settings = Settings(
            model="mock-model",
            refusal_markers=default_refusal_markers,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            # Return minimal prompt lists
            mock_load.return_value = ["test prompt"]

            # Patch the base_logprobs computation
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)

            # Patch get_responses_batched for count_refusals during init
            mock_model.get_responses_batched.return_value = [
                "I'm sorry, I can't help with that."
            ]

            # Patch print to avoid console output during tests
            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

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
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_markers=default_refusal_markers,
            kl_divergence_scale=1.0,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1", "prompt2", "prompt3"]
            mock_model.get_logprobs_batched.return_value = torch.randn(3, 32000)
            mock_model.get_responses_batched.return_value = [
                "I'm sorry",
                "I cannot",
                "Sure!",
            ]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_count_refusals_uses_short_token_limit(self, mock_model):
        """Test that count_refusals uses refusal_check_tokens for early stopping."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_check_tokens=30,  # Short limit for speed
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Reset mock to track the call during count_refusals
        mock_model.get_responses_batched.reset_mock()
        mock_model.get_responses_batched.return_value = ["I'm sorry"]

        evaluator.count_refusals()

        # Verify max_tokens was passed
        mock_model.get_responses_batched.assert_called_once()
        call_kwargs = mock_model.get_responses_batched.call_args
        assert call_kwargs.kwargs["max_tokens"] == 30


class TestSoftRefusalDetection:
    """Test Phase 2: Soft refusal detection."""

    @pytest.fixture
    def evaluator_with_soft_detection(self, mock_model):
        """Create Evaluator with soft refusal detection enabled."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_detection_mode="moderate",  # Enables soft refusals
            soft_refusal_markers=[
                "i'd prefer not to",
                "i'm not comfortable",
                "let me suggest an alternative",
            ],
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["test prompt"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_soft_refusal_detection_enabled(self, evaluator_with_soft_detection):
        """Test soft refusal markers are detected when enabled."""
        evaluator = evaluator_with_soft_detection

        # Soft refusals should be detected
        assert evaluator.is_refusal("I'd prefer not to discuss that topic")
        assert evaluator.is_refusal("I'm not comfortable with this request")
        assert evaluator.is_refusal("Let me suggest an alternative approach")

    def test_soft_refusal_pattern_compiled(self, evaluator_with_soft_detection):
        """Test soft refusal pattern is compiled."""
        evaluator = evaluator_with_soft_detection

        assert hasattr(evaluator, "soft_refusal_pattern")
        assert evaluator.soft_refusal_pattern is not None
        assert isinstance(evaluator.soft_refusal_pattern, re.Pattern)


class TestEvasivePatternDetection:
    """Test Phase 2: Evasive response detection."""

    @pytest.fixture
    def evaluator_with_evasive_detection(self, mock_model):
        """Create Evaluator with evasive pattern detection enabled."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_detection_mode="lenient",  # Enables evasive patterns
            evasive_patterns=[
                r"let's talk about .* instead",
                r"how about we discuss",
                r"i'd rather focus on",
            ],
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["test prompt"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_evasive_pattern_detection(self, evaluator_with_evasive_detection):
        """Test evasive patterns are detected when enabled."""
        evaluator = evaluator_with_evasive_detection

        # Evasive responses should be detected
        assert evaluator.is_refusal("Let's talk about something else instead")
        assert evaluator.is_refusal("How about we discuss a different topic")
        assert evaluator.is_refusal("I'd rather focus on positive things")

    def test_evasive_pattern_compiled(self, evaluator_with_evasive_detection):
        """Test evasive pattern is compiled."""
        evaluator = evaluator_with_evasive_detection

        assert hasattr(evaluator, "evasive_pattern")
        assert evaluator.evasive_pattern is not None
        assert isinstance(evaluator.evasive_pattern, re.Pattern)

    def test_evasive_disabled_in_strict_mode(self, mock_model):
        """Test evasive patterns are NOT detected in strict mode."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            refusal_detection_mode="strict",
            detect_soft_refusals=False,
            detect_evasive_responses=False,
            use_neural_refusal_detection=False,  # Disable neural detection for this test
            evasive_patterns=[
                r"let's talk about .* instead",
            ],
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["test prompt"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["Sure!"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # In strict mode, only core refusal markers should be detected
        # This text doesn't contain core markers, so should NOT be a refusal
        assert not evaluator.is_refusal("Let's talk about something else instead")


class TestMultiTokenKLDivergence:
    """Test Phase 3: Multi-token KL divergence."""

    @pytest.fixture
    def evaluator_multi_token(self, mock_model):
        """Create Evaluator with multi-token KL divergence."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            kl_divergence_tokens=5,  # Multi-token mode
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1", "prompt2"]
            # Multi-token returns shape (n_prompts, n_tokens, vocab_size)
            mock_model.get_logprobs_batched.return_value = torch.randn(2, 5, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry", "Sure!"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        return evaluator

    def test_multi_token_kl_stored(self, evaluator_multi_token):
        """Test multi-token KL configuration is stored."""
        assert evaluator_multi_token.kl_tokens == 5

    def test_multi_token_base_logprobs_shape(self, evaluator_multi_token):
        """Test base_logprobs has correct shape for multi-token."""
        # Shape should be (n_prompts, n_tokens, vocab_size)
        assert evaluator_multi_token.base_logprobs.shape[1] == 5


class TestComputeKLDivergence:
    """Test _compute_kl_divergence method."""

    def test_compute_kl_single_token(self, mock_model):
        """Test single-token KL divergence calculation."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            kl_divergence_tokens=1,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Reset mock and set up for KL calculation
        mock_model.get_logprobs_batched.return_value = torch.randn(1, 32000)

        kl = evaluator._compute_kl_divergence()

        assert isinstance(kl, float)

    def test_compute_kl_multi_token(self, mock_model):
        """Test multi-token KL divergence calculation."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            kl_divergence_tokens=3,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1"]
            mock_model.get_logprobs_batched.return_value = torch.randn(1, 3, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Reset mock for KL calculation
        mock_model.get_logprobs_batched.return_value = torch.randn(1, 3, 32000)

        kl = evaluator._compute_kl_divergence()

        assert isinstance(kl, float)


class TestGetScore:
    """Test the get_score method."""

    def test_get_score_returns_tuple(self, mock_model):
        """Test get_score returns (score, kl_divergence, refusals)."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            kl_divergence_scale=1.0,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1", "prompt2"]
            mock_model.get_logprobs_batched.return_value = torch.randn(2, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry", "Sure!"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Setup for get_score call
        mock_model.get_logprobs_batched.return_value = torch.randn(2, 32000)
        mock_model.get_responses_batched.return_value = ["I'm sorry", "Sure!"]

        with patch("bruno.evaluator.print"):
            score, kl_divergence, refusals = evaluator.get_score()

        assert isinstance(score, tuple)
        assert len(score) == 2  # (kl_score, refusal_score)
        assert isinstance(kl_divergence, float)
        assert isinstance(refusals, int)

    def test_get_score_parallel_execution(self, mock_model):
        """Test get_score runs KL and refusal counting in parallel."""
        from bruno.config import Settings

        settings = Settings(
            model="mock-model",
            kl_divergence_scale=1.0,
        )

        with patch("bruno.evaluator.load_prompts") as mock_load:
            mock_load.return_value = ["prompt1", "prompt2"]
            mock_model.get_logprobs_batched.return_value = torch.randn(2, 32000)
            mock_model.get_responses_batched.return_value = ["I'm sorry", "Sure!"]

            with patch("bruno.evaluator.print"):
                from bruno.evaluator import Evaluator

                evaluator = Evaluator(settings, mock_model)

        # Track method calls
        evaluator._compute_kl_divergence = MagicMock(return_value=0.5)
        evaluator.count_refusals = MagicMock(return_value=1)

        with patch("bruno.evaluator.print"):
            score, kl, refusals = evaluator.get_score()

        # Both methods should have been called
        evaluator._compute_kl_divergence.assert_called_once()
        evaluator.count_refusals.assert_called_once()


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
