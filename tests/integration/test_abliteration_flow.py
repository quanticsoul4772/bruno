# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Integration tests for the full abliteration workflow.

These tests are marked slow and require GPU access.
They are intended for nightly CI runs, not PR checks.
"""
import pytest


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.gpu
class TestAbliterationFlow:
    """Test the full abliteration workflow."""

    def test_abliteration_with_small_model(self):
        """Test abliteration with a small model (gpt2)."""
        pytest.skip("Integration test - requires GPU and model download")
        # Future implementation:
        # 1. Load gpt2 (small, ~500MB)
        # 2. Run abliteration with n_trials=5
        # 3. Verify model weights are modified
        # 4. Verify refusal count changes

    def test_model_save_and_load(self):
        """Test that abliterated models can be saved and loaded."""
        pytest.skip("Integration test - requires GPU and model download")
        # Future implementation:
        # 1. Abliterate a model
        # 2. Save to disk
        # 3. Load from disk
        # 4. Verify the loaded model has same behavior

    def test_optuna_persistence(self):
        """Test that Optuna studies can be resumed."""
        pytest.skip("Integration test - requires GPU")
        # Future implementation:
        # 1. Run 5 trials with storage
        # 2. Stop
        # 3. Resume with same storage
        # 4. Verify trials continue from 6
