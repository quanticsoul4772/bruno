# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for config validators.

Tests the Pydantic validators added to config.py for range validation.
"""

import pytest
from pydantic import ValidationError

from heretic.config import LayerRangeProfileConfig, Settings


class TestLayerRangeProfileConfig:
    """Test LayerRangeProfileConfig validation."""

    def test_valid_profile(self):
        """Test that valid profiles pass validation."""
        profile = LayerRangeProfileConfig(
            range_start=0.0,
            range_end=0.4,
            weight_multiplier=0.5,
        )
        assert profile.range_start == 0.0
        assert profile.range_end == 0.4
        assert profile.weight_multiplier == 0.5

    def test_edge_case_full_range(self):
        """Test profile covering full layer range."""
        profile = LayerRangeProfileConfig(
            range_start=0.0,
            range_end=1.0,
            weight_multiplier=1.0,
        )
        assert profile.range_end == 1.0


class TestSettingsValidators:
    """Test Settings field validators.

    Note: Full Settings validation requires a model name, which we mock.
    """

    def test_n_trials_must_be_positive(self):
        """Test that n_trials > 0 is enforced."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                n_trials=0,
            )
        assert "n_trials must be greater than 0" in str(exc_info.value)

    def test_n_trials_negative_fails(self):
        """Test that negative n_trials fails."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                n_trials=-5,
            )
        assert "n_trials must be greater than 0" in str(exc_info.value)

    def test_batch_size_non_negative(self):
        """Test that batch_size >= 0 is enforced."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                batch_size=-1,
            )
        assert "batch_size must be >= 0" in str(exc_info.value)

    def test_batch_size_zero_valid(self):
        """Test that batch_size=0 (auto) is valid."""
        # This should not raise
        settings = Settings(
            model="test-model",
            batch_size=0,
        )
        assert settings.batch_size == 0

    def test_sacred_overlap_threshold_range(self):
        """Test sacred_overlap_threshold must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                sacred_overlap_threshold=1.5,
            )
        assert "sacred_overlap_threshold must be between 0.0 and 1.0" in str(
            exc_info.value
        )

    def test_sacred_overlap_threshold_negative_fails(self):
        """Test negative sacred_overlap_threshold fails."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                sacred_overlap_threshold=-0.1,
            )
        assert "sacred_overlap_threshold must be between 0.0 and 1.0" in str(
            exc_info.value
        )

    def test_activation_target_percentile_range(self):
        """Test activation_target_percentile must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                activation_target_percentile=2.0,
            )
        assert "activation_target_percentile must be between 0.0 and 1.0" in str(
            exc_info.value
        )

    def test_min_probe_accuracy_range(self):
        """Test min_probe_accuracy must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                min_probe_accuracy=1.5,
            )
        assert "min_probe_accuracy must be between 0.0 and 1.0" in str(exc_info.value)

    def test_neural_detection_threshold_range(self):
        """Test neural_detection_threshold must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                neural_detection_threshold=-0.5,
            )
        assert "neural_detection_threshold must be between 0.0 and 1.0" in str(
            exc_info.value
        )

    def test_caa_addition_strength_range(self):
        """Test caa_addition_strength must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                caa_addition_strength=1.5,
            )
        assert "caa_addition_strength must be between 0.0 and 1.0" in str(
            exc_info.value
        )

    def test_caa_max_overlap_range(self):
        """Test caa_max_overlap must be 0.0-1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                caa_max_overlap=2.0,
            )
        assert "caa_max_overlap must be between 0.0 and 1.0" in str(exc_info.value)

    def test_n_sacred_directions_positive(self):
        """Test n_sacred_directions must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                n_sacred_directions=0,
            )
        assert "n_sacred_directions must be greater than 0" in str(exc_info.value)

    def test_n_refusal_directions_positive(self):
        """Test n_refusal_directions must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                n_refusal_directions=0,
            )
        assert "n_refusal_directions must be greater than 0" in str(exc_info.value)

    def test_n_concept_cones_positive(self):
        """Test n_concept_cones must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                n_concept_cones=0,
            )
        assert "n_concept_cones must be greater than 0" in str(exc_info.value)

    def test_min_cone_size_positive(self):
        """Test min_cone_size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                min_cone_size=0,
            )
        assert "min_cone_size must be greater than 0" in str(exc_info.value)

    def test_iterative_rounds_at_least_one(self):
        """Test iterative_rounds must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                iterative_rounds=0,
            )
        assert "iterative_rounds must be at least 1" in str(exc_info.value)

    def test_kl_divergence_tokens_positive(self):
        """Test kl_divergence_tokens must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                kl_divergence_tokens=0,
            )
        assert "kl_divergence_tokens must be greater than 0" in str(exc_info.value)

    def test_ensemble_weights_must_sum_to_one(self):
        """Test that ensemble weights must sum to 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                ensemble_weight_probe=0.5,
                ensemble_weight_pca=0.3,  # Sum = 0.8, not 1.0
            )
        assert "should sum to 1.0" in str(exc_info.value)

    def test_ensemble_weights_valid(self):
        """Test that valid ensemble weights pass."""
        settings = Settings(
            model="test-model",
            ensemble_weight_probe=0.6,
            ensemble_weight_pca=0.4,
        )
        assert settings.ensemble_weight_probe == 0.6
        assert settings.ensemble_weight_pca == 0.4

    def test_valid_layer_profiles(self):
        """Test that valid layer profiles pass validation."""
        settings = Settings(
            model="test-model",
            layer_range_profiles=[
                LayerRangeProfileConfig(
                    range_start=0.0, range_end=0.4, weight_multiplier=0.5
                ),
                LayerRangeProfileConfig(
                    range_start=0.4, range_end=0.7, weight_multiplier=1.0
                ),
                LayerRangeProfileConfig(
                    range_start=0.7, range_end=1.0, weight_multiplier=0.8
                ),
            ],
        )
        assert len(settings.layer_range_profiles) == 3

    def test_layer_profile_invalid_range_order(self):
        """Test that start >= end fails."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                layer_range_profiles=[
                    LayerRangeProfileConfig(
                        range_start=0.5, range_end=0.4, weight_multiplier=1.0
                    ),
                ],
            )
        assert "range_start" in str(exc_info.value) and "range_end" in str(
            exc_info.value
        )

    def test_layer_profile_negative_weight(self):
        """Test that negative weight_multiplier fails."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                layer_range_profiles=[
                    LayerRangeProfileConfig(
                        range_start=0.0, range_end=0.4, weight_multiplier=-0.5
                    ),
                ],
            )
        assert "weight_multiplier must be non-negative" in str(exc_info.value)

    def test_layer_profile_out_of_range_start(self):
        """Test that range_start > 1.0 fails."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                model="test-model",
                layer_range_profiles=[
                    LayerRangeProfileConfig(
                        range_start=1.5, range_end=2.0, weight_multiplier=1.0
                    ),
                ],
            )
        assert "range_start must be between 0.0 and 1.0" in str(exc_info.value)
