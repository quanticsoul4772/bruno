# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for the constants module.

Tests that all constants are properly defined and have sensible values.
"""

import pytest

from heretic.constants import (
    CACHE_CLEAR_INTERVAL,
    EPSILON,
    LOW_MAGNITUDE_WARNING,
    MAX_OOM_RETRIES,
    # Direct exports
    NEAR_ZERO,
    SACRED_PCA_ALPHA,
    Ablation,
    Circuit,
    Clustering,
    LayerPos,
    Memory,
    Probe,
    Retry,
    # Singleton instances
    Thresholds,
    Validation,
    Weights,
)


class TestNumericalThresholds:
    """Test NumericalThresholds constants."""

    def test_near_zero_is_small(self):
        """Test NEAR_ZERO is appropriately small."""
        assert Thresholds.NEAR_ZERO == 1e-6
        assert Thresholds.NEAR_ZERO > 0
        assert Thresholds.NEAR_ZERO < 1e-4

    def test_epsilon_is_smaller(self):
        """Test EPSILON is smaller than NEAR_ZERO."""
        assert Thresholds.EPSILON == 1e-8
        assert Thresholds.EPSILON < Thresholds.NEAR_ZERO

    def test_low_magnitude_warning(self):
        """Test LOW_MAGNITUDE_WARNING is reasonable."""
        assert Thresholds.LOW_MAGNITUDE_WARNING == 0.5
        assert 0 < Thresholds.LOW_MAGNITUDE_WARNING < 1

    def test_float_tolerance(self):
        """Test FLOAT_TOLERANCE is reasonable for comparisons."""
        assert Thresholds.FLOAT_TOLERANCE == 1e-4


class TestLayerPositions:
    """Test LayerPositions constants."""

    def test_layer_ranges_are_valid(self):
        """Test layer range fractions are valid."""
        assert 0 <= LayerPos.EARLY_START < LayerPos.EARLY_END <= 1
        assert 0 <= LayerPos.MIDDLE_START < LayerPos.MIDDLE_END <= 1
        assert 0 <= LayerPos.LATE_START < LayerPos.LATE_END <= 1

    def test_layer_ranges_are_consecutive(self):
        """Test default layer ranges are consecutive."""
        assert LayerPos.EARLY_END == LayerPos.MIDDLE_START
        assert LayerPos.MIDDLE_END == LayerPos.LATE_START

    def test_layer_ranges_cover_full_range(self):
        """Test layer ranges cover 0.0 to 1.0."""
        assert LayerPos.EARLY_START == 0.0
        assert LayerPos.LATE_END == 1.0

    def test_weight_multipliers_are_positive(self):
        """Test all weight multipliers are positive."""
        assert LayerPos.EARLY_WEIGHT_MULTIPLIER > 0
        assert LayerPos.MIDDLE_WEIGHT_MULTIPLIER > 0
        assert LayerPos.LATE_WEIGHT_MULTIPLIER > 0

    def test_direction_index_range(self):
        """Test direction index range is within valid bounds."""
        assert 0 < LayerPos.DIRECTION_INDEX_MIN_FRAC < 1
        assert 0 < LayerPos.DIRECTION_INDEX_MAX_FRAC <= 1
        assert LayerPos.DIRECTION_INDEX_MIN_FRAC < LayerPos.DIRECTION_INDEX_MAX_FRAC


class TestAblationDefaults:
    """Test AblationDefaults constants."""

    def test_weight_range(self):
        """Test weight range is valid."""
        assert Ablation.MAX_WEIGHT_MIN > 0
        assert Ablation.MAX_WEIGHT_MIN <= Ablation.MAX_WEIGHT_MAX

    def test_calibration_factors(self):
        """Test calibration factor range is valid."""
        assert Ablation.CALIBRATION_MIN_FACTOR > 0
        assert Ablation.CALIBRATION_MIN_FACTOR < Ablation.CALIBRATION_MAX_FACTOR

    def test_sacred_pca_alpha(self):
        """Test sacred PCA alpha is reasonable."""
        assert 0 <= Ablation.SACRED_PCA_ALPHA <= 1

    def test_caa_defaults(self):
        """Test CAA defaults are reasonable."""
        assert 0 <= Ablation.CAA_ADDITION_STRENGTH <= 1
        assert 0 <= Ablation.CAA_MAX_OVERLAP <= 1


class TestProbeDefaults:
    """Test ProbeDefaults constants."""

    def test_min_samples(self):
        """Test minimum samples is positive."""
        assert Probe.MIN_SAMPLES_PER_CLASS > 0

    def test_cv_splits(self):
        """Test CV splits are reasonable."""
        assert Probe.MIN_CV_SPLITS >= 2
        assert Probe.MIN_CV_SPLITS <= Probe.MAX_CV_SPLITS

    def test_ensemble_weights_sum_to_one(self):
        """Test default ensemble weights sum to 1.0."""
        total = Probe.ENSEMBLE_PROBE_WEIGHT + Probe.ENSEMBLE_PCA_WEIGHT
        assert abs(total - 1.0) < 1e-6

    def test_min_probe_accuracy(self):
        """Test min probe accuracy is reasonable."""
        assert 0.5 <= Probe.MIN_PROBE_ACCURACY <= 1.0


class TestClusteringDefaults:
    """Test ClusteringDefaults constants."""

    def test_kmeans_params(self):
        """Test KMeans parameters are valid."""
        assert Clustering.N_INIT > 0
        assert Clustering.RANDOM_STATE >= 0

    def test_silhouette_score_range(self):
        """Test silhouette score threshold is valid."""
        # Silhouette score ranges from -1 to 1
        assert -1 <= Clustering.MIN_SILHOUETTE_SCORE <= 1

    def test_cluster_params(self):
        """Test cluster parameters are positive."""
        assert Clustering.MIN_CLUSTER_SIZE > 0
        assert Clustering.N_CONCEPT_CONES > 0
        assert Clustering.DIRECTIONS_PER_CONE > 0


class TestCircuitDefaults:
    """Test CircuitDefaults constants."""

    def test_n_circuits(self):
        """Test number of circuits is positive."""
        assert Circuit.N_REFUSAL_CIRCUITS > 0

    def test_importance_threshold(self):
        """Test importance threshold is valid."""
        assert 0 <= Circuit.IMPORTANCE_THRESHOLD <= 1

    def test_ablation_strength(self):
        """Test ablation strength is positive."""
        assert Circuit.CIRCUIT_ABLATION_STRENGTH > 0


class TestMemoryDefaults:
    """Test MemoryDefaults constants."""

    def test_cache_clear_interval(self):
        """Test cache clear interval is positive."""
        assert Memory.CACHE_CLEAR_INTERVAL > 0

    def test_max_oom_retries(self):
        """Test max OOM retries is positive and reasonable."""
        assert Memory.MAX_OOM_RETRIES > 0
        assert Memory.MAX_OOM_RETRIES <= 10  # Don't retry too many times

    def test_dynamo_cache_size(self):
        """Test dynamo cache size is reasonable."""
        assert Memory.DYNAMO_CACHE_SIZE_LIMIT >= 8  # Default is 8


class TestValidationDefaults:
    """Test ValidationDefaults constants."""

    def test_mmlu_params(self):
        """Test MMLU parameters are positive."""
        assert Validation.MMLU_SAMPLES_PER_CATEGORY > 0
        assert Validation.MMLU_FEW_SHOT >= 0

    def test_kl_thresholds(self):
        """Test KL thresholds are positive."""
        assert Validation.MAX_ACCEPTABLE_KL > 0

    def test_mmlu_drop_threshold(self):
        """Test MMLU drop threshold is reasonable."""
        assert 0 <= Validation.MAX_ACCEPTABLE_MMLU_DROP <= 1

    def test_refusal_check_tokens(self):
        """Test refusal check tokens is positive."""
        assert Validation.REFUSAL_CHECK_TOKENS > 0

    def test_neural_detection_threshold(self):
        """Test neural detection threshold is valid."""
        assert 0 <= Validation.NEURAL_DETECTION_THRESHOLD <= 1


class TestDirectionWeights:
    """Test DirectionWeights constants."""

    def test_default_weights(self):
        """Test default weights are valid."""
        assert len(Weights.DEFAULT_WEIGHTS) > 0
        assert all(w > 0 for w in Weights.DEFAULT_WEIGHTS)
        # First weight should be 1.0
        assert Weights.DEFAULT_WEIGHTS[0] == 1.0
        # Weights should be decreasing
        for i in range(1, len(Weights.DEFAULT_WEIGHTS)):
            assert Weights.DEFAULT_WEIGHTS[i] <= Weights.DEFAULT_WEIGHTS[i - 1]

    def test_eigenvalue_temperature(self):
        """Test eigenvalue temperature is positive."""
        assert Weights.EIGENVALUE_TEMPERATURE > 0


class TestRetryDefaults:
    """Test RetryDefaults constants."""

    def test_delays(self):
        """Test delay values are positive."""
        assert Retry.MAX_RETRY_DELAY > 0
        assert Retry.BASE_RETRY_DELAY > 0
        assert Retry.BASE_RETRY_DELAY <= Retry.MAX_RETRY_DELAY

    def test_timeouts(self):
        """Test timeout values are positive."""
        assert Retry.SSH_CONNECT_TIMEOUT > 0
        assert Retry.COMMAND_TIMEOUT > 0


class TestDirectExports:
    """Test direct export aliases."""

    def test_near_zero_alias(self):
        """Test NEAR_ZERO alias matches singleton."""
        assert NEAR_ZERO == Thresholds.NEAR_ZERO

    def test_epsilon_alias(self):
        """Test EPSILON alias matches singleton."""
        assert EPSILON == Thresholds.EPSILON

    def test_low_magnitude_warning_alias(self):
        """Test LOW_MAGNITUDE_WARNING alias matches singleton."""
        assert LOW_MAGNITUDE_WARNING == Thresholds.LOW_MAGNITUDE_WARNING

    def test_sacred_pca_alpha_alias(self):
        """Test SACRED_PCA_ALPHA alias matches singleton."""
        assert SACRED_PCA_ALPHA == Ablation.SACRED_PCA_ALPHA

    def test_max_oom_retries_alias(self):
        """Test MAX_OOM_RETRIES alias matches singleton."""
        assert MAX_OOM_RETRIES == Memory.MAX_OOM_RETRIES

    def test_cache_clear_interval_alias(self):
        """Test CACHE_CLEAR_INTERVAL alias matches singleton."""
        assert CACHE_CLEAR_INTERVAL == Memory.CACHE_CLEAR_INTERVAL


class TestImmutability:
    """Test that dataclasses are frozen (immutable)."""

    def test_thresholds_frozen(self):
        """Test Thresholds is frozen."""
        with pytest.raises(AttributeError):
            Thresholds.NEAR_ZERO = 1e-10

    def test_layer_pos_frozen(self):
        """Test LayerPos is frozen."""
        with pytest.raises(AttributeError):
            LayerPos.EARLY_START = 0.5

    def test_ablation_frozen(self):
        """Test Ablation is frozen."""
        with pytest.raises(AttributeError):
            Ablation.SACRED_PCA_ALPHA = 0.9
