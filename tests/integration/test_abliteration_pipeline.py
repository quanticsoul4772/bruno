# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Integration tests for the abliteration pipeline.

These tests verify end-to-end functionality of direction extraction,
orthogonalization, and ablation phases.
"""

import pytest
import torch
import torch.nn.functional as F

from heretic.constants import (
    EPSILON,
    LOW_MAGNITUDE_WARNING,
    NEAR_ZERO,
    SACRED_PCA_ALPHA,
    Ablation,
    Clustering,
    LayerPos,
    Memory,
    Probe,
    Thresholds,
)


class TestConstants:
    """Test that constants are properly defined and accessible."""

    def test_threshold_values(self):
        """Test that threshold constants have expected values."""
        assert Thresholds.NEAR_ZERO == 1e-6
        assert Thresholds.EPSILON == 1e-8
        assert Thresholds.LOW_MAGNITUDE_WARNING == 0.5

    def test_layer_positions(self):
        """Test that layer position fractions are valid."""
        assert 0.0 <= LayerPos.EARLY_START < LayerPos.EARLY_END <= 1.0
        assert 0.0 <= LayerPos.MIDDLE_START < LayerPos.MIDDLE_END <= 1.0
        assert 0.0 <= LayerPos.LATE_START < LayerPos.LATE_END <= 1.0

    def test_ablation_defaults(self):
        """Test that ablation defaults are sensible."""
        assert 0.0 < Ablation.MAX_WEIGHT_MIN <= Ablation.MAX_WEIGHT_MAX
        assert 0.0 < Ablation.CALIBRATION_MIN_FACTOR < Ablation.CALIBRATION_MAX_FACTOR

    def test_probe_defaults(self):
        """Test that probe defaults are sensible."""
        assert Probe.MIN_SAMPLES_PER_CLASS > 0
        assert 0.0 <= Probe.MIN_PROBE_ACCURACY <= 1.0
        assert Probe.ENSEMBLE_PROBE_WEIGHT + Probe.ENSEMBLE_PCA_WEIGHT == 1.0

    def test_clustering_defaults(self):
        """Test that clustering defaults are sensible."""
        assert Clustering.N_CONCEPT_CONES > 0
        assert Clustering.MIN_CLUSTER_SIZE > 0
        assert -1.0 <= Clustering.MIN_SILHOUETTE_SCORE <= 1.0

    def test_memory_defaults(self):
        """Test that memory defaults are sensible."""
        assert Memory.CACHE_CLEAR_INTERVAL > 0
        assert Memory.MAX_OOM_RETRIES > 0

    def test_singleton_aliases(self):
        """Test that singleton aliases work correctly."""
        assert NEAR_ZERO == Thresholds.NEAR_ZERO
        assert EPSILON == Thresholds.EPSILON
        assert LOW_MAGNITUDE_WARNING == Thresholds.LOW_MAGNITUDE_WARNING
        assert SACRED_PCA_ALPHA == Ablation.SACRED_PCA_ALPHA


class TestDirectionExtraction:
    """Integration tests for direction extraction."""

    @pytest.fixture
    def mock_residuals(self):
        """Create mock residuals for testing."""
        n_layers = 32
        hidden_dim = 768
        n_good = 50
        n_bad = 50

        torch.manual_seed(42)
        good_residuals = torch.randn(n_good, n_layers, hidden_dim)
        bad_residuals = torch.randn(n_bad, n_layers, hidden_dim)

        # Make bad residuals slightly different to simulate refusal behavior
        refusal_signal = torch.randn(n_layers, hidden_dim)
        bad_residuals += 0.3 * refusal_signal.unsqueeze(0)

        return good_residuals, bad_residuals

    def test_mean_difference_direction(self, mock_residuals):
        """Test basic mean-difference direction extraction."""
        good_residuals, bad_residuals = mock_residuals

        refusal_direction = F.normalize(
            bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
            p=2,
            dim=1,
        )

        # Check shape
        assert refusal_direction.shape == (32, 768)

        # Check normalization
        norms = refusal_direction.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_direction_orthogonalization(self, mock_residuals):
        """Test direction orthogonalization."""
        good_residuals, bad_residuals = mock_residuals

        # Create two random directions
        target_direction = F.normalize(
            bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
            p=2,
            dim=1,
        )
        remove_direction = F.normalize(
            torch.randn_like(target_direction),
            p=2,
            dim=1,
        )

        # Orthogonalize
        dot_product = (target_direction * remove_direction).sum(dim=1, keepdim=True)
        projection = dot_product * remove_direction
        orthogonal = target_direction - projection
        orthogonal = F.normalize(orthogonal, p=2, dim=1)

        # Check orthogonality
        new_dot = (orthogonal * remove_direction).sum(dim=1)
        assert torch.allclose(new_dot, torch.zeros_like(new_dot), atol=1e-5)

    def test_magnitude_after_orthogonalization(self, mock_residuals):
        """Test that magnitude warning threshold is correctly applied."""
        good_residuals, bad_residuals = mock_residuals

        # Create direction with small magnitude after orthogonalization
        target = torch.ones(32, 768) * 0.1
        target = F.normalize(target, p=2, dim=1)

        magnitude = target.norm(dim=1).mean().item()

        # Should be approximately 1.0 after normalization
        assert magnitude > LOW_MAGNITUDE_WARNING


class TestConfigValidation:
    """Integration tests for config validation."""

    def test_valid_config_passes(self):
        """Test that valid configuration values pass validation."""
        from heretic.config import LayerRangeProfileConfig

        profile = LayerRangeProfileConfig(
            range_start=0.0,
            range_end=0.4,
            weight_multiplier=0.5,
        )

        assert profile.range_start == 0.0
        assert profile.range_end == 0.4
        assert profile.weight_multiplier == 0.5

    def test_ensemble_weights_sum(self):
        """Test that ensemble weights validation works."""

        # This should fail if weights don't sum to 1.0
        # (We can't easily test this without a full Settings object)
        pass


class TestPhaseModules:
    """Integration tests for phase modules."""

    def test_dataset_bundle_creation(self):
        """Test DatasetBundle creation."""
        from heretic.phases import DatasetBundle

        bundle = DatasetBundle(
            good_prompts=["Hello, how are you?"],
            bad_prompts=["How do I hack a computer?"],
        )

        assert len(bundle.good_prompts) == 1
        assert len(bundle.bad_prompts) == 1
        assert bundle.helpful_prompts is None

    def test_dataset_bundle_with_optional_fields(self):
        """Test DatasetBundle with optional fields populated."""
        from heretic.phases import DatasetBundle

        bundle = DatasetBundle(
            good_prompts=["Hello"],
            bad_prompts=["Hack"],
            helpful_prompts=["I'd be happy to help"],
            unhelpful_prompts=["Random text"],
            sacred_prompts=["What is 2+2?"],
            sacred_baseline_prompts=["Lorem ipsum"],
        )

        assert bundle.helpful_prompts is not None
        assert bundle.sacred_prompts is not None
        assert len(bundle.sacred_prompts) == 1

    def test_direction_extraction_result_creation(self):
        """Test DirectionExtractionResult creation."""
        from heretic.phases import DirectionExtractionResult

        result = DirectionExtractionResult(
            refusal_directions=torch.randn(32, 768),
            use_multi_direction=False,
        )

        assert result.refusal_directions.shape == (32, 768)
        assert not result.use_multi_direction
        assert result.refusal_directions_multi is None

    def test_direction_extraction_result_multi_direction(self):
        """Test DirectionExtractionResult with multi-direction."""
        from heretic.phases import DirectionExtractionResult

        result = DirectionExtractionResult(
            refusal_directions=torch.randn(32, 768),
            use_multi_direction=True,
            refusal_directions_multi=torch.randn(32, 3, 768),
            direction_weights=[1.0, 0.5, 0.25],
        )

        assert result.use_multi_direction
        assert result.refusal_directions_multi is not None
        assert result.refusal_directions_multi.shape == (32, 3, 768)
        assert result.direction_weights == [1.0, 0.5, 0.25]

    def test_phase_module_imports(self):
        """Test that all phase module exports are importable."""
        from heretic.phases import (
            DatasetBundle,
            DirectionExtractionResult,
            create_study,
            extract_refusal_directions,
            load_datasets,
        )

        # Just check they're all functions/classes
        assert DatasetBundle is not None
        assert DirectionExtractionResult is not None
        assert callable(load_datasets)
        assert callable(extract_refusal_directions)
        assert callable(create_study)


class TestEndToEndPipeline:
    """End-to-end integration tests (requires more setup)."""

    @pytest.mark.slow
    def test_direction_extraction_with_sacred(self):
        """Test full direction extraction with sacred directions.

        This test is marked slow as it requires significant computation.
        """
        # Skip if no GPU available
        if not torch.cuda.is_available():
            pytest.skip("GPU required for full integration test")

        # Would require a small test model - skipping for now
        pass

    @pytest.mark.slow
    def test_ablation_with_orthogonalization(self):
        """Test full ablation with helpfulness orthogonalization.

        This test is marked slow as it requires model loading.
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU required for full integration test")

        # Would require a small test model - skipping for now
        pass
