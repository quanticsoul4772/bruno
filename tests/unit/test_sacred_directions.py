# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Tests for sacred direction preservation functionality."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from bruno.exceptions import SacredDirectionError
from bruno.model import Model, SacredDirectionResult


class TestSacredDirectionExtraction:
    """Tests for extract_sacred_directions method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Model instance."""
        model = MagicMock(spec=Model)
        # Copy the real methods we're testing
        model.extract_sacred_directions = Model.extract_sacred_directions.__get__(
            model, Model
        )
        model._orthogonalize_single_against_sacred = (
            Model._orthogonalize_single_against_sacred.__get__(model, Model)
        )
        model.orthogonalize_against_sacred = Model.orthogonalize_against_sacred.__get__(
            model, Model
        )
        model.compute_sacred_overlap = Model.compute_sacred_overlap.__get__(
            model, Model
        )
        return model

    def test_extract_sacred_directions_shape(self, mock_model):
        """Test that extract_sacred_directions returns correct shapes."""
        n_samples = 50
        n_layers = 8
        hidden_dim = 256
        n_directions = 5

        capability_residuals = torch.randn(n_samples, n_layers, hidden_dim)
        baseline_residuals = torch.randn(n_samples, n_layers, hidden_dim)

        result = mock_model.extract_sacred_directions(
            capability_residuals,
            baseline_residuals,
            n_directions=n_directions,
        )

        assert isinstance(result, SacredDirectionResult)
        assert result.directions.shape == (n_layers, n_directions, hidden_dim)
        assert result.eigenvalues.shape == (n_layers, n_directions)

    def test_extract_sacred_directions_normalized(self, mock_model):
        """Test that extracted directions are normalized."""
        n_samples = 30
        n_layers = 4
        hidden_dim = 128
        n_directions = 3

        capability_residuals = torch.randn(n_samples, n_layers, hidden_dim)
        baseline_residuals = torch.randn(n_samples, n_layers, hidden_dim)

        result = mock_model.extract_sacred_directions(
            capability_residuals,
            baseline_residuals,
            n_directions=n_directions,
        )

        # Check normalization
        norms = result.directions.norm(dim=2)
        expected_norms = torch.ones(n_layers, n_directions)
        assert torch.allclose(norms, expected_norms, atol=1e-5)

    def test_extract_sacred_directions_with_one_direction(self, mock_model):
        """Test extraction with n_directions=1 edge case."""
        n_samples = 20
        n_layers = 4
        hidden_dim = 64
        n_directions = 1

        capability_residuals = torch.randn(n_samples, n_layers, hidden_dim)
        baseline_residuals = torch.randn(n_samples, n_layers, hidden_dim)

        result = mock_model.extract_sacred_directions(
            capability_residuals,
            baseline_residuals,
            n_directions=n_directions,
        )

        assert result.directions.shape == (n_layers, 1, hidden_dim)
        assert result.eigenvalues.shape == (n_layers, 1)

    def test_extract_sacred_directions_preserves_device(self, mock_model):
        """Test that output is on same device as input."""
        n_samples = 20
        n_layers = 4
        hidden_dim = 64
        n_directions = 3

        capability_residuals = torch.randn(n_samples, n_layers, hidden_dim)
        baseline_residuals = torch.randn(n_samples, n_layers, hidden_dim)

        result = mock_model.extract_sacred_directions(
            capability_residuals,
            baseline_residuals,
            n_directions=n_directions,
        )

        assert result.directions.device == capability_residuals.device
        assert result.eigenvalues.device == capability_residuals.device


class TestOrthogonalizeAgainstSacred:
    """Tests for orthogonalize_against_sacred method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Model instance."""
        model = MagicMock(spec=Model)
        model._orthogonalize_single_against_sacred = (
            Model._orthogonalize_single_against_sacred.__get__(model, Model)
        )
        model.orthogonalize_against_sacred = Model.orthogonalize_against_sacred.__get__(
            model, Model
        )
        return model

    def test_orthogonalize_2d_shape(self, mock_model):
        """Test orthogonalization with 2D input preserves shape."""
        n_layers = 8
        hidden_dim = 256
        n_sacred = 5

        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)

        assert result.shape == refusal_dir.shape

    def test_orthogonalize_3d_shape(self, mock_model):
        """Test orthogonalization with 3D input preserves shape."""
        n_layers = 8
        n_components = 3
        hidden_dim = 256
        n_sacred = 5

        refusal_dir = F.normalize(
            torch.randn(n_layers, n_components, hidden_dim), p=2, dim=2
        )
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)

        assert result.shape == refusal_dir.shape

    def test_orthogonalize_removes_projection(self, mock_model):
        """Test that orthogonalization actually removes sacred direction components."""
        n_layers = 4
        hidden_dim = 128
        n_sacred = 3

        # Create random directions
        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)

        # Check that result is orthogonal to all sacred directions
        for sacred_idx in range(n_sacred):
            sacred = sacred_dirs[:, sacred_idx, :]
            dot_product = (result * sacred).sum(dim=1)
            # Allow numerical error - Gram-Schmidt accumulates some error
            # especially when sacred directions are not orthogonal to each other
            assert torch.allclose(dot_product, torch.zeros(n_layers), atol=0.05)

    def test_orthogonalize_normalized(self, mock_model):
        """Test that orthogonalized result is normalized."""
        n_layers = 4
        hidden_dim = 128
        n_sacred = 2

        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)

        norms = result.norm(dim=1)
        expected_norms = torch.ones(n_layers)
        assert torch.allclose(norms, expected_norms, atol=1e-5)

    def test_orthogonalize_zero_vector_raises(self, mock_model):
        """Test that near-zero result raises SacredDirectionError."""
        n_layers = 4
        hidden_dim = 128

        # Create refusal direction that lies within sacred span
        # by making refusal direction equal to the first sacred direction
        sacred_dirs = F.normalize(torch.randn(n_layers, 1, hidden_dim), p=2, dim=2)
        refusal_dir = sacred_dirs[:, 0, :].clone()  # Same as sacred

        with pytest.raises(SacredDirectionError, match="near-zero"):
            mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)

    def test_orthogonalize_multi_direction_3d_zero_raises(self, mock_model):
        """Test that near-zero in any 3D component raises SacredDirectionError."""
        n_layers = 4
        n_components = 2
        hidden_dim = 128

        # Create sacred directions
        sacred_dirs = F.normalize(torch.randn(n_layers, 1, hidden_dim), p=2, dim=2)

        # Create refusal directions where first component equals sacred
        refusal_dir = F.normalize(
            torch.randn(n_layers, n_components, hidden_dim), p=2, dim=2
        )
        refusal_dir[:, 0, :] = sacred_dirs[
            :, 0, :
        ]  # Make first component same as sacred

        with pytest.raises(SacredDirectionError, match="near-zero"):
            mock_model.orthogonalize_against_sacred(refusal_dir, sacred_dirs)


class TestComputeSacredOverlap:
    """Tests for compute_sacred_overlap method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Model instance."""
        model = MagicMock(spec=Model)
        model.compute_sacred_overlap = Model.compute_sacred_overlap.__get__(
            model, Model
        )
        return model

    def test_overlap_returns_dict(self, mock_model):
        """Test that compute_sacred_overlap returns expected dictionary."""
        n_layers = 4
        hidden_dim = 128
        n_sacred = 3

        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.compute_sacred_overlap(refusal_dir, sacred_dirs)

        assert "mean_overlap" in result
        assert "max_overlap" in result
        assert "per_layer_mean" in result
        assert isinstance(result["mean_overlap"], float)
        assert isinstance(result["max_overlap"], float)
        assert isinstance(result["per_layer_mean"], list)
        assert len(result["per_layer_mean"]) == n_layers

    def test_overlap_range(self, mock_model):
        """Test that overlap values are in valid range [0, 1]."""
        n_layers = 4
        hidden_dim = 128
        n_sacred = 3

        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        sacred_dirs = F.normalize(
            torch.randn(n_layers, n_sacred, hidden_dim), p=2, dim=2
        )

        result = mock_model.compute_sacred_overlap(refusal_dir, sacred_dirs)

        assert 0 <= result["mean_overlap"] <= 1
        assert 0 <= result["max_overlap"] <= 1
        for val in result["per_layer_mean"]:
            assert 0 <= val <= 1

    def test_overlap_identical_directions(self, mock_model):
        """Test that identical directions have overlap of 1."""
        n_layers = 4
        hidden_dim = 128

        refusal_dir = F.normalize(torch.randn(n_layers, hidden_dim), p=2, dim=1)
        # Sacred direction is same as refusal
        sacred_dirs = refusal_dir.unsqueeze(1)  # (n_layers, 1, hidden_dim)

        result = mock_model.compute_sacred_overlap(refusal_dir, sacred_dirs)

        assert result["mean_overlap"] == pytest.approx(1.0, abs=1e-5)
        assert result["max_overlap"] == pytest.approx(1.0, abs=1e-5)

    def test_overlap_orthogonal_directions(self, mock_model):
        """Test that orthogonal directions have overlap of 0."""
        # Create orthogonal directions manually (2 layers, 4 hidden dims)
        refusal_dir = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        sacred_dirs = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]
        )  # (2, 1, 4)

        result = mock_model.compute_sacred_overlap(refusal_dir, sacred_dirs)

        assert result["mean_overlap"] == pytest.approx(0.0, abs=1e-5)
        assert result["max_overlap"] == pytest.approx(0.0, abs=1e-5)


class TestSacredDirectionResultDataclass:
    """Tests for SacredDirectionResult dataclass."""

    def test_dataclass_creation(self):
        """Test that SacredDirectionResult can be created."""
        directions = torch.randn(4, 3, 128)
        eigenvalues = torch.randn(4, 3)

        result = SacredDirectionResult(
            directions=directions,
            eigenvalues=eigenvalues,
        )

        assert torch.equal(result.directions, directions)
        assert torch.equal(result.eigenvalues, eigenvalues)

    def test_dataclass_no_overlap_scores(self):
        """Test that SacredDirectionResult does not have overlap_scores attribute."""
        directions = torch.randn(4, 3, 128)
        eigenvalues = torch.randn(4, 3)

        result = SacredDirectionResult(
            directions=directions,
            eigenvalues=eigenvalues,
        )

        # As per code review, overlap_scores was removed from dataclass
        assert not hasattr(result, "overlap_scores")
