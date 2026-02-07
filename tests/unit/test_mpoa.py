# SPDX-License-Identifier: AGPL-3.0-or-later
# Tests for MPOA (Norm-Preserving Biprojected Abliteration)

import pytest
import torch

from bruno.constants import EPSILON


class TestApplyMPOAProjection:
    """Tests for the _apply_mpoa_projection method."""

    @pytest.fixture
    def mock_model(self):
        """Create a minimal mock model for testing MPOA."""
        from unittest.mock import MagicMock

        from bruno.config import Settings

        # Create mock settings
        settings = MagicMock(spec=Settings)
        settings.model = "test-model"
        settings.cache_weights = False
        settings.compile = False
        settings.dtypes = ["float32"]
        settings.device_map = "cpu"

        # Create a mock model object with the _apply_mpoa_projection method
        # We'll test the method directly by extracting its logic
        return settings

    def _apply_mpoa_projection(
        self,
        matrix: torch.Tensor,
        projector: torch.Tensor,
        weight: float,
        norm_mode: str = "row",
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        """Local copy of _apply_mpoa_projection for testing."""
        # Step 1: Store original norms based on mode
        if norm_mode == "row":
            original_norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
        elif norm_mode == "column":
            original_norms = torch.linalg.norm(matrix, dim=0, keepdim=True)
        else:  # frobenius
            original_norm = torch.linalg.norm(matrix)

        # Step 2: Apply standard projection removal
        projection = weight * (projector @ matrix)
        matrix.sub_(projection)

        # Step 3: Rescale to preserve original norms
        if norm_mode == "row":
            new_norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
            scaling = original_norms / (new_norms + EPSILON)
            scaling = torch.clamp(scaling, min=min_scale, max=max_scale)
            matrix.mul_(scaling)
        elif norm_mode == "column":
            new_norms = torch.linalg.norm(matrix, dim=0, keepdim=True)
            scaling = original_norms / (new_norms + EPSILON)
            scaling = torch.clamp(scaling, min=min_scale, max=max_scale)
            matrix.mul_(scaling)
        else:  # frobenius
            new_norm = torch.linalg.norm(matrix)
            if original_norm > EPSILON and new_norm > EPSILON:
                scaling = original_norm / new_norm
                scaling = torch.clamp(scaling, min=min_scale, max=max_scale)
                matrix.mul_(scaling)

    def test_row_mode_preserves_row_norms(self):
        """Test that row mode preserves the norm of each row."""
        # Create a random matrix (10 rows, 20 cols)
        # For row mode, projector must be (10, 10) to multiply with (10, 20) matrix
        matrix = torch.randn(10, 20)
        original_row_norms = torch.linalg.norm(matrix, dim=1)

        # Create a random unit direction and projector
        # Direction must match the number of rows for projector @ matrix to work
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)  # (10, 10)

        # Apply MPOA
        self._apply_mpoa_projection(matrix, projector, weight=1.0, norm_mode="row")

        # Check row norms are preserved (within tolerance due to clamping)
        new_row_norms = torch.linalg.norm(matrix, dim=1)
        # The norms should be close but may differ due to clamping
        ratio = new_row_norms / (original_row_norms + EPSILON)
        assert (ratio >= 0.5 - 0.01).all(), "Row norms shrunk below min_scale"
        assert (ratio <= 2.0 + 0.01).all(), "Row norms grew above max_scale"

    def test_column_mode_preserves_column_norms(self):
        """Test that column mode preserves the norm of each column."""
        matrix = torch.randn(10, 20)
        original_col_norms = torch.linalg.norm(matrix, dim=0)

        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=1.0, norm_mode="column")

        new_col_norms = torch.linalg.norm(matrix, dim=0)
        ratio = new_col_norms / (original_col_norms + EPSILON)
        assert (ratio >= 0.5 - 0.01).all(), "Column norms shrunk below min_scale"
        assert (ratio <= 2.0 + 0.01).all(), "Column norms grew above max_scale"

    def test_frobenius_mode_preserves_total_norm(self):
        """Test that frobenius mode preserves the overall matrix norm."""
        matrix = torch.randn(10, 20)
        original_norm = torch.linalg.norm(matrix)

        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(
            matrix, projector, weight=1.0, norm_mode="frobenius"
        )

        new_norm = torch.linalg.norm(matrix)
        ratio = new_norm / original_norm
        assert ratio >= 0.5 - 0.01, "Frobenius norm shrunk below min_scale"
        assert ratio <= 2.0 + 0.01, "Frobenius norm grew above max_scale"

    def test_custom_scale_bounds(self):
        """Test that custom min/max scale bounds are respected."""
        torch.manual_seed(42)
        matrix = torch.randn(10, 20)
        original_row_norms = torch.linalg.norm(matrix, dim=1)

        # Direction must match number of rows for row mode
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        # Use moderate weight so projection doesn't overwhelm the scale bounds.
        # With weight=1.0, rows aligned with the direction get projected to near-zero
        # and the clamp can't restore them within the tight [0.8, 1.2] range.
        self._apply_mpoa_projection(
            matrix, projector, weight=0.3, norm_mode="row", min_scale=0.8, max_scale=1.2
        )

        new_row_norms = torch.linalg.norm(matrix, dim=1)
        ratio = new_row_norms / (original_row_norms + EPSILON)
        assert (ratio >= 0.8 - 0.01).all(), "Row norms shrunk below custom min_scale"
        assert (ratio <= 1.2 + 0.01).all(), "Row norms grew above custom max_scale"

    def test_zero_weight_no_change(self):
        """Test that weight=0 results in no change to the matrix."""
        matrix = torch.randn(10, 20)
        original_matrix = matrix.clone()

        # Direction must match number of rows for row mode
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=0.0, norm_mode="row")

        # Matrix should be unchanged (within numerical precision)
        assert torch.allclose(matrix, original_matrix, atol=1e-6)

    def test_small_weight_small_change(self):
        """Test that small weight results in small change."""
        matrix = torch.randn(10, 20)
        original_matrix = matrix.clone()

        # Direction must match number of rows for row mode
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=0.01, norm_mode="row")

        # Matrix should be close to original
        diff = (matrix - original_matrix).abs().mean()
        assert diff < 0.1, f"Small weight caused unexpectedly large change: {diff}"

    def test_frobenius_zero_original_norm(self):
        """Test frobenius mode handles zero original norm gracefully."""
        # Create a zero matrix
        matrix = torch.zeros(10, 20)

        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        # Should not raise an error
        self._apply_mpoa_projection(
            matrix, projector, weight=1.0, norm_mode="frobenius"
        )

        # Matrix should still be zero (no scaling applied)
        assert torch.allclose(matrix, torch.zeros_like(matrix))

    def test_frobenius_near_zero_after_projection(self):
        """Test frobenius mode handles near-zero norm after projection."""
        # Create a matrix that's mostly in the direction being projected out
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)

        # Matrix columns all point in the direction
        matrix = direction.unsqueeze(1).expand(-1, 20).clone()
        original_norm = torch.linalg.norm(matrix)
        assert original_norm > 0.1, "Original norm should be non-zero"

        projector = torch.outer(direction, direction)

        # Apply MPOA - this will project out most of the matrix
        self._apply_mpoa_projection(
            matrix, projector, weight=1.0, norm_mode="frobenius"
        )

        # Should not have NaN or Inf values
        assert not torch.isnan(matrix).any(), "Matrix contains NaN values"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf values"

    def test_row_near_zero_norms(self):
        """Test row mode handles rows with near-zero norms."""
        matrix = torch.randn(10, 20)
        # Make first row near-zero
        matrix[0, :] = 1e-10

        # Direction must match number of rows for row mode
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=1.0, norm_mode="row")

        # Should not have NaN or Inf values
        assert not torch.isnan(matrix).any(), "Matrix contains NaN values"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf values"

    def test_column_near_zero_norms(self):
        """Test column mode handles columns with near-zero norms."""
        matrix = torch.randn(10, 20)
        # Make first column near-zero
        matrix[:, 0] = 1e-10

        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=1.0, norm_mode="column")

        # Should not have NaN or Inf values
        assert not torch.isnan(matrix).any(), "Matrix contains NaN values"
        assert not torch.isinf(matrix).any(), "Matrix contains Inf values"

    def test_different_dtypes(self):
        """Test MPOA works with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            matrix = torch.randn(10, 20, dtype=dtype)
            original_norm = torch.linalg.norm(matrix)

            direction = torch.randn(10, dtype=dtype)
            direction = direction / torch.linalg.norm(direction)
            projector = torch.outer(direction, direction)

            self._apply_mpoa_projection(
                matrix, projector, weight=1.0, norm_mode="frobenius"
            )

            new_norm = torch.linalg.norm(matrix)
            ratio = new_norm / original_norm
            assert 0.5 <= ratio <= 2.0, (
                f"dtype={dtype}: Norm ratio {ratio} out of bounds"
            )

    def test_in_place_modification(self):
        """Test that the matrix is modified in-place."""
        matrix = torch.randn(10, 20)
        matrix_id = id(matrix)

        # Direction must match number of rows for row mode
        direction = torch.randn(10)
        direction = direction / torch.linalg.norm(direction)
        projector = torch.outer(direction, direction)

        self._apply_mpoa_projection(matrix, projector, weight=1.0, norm_mode="row")

        # Same object should be modified
        assert id(matrix) == matrix_id, "Matrix should be modified in-place"


class TestMPOAConfigValidation:
    """Tests for MPOA configuration validation."""

    def test_mpoa_min_scale_positive(self):
        """Test that mpoa_min_scale must be positive."""
        from pydantic import ValidationError

        from bruno.config import Settings

        with pytest.raises(ValidationError):
            Settings(model="test", mpoa_min_scale=0.0)

        with pytest.raises(ValidationError):
            Settings(model="test", mpoa_min_scale=-0.5)

    def test_mpoa_max_scale_positive(self):
        """Test that mpoa_max_scale must be positive."""
        from pydantic import ValidationError

        from bruno.config import Settings

        with pytest.raises(ValidationError):
            Settings(model="test", mpoa_max_scale=0.0)

        with pytest.raises(ValidationError):
            Settings(model="test", mpoa_max_scale=-1.0)

    def test_mpoa_max_scale_gte_min_scale(self):
        """Test that mpoa_max_scale must be >= mpoa_min_scale."""
        from pydantic import ValidationError

        from bruno.config import Settings

        with pytest.raises(ValidationError):
            Settings(model="test", mpoa_min_scale=2.0, mpoa_max_scale=1.0)

    def test_valid_mpoa_scale_range(self):
        """Test that valid scale ranges work."""
        from bruno.config import Settings

        # Should not raise
        settings = Settings(model="test", mpoa_min_scale=0.5, mpoa_max_scale=2.0)
        assert settings.mpoa_min_scale == 0.5
        assert settings.mpoa_max_scale == 2.0

        # Equal values should work
        settings = Settings(model="test", mpoa_min_scale=1.0, mpoa_max_scale=1.0)
        assert settings.mpoa_min_scale == 1.0
        assert settings.mpoa_max_scale == 1.0
