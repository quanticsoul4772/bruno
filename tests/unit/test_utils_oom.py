# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for OOM recovery utilities."""

from unittest.mock import MagicMock, patch

import pytest


class TestBatchSizeError:
    """Test BatchSizeError exception."""

    def test_batch_size_error_is_exception(self):
        """Test BatchSizeError inherits from Exception."""
        from bruno.utils import BatchSizeError

        assert issubclass(BatchSizeError, Exception)

    def test_batch_size_error_message(self):
        """Test BatchSizeError can be raised with message."""
        from bruno.utils import BatchSizeError

        with pytest.raises(BatchSizeError, match="out of memory"):
            raise BatchSizeError("out of memory")


class TestGPUMemoryInfo:
    """Test GPU memory info utility."""

    def test_get_gpu_memory_info_no_cuda(self):
        """Test returns zeros when CUDA not available."""
        with patch("heretic.utils.torch.cuda.is_available", return_value=False):
            from bruno.utils import get_gpu_memory_info

            info = get_gpu_memory_info()

            assert info["total_gb"] == 0
            assert info["used_gb"] == 0
            assert info["free_gb"] == 0
            assert info["utilization_pct"] == 0

    def test_get_gpu_memory_info_with_cuda(self):
        """Test returns memory info when CUDA available."""
        mock_props = MagicMock()
        mock_props.total_memory = 8_000_000_000  # 8 GB

        with (
            patch("heretic.utils.torch.cuda.is_available", return_value=True),
            patch(
                "heretic.utils.torch.cuda.get_device_properties",
                return_value=mock_props,
            ),
            patch(
                "heretic.utils.torch.cuda.memory_reserved", return_value=2_000_000_000
            ),
            patch(
                "heretic.utils.torch.cuda.memory_allocated", return_value=1_500_000_000
            ),
        ):
            from bruno.utils import get_gpu_memory_info

            info = get_gpu_memory_info()

            assert info["total_gb"] == pytest.approx(8.0, rel=0.01)
            assert info["used_gb"] == pytest.approx(1.5, rel=0.01)
            assert info["free_gb"] == pytest.approx(6.0, rel=0.01)
            assert info["utilization_pct"] == pytest.approx(18.75, rel=0.01)

    def test_get_gpu_memory_info_handles_exception(self):
        """Test returns zeros on exception."""
        with (
            patch("heretic.utils.torch.cuda.is_available", return_value=True),
            patch(
                "heretic.utils.torch.cuda.get_device_properties",
                side_effect=RuntimeError("GPU error"),
            ),
        ):
            from bruno.utils import get_gpu_memory_info

            info = get_gpu_memory_info()

            assert info["total_gb"] == 0
            assert info["utilization_pct"] == 0
