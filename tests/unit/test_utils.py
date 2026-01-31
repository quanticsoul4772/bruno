# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for utils.py helper functions."""

from unittest.mock import MagicMock, patch


class TestFormatDuration:
    """Test format_duration function."""

    def test_format_duration_seconds_only(self):
        """Test formatting durations under 1 minute."""
        from bruno.utils import format_duration

        assert format_duration(0) == "0s"
        assert format_duration(1) == "1s"
        assert format_duration(30) == "30s"
        assert format_duration(59) == "59s"

    def test_format_duration_minutes_and_seconds(self):
        """Test formatting durations between 1-60 minutes."""
        from bruno.utils import format_duration

        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_format_duration_hours_and_minutes(self):
        """Test formatting durations over 1 hour."""
        from bruno.utils import format_duration

        assert format_duration(3600) == "1h 0m"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(7200) == "2h 0m"
        assert format_duration(7325) == "2h 2m"  # 2*3600 + 2*60 + 5

    def test_format_duration_rounds(self):
        """Test that fractional seconds are rounded."""
        from bruno.utils import format_duration

        assert format_duration(1.4) == "1s"
        assert format_duration(1.6) == "2s"
        assert format_duration(59.9) == "1m 0s"  # rounds to 60


class TestLoadPrompts:
    """Test load_prompts function."""

    def test_load_prompts_from_huggingface(self):
        """Test loading prompts from HuggingFace dataset."""
        from bruno.config import DatasetSpecification
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="test-org/test-dataset",
            split="train",
            column="text",
        )

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(
            return_value=["prompt1", "prompt2", "prompt3"]
        )

        with patch("heretic.utils.load_dataset", return_value=mock_dataset):
            with patch("heretic.utils.Path") as mock_path:
                # Make path not exist (so it uses HuggingFace)
                mock_path.return_value.exists.return_value = False

                prompts = load_prompts(spec)

        assert prompts == ["prompt1", "prompt2", "prompt3"]

    def test_load_prompts_from_local_disk(self, tmp_path):
        """Test loading prompts from local dataset."""
        from bruno.config import DatasetSpecification
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset=str(tmp_path),
            split="train",
            column="text",
        )

        mock_dataset_dict = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=["local1", "local2"])
        mock_dataset_dict.__getitem__ = MagicMock(return_value=mock_dataset)

        with patch("heretic.utils.load_from_disk", return_value=mock_dataset_dict):
            # Create actual directory so exists() returns True
            prompts = load_prompts(spec)

        assert prompts == ["local1", "local2"]

    def test_load_prompts_c4_streaming(self):
        """Test that C4 dataset uses streaming to avoid large downloads."""

        from bruno.config import DatasetSpecification
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config="en",
            split="train[:200]",
            column="text",
        )

        # Mock streaming dataset
        mock_examples = [{"text": f"c4_prompt_{i}"} for i in range(200)]
        mock_stream = iter(mock_examples)

        # Create a mock dataset that supports iteration
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=mock_stream)

        with patch(
            "heretic.utils.load_dataset", return_value=mock_dataset
        ) as mock_load:
            with patch("heretic.utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = False

                prompts = load_prompts(spec)

        # Verify streaming was used with base split (no slice notation)
        mock_load.assert_called_once_with(
            "allenai/c4", "en", split="train", streaming=True
        )

        # Verify correct number of prompts
        assert len(prompts) == 200
        assert prompts[0] == "c4_prompt_0"
        assert prompts[199] == "c4_prompt_199"

    def test_load_prompts_c4_requires_config(self):
        """Test that C4 without config fails loudly."""
        import pytest

        from bruno.config import DatasetSpecification
        from bruno.exceptions import DatasetConfigError
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config=None,  # Missing config
            split="train[:200]",
            column="text",
        )

        with patch("heretic.utils.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            with pytest.raises(
                DatasetConfigError, match="C4 dataset requires config parameter"
            ):
                load_prompts(spec)

    def test_load_prompts_c4_requires_sample_count(self):
        """Test that C4 without sample count fails loudly."""
        import pytest

        from bruno.config import DatasetSpecification
        from bruno.exceptions import DatasetConfigError
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config="en",
            split="train",  # No sample count
            column="text",
        )

        with patch("heretic.utils.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            with pytest.raises(
                DatasetConfigError, match="C4 dataset requires explicit sample count"
            ):
                load_prompts(spec)

    def test_load_prompts_c4_stream_failure(self):
        """Test that streaming failures raise clear errors."""
        import pytest
        from requests.exceptions import ConnectionError as RequestsConnectionError

        from bruno.config import DatasetSpecification
        from bruno.exceptions import NetworkTimeoutError
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config="en",
            split="train[:200]",
            column="text",
        )

        with patch("heretic.utils.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            # Mock time.sleep to avoid delays during retry loop
            with patch("heretic.utils.time.sleep"):
                # Use requests.exceptions.ConnectionError, not builtins.ConnectionError
                with patch(
                    "heretic.utils.load_dataset",
                    side_effect=RequestsConnectionError("Network down"),
                ):
                    with pytest.raises(
                        NetworkTimeoutError, match="Network error after"
                    ):
                        load_prompts(spec)

    def test_load_prompts_c4_stream_exhausted(self):
        """Test that early stream exhaustion raises error."""
        import pytest

        from bruno.config import DatasetSpecification
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config="en",
            split="train[:200]",
            column="text",
        )

        # Mock streaming dataset with only 100 examples (less than requested 200)
        mock_examples = [{"text": f"c4_prompt_{i}"} for i in range(100)]
        mock_stream = iter(mock_examples)

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=mock_stream)

        with patch("heretic.utils.load_dataset", return_value=mock_dataset):
            with patch("heretic.utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = False

                with pytest.raises(ValueError, match="Stream exhausted early"):
                    load_prompts(spec)

    def test_load_prompts_c4_stream_iteration_error(self):
        """Test that iteration errors during streaming raise clear errors."""
        import pytest

        from bruno.config import DatasetSpecification
        from bruno.utils import load_prompts

        spec = DatasetSpecification(
            dataset="allenai/c4",
            config="en",
            split="train[:200]",
            column="text",
        )

        # Mock a dataset that fails during iteration
        def failing_iterator():
            yield {"text": "prompt_0"}
            yield {"text": "prompt_1"}
            raise ConnectionError("Network interruption")

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=failing_iterator())

        with patch("heretic.utils.load_dataset", return_value=mock_dataset):
            with patch("heretic.utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = False

                # The ConnectionError during iteration is not caught by the retry logic
                # so it propagates as-is
                with pytest.raises(ConnectionError, match="Network interruption"):
                    load_prompts(spec)


class TestParseSplitCount:
    """Test _parse_split_count helper function."""

    def test_parse_split_count_simple_slice(self):
        """Test parsing simple slice like train[:200]."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("train[:200]") == 200

    def test_parse_split_count_range_slice(self):
        """Test parsing range slice like train[400:600]."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("train[400:600]") == 200

    def test_parse_split_count_small_range(self):
        """Test parsing small range like train[100:150]."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("train[100:150]") == 50

    def test_parse_split_count_no_slice(self):
        """Test parsing split without slice notation."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("train") is None

    def test_parse_split_count_validation_split(self):
        """Test parsing validation split with slice."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("validation[:100]") == 100

    def test_parse_split_count_zero_start(self):
        """Test parsing slice starting from zero."""
        from bruno.utils import _parse_split_count

        assert _parse_split_count("train[0:50]") == 50


class TestBatchify:
    """Test batchify function."""

    def test_batchify_even_split(self):
        """Test batchify with evenly divisible items."""
        from bruno.utils import batchify

        items = [1, 2, 3, 4, 5, 6]
        batches = batchify(items, 2)

        assert batches == [[1, 2], [3, 4], [5, 6]]

    def test_batchify_uneven_split(self):
        """Test batchify with remainder items."""
        from bruno.utils import batchify

        items = [1, 2, 3, 4, 5]
        batches = batchify(items, 2)

        assert batches == [[1, 2], [3, 4], [5]]

    def test_batchify_single_batch(self):
        """Test batchify when batch_size >= len(items)."""
        from bruno.utils import batchify

        items = [1, 2, 3]
        batches = batchify(items, 10)

        assert batches == [[1, 2, 3]]

    def test_batchify_empty_list(self):
        """Test batchify with empty input."""
        from bruno.utils import batchify

        items = []
        batches = batchify(items, 5)

        assert batches == []

    def test_batchify_batch_size_one(self):
        """Test batchify with batch_size=1."""
        from bruno.utils import batchify

        items = [1, 2, 3]
        batches = batchify(items, 1)

        assert batches == [[1], [2], [3]]


class TestEmptyCache:
    """Test empty_cache function."""

    def test_empty_cache_cuda(self):
        """Test empty_cache calls CUDA cache clearing."""
        from bruno.utils import empty_cache

        with (
            patch("heretic.utils.gc.collect") as mock_gc,
            patch("heretic.utils.torch.cuda.is_available", return_value=True),
            patch("heretic.utils.torch.cuda.empty_cache") as mock_cuda_cache,
            patch("heretic.utils.is_xpu_available", return_value=False),
        ):
            empty_cache()

            # gc.collect should be called twice (before and after)
            assert mock_gc.call_count == 2
            mock_cuda_cache.assert_called_once()

    def test_empty_cache_xpu(self):
        """Test empty_cache calls XPU cache clearing."""
        from bruno.utils import empty_cache

        with (
            patch("heretic.utils.gc.collect"),
            patch("heretic.utils.torch.cuda.is_available", return_value=False),
            patch("heretic.utils.is_xpu_available", return_value=True),
            patch("heretic.utils.torch.xpu.empty_cache") as mock_xpu_cache,
        ):
            empty_cache()

            mock_xpu_cache.assert_called_once()

    def test_empty_cache_mps(self):
        """Test empty_cache calls MPS cache clearing."""
        from bruno.utils import empty_cache

        with (
            patch("heretic.utils.gc.collect"),
            patch("heretic.utils.torch.cuda.is_available", return_value=False),
            patch("heretic.utils.is_xpu_available", return_value=False),
            patch("heretic.utils.is_mlu_available", return_value=False),
            patch("heretic.utils.is_sdaa_available", return_value=False),
            patch("heretic.utils.is_musa_available", return_value=False),
            patch("heretic.utils.torch.backends.mps.is_available", return_value=True),
            patch("heretic.utils.torch.mps.empty_cache") as mock_mps_cache,
        ):
            empty_cache()

            mock_mps_cache.assert_called_once()

    def test_empty_cache_no_accelerator(self):
        """Test empty_cache with no accelerator available."""
        from bruno.utils import empty_cache

        with (
            patch("heretic.utils.gc.collect") as mock_gc,
            patch("heretic.utils.torch.cuda.is_available", return_value=False),
            patch("heretic.utils.is_xpu_available", return_value=False),
            patch("heretic.utils.is_mlu_available", return_value=False),
            patch("heretic.utils.is_sdaa_available", return_value=False),
            patch("heretic.utils.is_musa_available", return_value=False),
            patch("heretic.utils.torch.backends.mps.is_available", return_value=False),
        ):
            empty_cache()

            # gc.collect should still be called twice
            assert mock_gc.call_count == 2


class TestGetTrialParameters:
    """Test get_trial_parameters function."""

    def test_get_trial_parameters_global_direction(self):
        """Test formatting trial parameters with global direction."""
        from bruno.utils import get_trial_parameters

        mock_trial = MagicMock()
        mock_trial.user_attrs = {
            "direction_index": 15.5,
            "parameters": {
                "attn.o_proj": {
                    "max_weight": 1.0,
                    "max_weight_position": 16.0,
                    "min_weight": 0.5,
                    "min_weight_distance": 8.0,
                },
                "mlp.down_proj": {
                    "max_weight": 0.8,
                    "max_weight_position": 14.0,
                    "min_weight": 0.2,
                    "min_weight_distance": 6.0,
                },
            },
        }

        params = get_trial_parameters(mock_trial)

        assert params["direction_index"] == "15.50"
        assert params["attn.o_proj.max_weight"] == "1.00"
        assert params["attn.o_proj.max_weight_position"] == "16.00"
        assert params["mlp.down_proj.min_weight"] == "0.20"

    def test_get_trial_parameters_per_layer_direction(self):
        """Test formatting trial parameters with per-layer direction."""
        from bruno.utils import get_trial_parameters

        mock_trial = MagicMock()
        mock_trial.user_attrs = {
            "direction_index": None,  # Per-layer
            "parameters": {
                "attn.o_proj": {
                    "max_weight": 1.0,
                    "max_weight_position": 16.0,
                    "min_weight": 0.0,
                    "min_weight_distance": 8.0,
                },
            },
        }

        params = get_trial_parameters(mock_trial)

        assert params["direction_index"] == "per layer"


class TestGetReadmeIntro:
    """Test get_readme_intro function."""

    def test_get_readme_intro_generates_markdown(self):
        """Test README intro generation."""
        from bruno.utils import get_readme_intro

        mock_settings = MagicMock()
        mock_settings.model = "test-org/test-model"

        mock_trial = MagicMock()
        mock_trial.user_attrs = {
            "direction_index": 15.0,
            "parameters": {
                "attn.o_proj": {
                    "max_weight": 1.0,
                    "max_weight_position": 16.0,
                    "min_weight": 0.0,
                    "min_weight_distance": 8.0,
                },
            },
            "kl_divergence": 0.25,
            "refusals": 3,
        }

        bad_prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
        base_refusals = 5

        intro = get_readme_intro(mock_settings, mock_trial, base_refusals, bad_prompts)

        # Should contain expected sections
        assert "decensored version" in intro
        assert "test-org/test-model" in intro
        assert "Heretic" in intro
        assert "Abliteration parameters" in intro
        assert "Performance" in intro
        assert "KL divergence" in intro
        assert "0.25" in intro
        assert "3/5" in intro  # refusals
        assert "5/5" in intro  # base refusals
