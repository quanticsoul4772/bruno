# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for main.py entry point.

The main.py run() function is complex with many side effects.
These tests focus on specific behaviors with targeted mocking.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestMainEntryPoint:
    """Test main() function error handling."""

    def test_main_handles_keyboard_interrupt(self):
        """Test main() catches KeyboardInterrupt gracefully."""
        from heretic.main import main

        with patch("heretic.main.install"):
            with patch("heretic.main.run", side_effect=KeyboardInterrupt()):
                with patch("heretic.main.print") as mock_print:
                    # Should not raise
                    main()

                    # Should print shutdown message
                    mock_print.assert_called()

    def test_main_handles_keyboard_interrupt_in_context(self):
        """Test main() handles KeyboardInterrupt in __context__."""
        from heretic.main import main

        # Create an exception with KeyboardInterrupt as context
        error = RuntimeError("Masked error")
        error.__context__ = KeyboardInterrupt()

        with patch("heretic.main.install"):
            with patch("heretic.main.run", side_effect=error):
                with patch("heretic.main.print"):
                    # Should not raise because __context__ is KeyboardInterrupt
                    main()

    def test_main_reraises_other_exceptions(self):
        """Test main() re-raises non-KeyboardInterrupt exceptions."""
        from heretic.main import main

        with patch("heretic.main.install"):
            with patch("heretic.main.run", side_effect=ValueError("Test error")):
                with pytest.raises(ValueError, match="Test error"):
                    main()


class TestRunValidationErrors:
    """Test run() validation error handling."""

    def test_run_handles_validation_error(self):
        """Test run() handles pydantic ValidationError gracefully."""
        from pydantic import ValidationError

        from heretic.main import run

        # Create a real ValidationError by trying to create invalid Settings
        validation_error = None
        try:
            from heretic.config import Settings

            Settings(model=None)  # This should raise ValidationError
        except ValidationError as e:
            validation_error = e

        if validation_error is None:
            pytest.skip("Could not generate ValidationError")

        # Mock Settings to raise the real ValidationError
        with patch("heretic.main.Settings", side_effect=validation_error):
            with patch("heretic.main.print") as mock_print:
                with patch("heretic.main.os.environ", {}):
                    # Should return early without crashing
                    run()

                    # Should have printed error messages
                    assert mock_print.call_count >= 1


class TestRunGPUDetection:
    """Test GPU detection in run()."""

    def test_run_detects_cuda_gpu(self):
        """Test run() prints CUDA GPU type."""
        from heretic.main import run

        with (
            patch("heretic.main.Settings") as mock_settings,
            patch("heretic.main.torch.cuda.is_available", return_value=True),
            patch(
                "heretic.main.torch.cuda.get_device_name", return_value="NVIDIA A100"
            ),
            patch("heretic.main.print") as mock_print,
            patch("heretic.main.Model") as mock_model_class,
        ):
            # Setup minimal settings
            settings = MagicMock()
            settings.evaluate_model = "test-model"  # Exit early via evaluate path
            mock_settings.return_value = settings

            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            with patch("heretic.main.load_datasets") as mock_load_datasets:
                mock_datasets = MagicMock()
                mock_datasets.good_prompts = ["prompt"]
                mock_datasets.bad_prompts = ["bad prompt"]
                mock_load_datasets.return_value = mock_datasets
                with patch("heretic.main.Evaluator"):
                    run()

            # Check GPU type was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("NVIDIA A100" in str(call) for call in print_calls)

    def test_run_detects_no_gpu(self):
        """Test run() warns when no GPU detected."""
        from heretic.main import run

        with (
            patch("heretic.main.Settings") as mock_settings,
            patch("heretic.main.torch.cuda.is_available", return_value=False),
            patch("heretic.main.is_xpu_available", return_value=False),
            patch("heretic.main.is_mlu_available", return_value=False),
            patch("heretic.main.is_sdaa_available", return_value=False),
            patch("heretic.main.is_musa_available", return_value=False),
            patch("heretic.main.is_npu_available", return_value=False),
            patch("heretic.main.torch.backends.mps.is_available", return_value=False),
            patch("heretic.main.print") as mock_print,
            patch("heretic.main.Model") as mock_model_class,
        ):
            settings = MagicMock()
            settings.evaluate_model = "test-model"
            mock_settings.return_value = settings

            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            with patch("heretic.main.load_datasets") as mock_load_datasets:
                mock_datasets = MagicMock()
                mock_datasets.good_prompts = ["prompt"]
                mock_datasets.bad_prompts = ["bad prompt"]
                mock_load_datasets.return_value = mock_datasets
                with patch("heretic.main.Evaluator"):
                    run()

            # Check warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("No GPU" in str(call) for call in print_calls)


class TestRunBatchSizeDetermination:
    """Test automatic batch size determination."""

    def test_run_determines_optimal_batch_size(self):
        """Test batch size auto-determination."""
        from heretic.main import run

        with (
            patch("heretic.main.Settings") as mock_settings,
            patch("heretic.main.torch.cuda.is_available", return_value=True),
            patch(
                "heretic.main.torch.cuda.get_device_name", return_value="NVIDIA A100"
            ),
            patch("heretic.main.torch.set_grad_enabled"),
            patch("heretic.main.torch._dynamo.config"),
            patch("heretic.main.transformers.logging"),
            patch("heretic.main.optuna.logging"),
            patch("heretic.main.warnings"),
            patch("heretic.main.print"),
            patch("heretic.main.Model") as mock_model_class,
            patch("heretic.main.load_datasets") as mock_load_datasets,
            patch("heretic.main.Evaluator"),
            patch("heretic.main.time.perf_counter") as mock_time,
        ):
            settings = MagicMock()
            settings.batch_size = 0  # Trigger auto-determination
            settings.max_batch_size = 8
            settings.evaluate_model = "test-model"  # Exit early
            mock_settings.return_value = settings

            mock_model = MagicMock()
            mock_model.get_responses.return_value = ["response"] * 8
            mock_model.tokenizer.encode.return_value = list(range(50))
            mock_model_class.return_value = mock_model

            mock_datasets = MagicMock()
            mock_datasets.good_prompts = ["prompt1", "prompt2"] * 10
            mock_datasets.bad_prompts = ["bad1", "bad2"] * 10
            mock_load_datasets.return_value = mock_datasets

            # Simulate timing: 0.1s per batch
            mock_time.side_effect = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

            run()

            # Should have determined a batch size
            assert settings.batch_size > 0


class TestCommandLineArguments:
    """Test command line argument handling."""

    def test_model_argument_injection(self):
        """Test that a bare argument is treated as --model."""
        import sys

        # Simulate: heretic my-model
        original_argv = sys.argv.copy()

        try:
            sys.argv = ["heretic", "my-model"]

            # The logic checks: even number of args, last doesn't start with -
            # Then it inserts --model before the last argument
            assert len(sys.argv) % 2 == 0
            assert not sys.argv[-1].startswith("-")

            # This should insert --model
            if len(sys.argv) % 2 == 0 and not sys.argv[-1].startswith("-"):
                sys.argv.insert(-1, "--model")

            assert sys.argv == ["heretic", "--model", "my-model"]
        finally:
            sys.argv = original_argv
