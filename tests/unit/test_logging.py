# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for structured logging module."""

import logging
import os
from unittest.mock import patch


class TestStructuredLoggingConfig:
    """Test structured logging environment variable detection."""

    def test_is_structured_logging_enabled_default_false(self):
        """Test default is False when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            from heretic.logging import is_structured_logging_enabled

            assert is_structured_logging_enabled() is False

    def test_is_structured_logging_enabled_true(self):
        """Test returns True when env var is set to true."""
        for value in ["true", "1", "yes", "TRUE", "True", "YES"]:
            with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": value}):
                from heretic.logging import is_structured_logging_enabled

                assert is_structured_logging_enabled() is True, f"Failed for {value}"

    def test_is_structured_logging_enabled_false(self):
        """Test returns False for other values."""
        for value in ["false", "0", "no", "invalid", ""]:
            with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": value}):
                from heretic.logging import is_structured_logging_enabled

                assert is_structured_logging_enabled() is False, f"Failed for {value}"


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        with patch.dict(os.environ, {}, clear=True):
            from heretic.logging import get_logger

            logger = get_logger("test.module")

            assert logger is not None
            assert isinstance(logger, logging.Logger)

    def test_get_logger_with_structlog_disabled(self):
        """Test get_logger returns standard logger when structlog disabled."""
        with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": "false"}):
            from heretic.logging import get_logger

            logger = get_logger("test.module")

            # Should return standard logging.Logger
            assert isinstance(logger, logging.Logger)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_standard(self):
        """Test standard logging setup."""
        from heretic.logging import setup_logging

        with patch("heretic.logging.logging.basicConfig") as mock_basic:
            setup_logging(enable_structured=False)

            mock_basic.assert_called_once()

    def test_setup_logging_structured_without_structlog(self):
        """Test structured logging falls back when structlog not available."""
        from heretic.logging import setup_logging

        with patch("heretic.logging.STRUCTLOG_AVAILABLE", False):
            with patch("heretic.logging.print") as mock_print:
                with patch("heretic.logging.logging.basicConfig") as mock_basic:
                    setup_logging(enable_structured=True)

                    # Should print warning and fall back to standard logging
                    mock_print.assert_called()
                    mock_basic.assert_called_once()

    def test_setup_logging_uses_env_var(self):
        """Test setup_logging uses env var when enable_structured is None."""
        from heretic.logging import setup_logging

        with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": "false"}):
            with patch("heretic.logging.logging.basicConfig") as mock_basic:
                setup_logging(enable_structured=None)

                # Should use standard logging (env var is false)
                mock_basic.assert_called_once()

    def test_setup_logging_structured_falls_back_without_structlog(self):
        """Test structured logging falls back to standard when structlog unavailable."""
        from heretic.logging import setup_logging

        with patch("heretic.logging.STRUCTLOG_AVAILABLE", False):
            with patch("heretic.logging.logging.basicConfig") as mock_basic:
                with patch("builtins.print"):  # Suppress warning
                    setup_logging(enable_structured=True)

                    # Should fall back to standard logging
                    mock_basic.assert_called_once()


class TestLogContext:
    """Test LogContext context manager."""

    def test_log_context_stores_context(self):
        """Test LogContext stores context dict."""
        from heretic.logging import LogContext

        ctx = LogContext(trial_id=1, model="test")
        assert ctx.context == {"trial_id": 1, "model": "test"}

    def test_log_context_enter_exit_without_structlog(self):
        """Test LogContext works without structlog available."""
        from heretic.logging import LogContext

        with patch("heretic.logging.STRUCTLOG_AVAILABLE", False):
            with patch.dict(os.environ, {}, clear=True):
                # Should not raise
                with LogContext(trial_id=1) as ctx:
                    assert ctx._token is None  # No token when structlog unavailable


class TestStructlogAvailability:
    """Test STRUCTLOG_AVAILABLE flag behavior."""

    def test_structlog_available_flag_exists(self):
        """Test STRUCTLOG_AVAILABLE flag is defined."""
        from heretic.logging import STRUCTLOG_AVAILABLE

        assert isinstance(STRUCTLOG_AVAILABLE, bool)


class TestSetupStandardLogging:
    """Test _setup_standard_logging internal function."""

    def test_setup_standard_logging_format(self):
        """Test standard logging uses correct format via setup_logging."""
        from heretic.logging import setup_logging

        with patch("heretic.logging.logging.basicConfig") as mock_basic:
            with patch("heretic.logging.logging.handlers.RotatingFileHandler"):
                setup_logging(enable_structured=False, log_level=logging.DEBUG)

                mock_basic.assert_called_once()
                call_kwargs = mock_basic.call_args[1]

                assert call_kwargs["level"] == logging.DEBUG
