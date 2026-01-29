# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for structured logging module."""

import os
from unittest.mock import patch


class TestStructuredLoggingConfig:
    """Test logging configuration."""
    
    def test_is_structured_logging_enabled_default_false(self):
        """Test structured logging is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            from heretic.logging import is_structured_logging_enabled
            
            assert is_structured_logging_enabled() is False
    
    def test_is_structured_logging_enabled_true(self):
        """Test structured logging enabled via env var."""
        for value in ["true", "TRUE", "1", "yes", "YES"]:
            with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": value}):
                from heretic.logging import is_structured_logging_enabled
                
                assert is_structured_logging_enabled() is True
    
    def test_is_structured_logging_enabled_false(self):
        """Test structured logging disabled via env var."""
        for value in ["false", "FALSE", "0", "no", ""]:
            with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": value}):
                from heretic.logging import is_structured_logging_enabled
                
                assert is_structured_logging_enabled() is False


class TestGetLogger:
    """Test logger retrieval."""
    
    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        with patch.dict(os.environ, {"HERETIC_STRUCTURED_LOGGING": "false"}):
            from heretic.logging import get_logger
            import logging
            
            logger = get_logger("test.module")
            
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test.module"


class TestSetupLogging:
    """Test logging setup."""
    
    def test_setup_logging_standard(self):
        """Test setup_logging configures standard logging."""
        from heretic.logging import setup_logging
        
        # Should not raise
        setup_logging(enable_structured=False)
    
    def test_setup_logging_structured_without_structlog(self):
        """Test setup_logging handles missing structlog gracefully."""
        from heretic.logging import setup_logging
        
        with patch("heretic.logging.STRUCTLOG_AVAILABLE", False):
            # Should fall back to standard logging without raising
            setup_logging(enable_structured=True)
