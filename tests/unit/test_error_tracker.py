"""Unit tests for the error tracker module."""

import pytest

from heretic.error_tracker import (
    ErrorSeverity,
    ErrorTracker,
    SuppressedError,
    error_tracker,
    get_error_tracker,
    record_suppressed_error,
)


class TestSuppressedError:
    """Tests for the SuppressedError dataclass."""

    def test_creation(self):
        """Test creating a SuppressedError."""
        error = SuppressedError(
            error_type="ValueError",
            error_message="test error",
            context="test_context",
            module="test_module",
            severity=ErrorSeverity.WARNING,
        )
        assert error.error_type == "ValueError"
        assert error.error_message == "test error"
        assert error.context == "test_context"
        assert error.module == "test_module"
        assert error.severity == ErrorSeverity.WARNING
        assert error.timestamp > 0
        assert error.details == {}
        assert error.traceback_str is None

    def test_to_dict(self):
        """Test converting SuppressedError to dictionary."""
        error = SuppressedError(
            error_type="ValueError",
            error_message="test",
            context="ctx",
            module="mod",
            severity=ErrorSeverity.ERROR,
            details={"key": "value"},
        )
        d = error.to_dict()
        assert d["error_type"] == "ValueError"
        assert d["severity"] == "error"
        assert d["details"] == {"key": "value"}

    def test_str_representation(self):
        """Test string representation of SuppressedError."""
        error = SuppressedError(
            error_type="RuntimeError",
            error_message="something failed",
            context="operation",
            module="mymodule",
            severity=ErrorSeverity.CRITICAL,
        )
        s = str(error)
        assert "CRITICAL" in s
        assert "RuntimeError" in s
        assert "something failed" in s


class TestErrorTracker:
    """Tests for the ErrorTracker class."""

    @pytest.fixture(autouse=True)
    def clear_tracker(self):
        """Clear the error tracker before and after each test."""
        error_tracker.clear()
        yield
        error_tracker.clear()

    def test_singleton(self):
        """Test that ErrorTracker is a singleton."""
        tracker1 = ErrorTracker()
        tracker2 = ErrorTracker()
        assert tracker1 is tracker2

    def test_get_error_tracker(self):
        """Test the get_error_tracker function."""
        tracker = get_error_tracker()
        assert tracker is error_tracker

    def test_record_with_exception(self):
        """Test recording an actual exception."""
        try:
            raise ValueError("test error message")
        except ValueError as e:
            record = error_tracker.record(
                error=e,
                context="test_operation",
                module="test",
            )

        assert record.error_type == "ValueError"
        assert record.error_message == "test error message"
        assert error_tracker.count() == 1

    def test_record_without_exception(self):
        """Test recording without an exception."""
        record = error_tracker.record(
            error=None,
            context="soft_error",
            module="test",
            severity="info",
        )
        assert record.error_type == "Unknown"
        assert record.severity == ErrorSeverity.INFO

    def test_record_with_string_severity(self):
        """Test recording with string severity."""
        record = error_tracker.record(
            error=None,
            context="test",
            module="test",
            severity="critical",
        )
        assert record.severity == ErrorSeverity.CRITICAL

    def test_record_with_details(self):
        """Test recording with additional details."""
        record = error_tracker.record(
            error=None,
            context="test",
            module="test",
            details={"key1": "value1", "count": 42},
        )
        assert record.details["key1"] == "value1"
        assert record.details["count"] == 42

    def test_record_with_traceback(self):
        """Test recording with traceback capture."""
        try:
            raise RuntimeError("traceback test")
        except RuntimeError as e:
            record = error_tracker.record(
                error=e,
                context="test",
                module="test",
                include_traceback=True,
            )

        assert record.traceback_str is not None
        assert "RuntimeError" in record.traceback_str
        assert "traceback test" in record.traceback_str

    def test_get_all(self):
        """Test getting all errors."""
        error_tracker.record(error=None, context="c1", module="m1")
        error_tracker.record(error=None, context="c2", module="m2")
        error_tracker.record(error=None, context="c3", module="m3")

        all_errors = error_tracker.get_all()
        assert len(all_errors) == 3
        # Verify it's a copy
        all_errors.clear()
        assert error_tracker.count() == 3

    def test_get_by_severity(self):
        """Test filtering by severity."""
        error_tracker.record(error=None, context="c1", module="m", severity="warning")
        error_tracker.record(error=None, context="c2", module="m", severity="error")
        error_tracker.record(error=None, context="c3", module="m", severity="warning")

        warnings = error_tracker.get_by_severity("warning")
        assert len(warnings) == 2
        errors = error_tracker.get_by_severity(ErrorSeverity.ERROR)
        assert len(errors) == 1

    def test_get_by_module(self):
        """Test filtering by module."""
        error_tracker.record(error=None, context="c", module="module_a")
        error_tracker.record(error=None, context="c", module="module_b")
        error_tracker.record(error=None, context="c", module="module_a")

        module_a_errors = error_tracker.get_by_module("module_a")
        assert len(module_a_errors) == 2

    def test_get_by_context(self):
        """Test filtering by context."""
        error_tracker.record(error=None, context="context_x", module="m")
        error_tracker.record(error=None, context="context_y", module="m")
        error_tracker.record(error=None, context="context_x", module="m")

        context_x_errors = error_tracker.get_by_context("context_x")
        assert len(context_x_errors) == 2

    def test_get_recent(self):
        """Test getting recent errors."""
        for i in range(10):
            error_tracker.record(error=None, context=f"c{i}", module="m")

        recent = error_tracker.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].context == "c9"
        assert recent[-2].context == "c8"
        assert recent[-3].context == "c7"

    def test_count(self):
        """Test counting errors."""
        assert error_tracker.count() == 0
        error_tracker.record(error=None, context="c", module="m")
        assert error_tracker.count() == 1
        error_tracker.record(error=None, context="c", module="m")
        assert error_tracker.count() == 2

    def test_count_by_severity(self):
        """Test counting by severity."""
        error_tracker.record(error=None, context="c", module="m", severity="debug")
        error_tracker.record(error=None, context="c", module="m", severity="warning")
        error_tracker.record(error=None, context="c", module="m", severity="warning")
        error_tracker.record(error=None, context="c", module="m", severity="error")

        counts = error_tracker.count_by_severity()
        assert counts["debug"] == 1
        assert counts["warning"] == 2
        assert counts["error"] == 1
        assert counts["info"] == 0

    def test_get_summary_empty(self):
        """Test summary when no errors recorded."""
        summary = error_tracker.get_summary()
        assert "No suppressed errors" in summary

    def test_get_summary_with_errors(self):
        """Test summary with recorded errors."""
        error_tracker.record(
            error=None, context="c1", module="mod1", severity="warning"
        )
        error_tracker.record(error=None, context="c2", module="mod2", severity="error")

        summary = error_tracker.get_summary()
        assert "2 total" in summary
        assert "WARNING" in summary
        assert "ERROR" in summary
        assert "mod1" in summary
        assert "mod2" in summary

    def test_clear(self):
        """Test clearing all errors."""
        error_tracker.record(error=None, context="c", module="m")
        error_tracker.record(error=None, context="c", module="m")
        assert error_tracker.count() == 2

        cleared = error_tracker.clear()
        assert cleared == 2
        assert error_tracker.count() == 0

    def test_export_json(self):
        """Test exporting errors as JSON-serializable list."""
        error_tracker.record(
            error=None,
            context="test",
            module="test",
            severity="warning",
            details={"key": "value"},
        )

        exported = error_tracker.export_json()
        assert len(exported) == 1
        assert exported[0]["context"] == "test"
        assert exported[0]["severity"] == "warning"
        assert exported[0]["details"] == {"key": "value"}

    def test_max_errors_limit(self):
        """Test that the tracker limits stored errors."""
        # Record more errors than the limit
        original_limit = error_tracker._max_errors
        error_tracker._max_errors = 10

        try:
            for i in range(20):
                error_tracker.record(error=None, context=f"c{i}", module="m")

            # Should only keep the last 10
            assert error_tracker.count() == 10
            # And they should be the most recent ones
            errors = error_tracker.get_all()
            assert errors[0].context == "c10"
            assert errors[-1].context == "c19"
        finally:
            error_tracker._max_errors = original_limit


class TestRecordSuppressedError:
    """Tests for the convenience function."""

    @pytest.fixture(autouse=True)
    def clear_tracker(self):
        """Clear the error tracker before and after each test."""
        error_tracker.clear()
        yield
        error_tracker.clear()

    def test_record_suppressed_error_function(self):
        """Test the convenience function."""
        try:
            raise KeyError("missing key")
        except KeyError as e:
            record = record_suppressed_error(
                error=e,
                context="dict_access",
                module="mymodule",
                severity="info",
                details={"key_name": "foo"},
            )

        assert record.error_type == "KeyError"
        assert record.context == "dict_access"
        assert record.module == "mymodule"
        assert record.severity == ErrorSeverity.INFO
        assert record.details["key_name"] == "foo"
        assert error_tracker.count() == 1
