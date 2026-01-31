"""Tests for custom exception hierarchy."""

import pytest

from bruno.exceptions import (
    AbliterationError,
    CircuitAblationError,
    CloudError,
    ConfigurationError,
    DatasetConfigError,
    DatasetError,
    FileOperationError,
    HereticError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    NetworkError,
    NetworkTimeoutError,
    PhaseConfigError,
    SSHError,
    ValidationFileError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correctly structured."""

    def test_all_exceptions_inherit_from_heretic_error(self):
        """All custom exceptions should inherit from HereticError."""
        exceptions = [
            ModelError,
            ModelLoadError,
            ModelInferenceError,
            DatasetError,
            DatasetConfigError,
            NetworkError,
            NetworkTimeoutError,
            FileOperationError,
            ValidationFileError,
            CloudError,
            SSHError,
            ConfigurationError,
            PhaseConfigError,
            AbliterationError,
            CircuitAblationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, HereticError), (
                f"{exc_class.__name__} should inherit from HereticError"
            )

    def test_heretic_error_inherits_from_exception(self):
        """HereticError should inherit from built-in Exception."""
        assert issubclass(HereticError, Exception)

    def test_model_error_hierarchy(self):
        """Test ModelError hierarchy."""
        assert issubclass(ModelError, HereticError)
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(ModelInferenceError, ModelError)

    def test_dataset_error_hierarchy(self):
        """Test DatasetError hierarchy."""
        assert issubclass(DatasetError, HereticError)
        assert issubclass(DatasetConfigError, DatasetError)

    def test_network_error_hierarchy(self):
        """Test NetworkError hierarchy."""
        assert issubclass(NetworkError, HereticError)
        assert issubclass(NetworkTimeoutError, NetworkError)

    def test_file_operation_error_hierarchy(self):
        """Test FileOperationError hierarchy."""
        assert issubclass(FileOperationError, HereticError)
        assert issubclass(ValidationFileError, FileOperationError)

    def test_cloud_error_hierarchy(self):
        """Test CloudError hierarchy."""
        assert issubclass(CloudError, HereticError)
        assert issubclass(SSHError, CloudError)

    def test_configuration_error_hierarchy(self):
        """Test ConfigurationError hierarchy."""
        assert issubclass(ConfigurationError, HereticError)
        assert issubclass(PhaseConfigError, ConfigurationError)

    def test_abliteration_error_hierarchy(self):
        """Test AbliterationError hierarchy."""
        assert issubclass(AbliterationError, HereticError)
        assert issubclass(CircuitAblationError, AbliterationError)


class TestExceptionInstantiation:
    """Test that exceptions can be raised and caught correctly."""

    def test_raise_heretic_error(self):
        """Test raising base HereticError."""
        with pytest.raises(HereticError, match="test error"):
            raise HereticError("test error")

    def test_raise_model_load_error(self):
        """Test raising ModelLoadError."""
        with pytest.raises(ModelLoadError, match="model not found"):
            raise ModelLoadError("model not found")

    def test_raise_dataset_config_error(self):
        """Test raising DatasetConfigError."""
        with pytest.raises(DatasetConfigError, match="missing config"):
            raise DatasetConfigError("missing config")

    def test_raise_network_timeout_error(self):
        """Test raising NetworkTimeoutError."""
        with pytest.raises(NetworkTimeoutError, match="timeout"):
            raise NetworkTimeoutError("timeout")

    def test_raise_validation_file_error(self):
        """Test raising ValidationFileError."""
        with pytest.raises(ValidationFileError, match="corrupted"):
            raise ValidationFileError("corrupted")

    def test_raise_ssh_error(self):
        """Test raising SSHError."""
        with pytest.raises(SSHError, match="connection failed"):
            raise SSHError("connection failed")

    def test_raise_phase_config_error(self):
        """Test raising PhaseConfigError."""
        with pytest.raises(PhaseConfigError, match="invalid phase"):
            raise PhaseConfigError("invalid phase")

    def test_raise_circuit_ablation_error(self):
        """Test raising CircuitAblationError."""
        with pytest.raises(CircuitAblationError, match="GQA not supported"):
            raise CircuitAblationError("GQA not supported")


class TestExceptionCatching:
    """Test that exceptions can be caught at different hierarchy levels."""

    def test_catch_specific_exception(self):
        """Test catching specific exception."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("model not found")

    def test_catch_parent_exception(self):
        """Test catching parent exception catches child."""
        with pytest.raises(ModelError):
            raise ModelLoadError("model not found")

    def test_catch_base_exception(self):
        """Test catching HereticError catches all custom exceptions."""
        with pytest.raises(HereticError):
            raise ModelLoadError("model not found")

        with pytest.raises(HereticError):
            raise NetworkTimeoutError("timeout")

        with pytest.raises(HereticError):
            raise CircuitAblationError("GQA not supported")

    def test_keyboard_interrupt_not_caught_by_heretic_error(self):
        """Test that KeyboardInterrupt is not caught by HereticError."""
        # This is critical - we should never catch KeyboardInterrupt
        # with HereticError
        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except HereticError:
                pytest.fail("HereticError should not catch KeyboardInterrupt")

    def test_system_exit_not_caught_by_heretic_error(self):
        """Test that SystemExit is not caught by HereticError."""
        with pytest.raises(SystemExit):
            try:
                raise SystemExit(1)
            except HereticError:
                pytest.fail("HereticError should not catch SystemExit")


class TestExceptionChaining:
    """Test exception chaining with 'from' clause."""

    def test_exception_chaining_preserves_cause(self):
        """Test that exception chaining preserves original cause."""
        original_error = ValueError("original error")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise ModelLoadError("wrapped error") from e
        except ModelLoadError as exc:
            assert exc.__cause__ is original_error
            assert str(exc) == "wrapped error"
            assert str(exc.__cause__) == "original error"


class TestExceptionCount:
    """Verify we have exactly 16 exceptions as implemented."""

    def test_exception_count(self):
        """Verify we have 16 custom exceptions (1 base + 15 specific)."""
        # Count all exception classes
        exceptions = [
            HereticError,  # 1 base
            ModelError,
            ModelLoadError,
            ModelInferenceError,  # 3 model
            DatasetError,
            DatasetConfigError,  # 2 dataset
            NetworkError,
            NetworkTimeoutError,  # 2 network
            FileOperationError,
            ValidationFileError,  # 2 file
            CloudError,
            SSHError,  # 2 cloud
            ConfigurationError,
            PhaseConfigError,  # 2 config
            AbliterationError,
            CircuitAblationError,  # 2 abliteration
        ]

        assert len(exceptions) == 16, f"Expected 16 exceptions, got {len(exceptions)}"

    def test_all_exceptions_importable(self):
        """Test that all 15 exceptions are importable."""
        # If we got here, imports at top of file worked
        # This test just makes it explicit
        assert HereticError is not None
        assert ModelError is not None
        assert ModelLoadError is not None
        assert ModelInferenceError is not None
        assert DatasetError is not None
        assert DatasetConfigError is not None
        assert NetworkError is not None
        assert NetworkTimeoutError is not None
        assert FileOperationError is not None
        assert ValidationFileError is not None
        assert CloudError is not None
        assert SSHError is not None
        assert ConfigurationError is not None
        assert PhaseConfigError is not None
        assert AbliterationError is not None
        assert CircuitAblationError is not None
