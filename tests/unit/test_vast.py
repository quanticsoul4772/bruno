# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Vast.ai CLI module.

These tests mock external dependencies (subprocess, fabric, file system)
to avoid making real API calls or requiring the vastai CLI.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAPIKeyValidation:
    """Test API key validation for security."""

    def test_validate_api_key_valid_alphanumeric(self):
        """Test valid alphanumeric API keys are accepted."""
        from bruno.vast import validate_api_key

        # Should not raise
        assert validate_api_key("abc123DEF456") is True
        assert validate_api_key("ABC") is True
        assert validate_api_key("123") is True

    def test_validate_api_key_valid_with_hyphens_underscores(self):
        """Test valid API keys with hyphens and underscores are accepted."""
        from bruno.vast import validate_api_key

        assert validate_api_key("abc123-DEF_456") is True
        assert validate_api_key("test-api-key") is True
        assert validate_api_key("test_api_key") is True
        assert validate_api_key("a-b_c-d_e") is True

    def test_validate_api_key_empty_allowed(self):
        """Test empty API key is allowed (fails later with clear message)."""
        from bruno.vast import validate_api_key

        assert validate_api_key("") is True

    def test_validate_api_key_rejects_semicolon(self):
        """Test semicolon injection is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key; rm -rf /")

    def test_validate_api_key_rejects_backticks(self):
        """Test backtick injection is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key`whoami`")

    def test_validate_api_key_rejects_dollar_sign(self):
        """Test dollar sign variable expansion is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key$HOME")

    def test_validate_api_key_rejects_pipe(self):
        """Test pipe command chaining is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key|cat /etc/passwd")

    def test_validate_api_key_rejects_ampersand(self):
        """Test ampersand background execution is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key&whoami")

    def test_validate_api_key_rejects_quotes(self):
        """Test quotes are blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key'injection")

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key('key"injection')

    def test_validate_api_key_rejects_newlines(self):
        """Test newline injection is blocked."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key\nwhoami")

    def test_validate_api_key_rejects_spaces(self):
        """Test spaces are blocked (could break command parsing)."""
        from bruno.vast import APIKeyValidationError, validate_api_key

        with pytest.raises(APIKeyValidationError, match="invalid characters"):
            validate_api_key("key with spaces")


class TestVastConfig:
    """Test VastConfig dataclass and configuration loading."""

    def test_vast_config_defaults(self):
        """Test VastConfig default values."""
        from bruno.vast import VastConfig

        config = VastConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.instance_id is None
        assert config.ssh_host is None
        assert config.ssh_port is None
        assert config.local_models_dir == "./models"

    def test_vast_config_custom_values(self):
        """Test VastConfig with custom values."""
        from bruno.vast import VastConfig

        config = VastConfig(
            api_key="my-api-key",
            instance_id="12345",
            ssh_host="ssh1.vast.ai",
            ssh_port=22222,
            local_models_dir="/custom/models",
        )

        assert config.api_key == "my-api-key"
        assert config.instance_id == "12345"
        assert config.ssh_host == "ssh1.vast.ai"
        assert config.ssh_port == 22222
        assert config.local_models_dir == "/custom/models"

    def test_vast_config_from_env_with_env_vars(self):
        """Test VastConfig.from_env loads from environment variables."""
        from bruno.vast import VastConfig

        with patch.dict(
            "os.environ",
            {
                "VAST_API_KEY": "env-api-key",
                "LOCAL_MODELS_DIR": "/env/models",
            },
            clear=False,
        ):
            with patch("pathlib.Path.exists", return_value=False):
                config = VastConfig.from_env()

        assert config.api_key == "env-api-key"
        assert config.local_models_dir == "/env/models"

    def test_vast_config_from_env_with_dotenv_file(self):
        """Test VastConfig.from_env reads .env file."""
        from bruno.vast import VastConfig

        env_content = """
# Comment line
VAST_API_KEY=dotenv-api-key
LOCAL_MODELS_DIR=/dotenv/models
EMPTY_VAR=
PLACEHOLDER_VAR=your_api_key_here
"""

        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=env_content):
                    config = VastConfig.from_env()

        assert config.api_key == "dotenv-api-key"

    def test_vast_config_from_env_empty_api_key(self):
        """Test VastConfig.from_env handles empty API key."""
        from bruno.vast import VastConfig

        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                config = VastConfig.from_env()

        assert config.api_key == ""

    def test_vast_config_from_env_validates_api_key(self):
        """Test VastConfig.from_env validates API key for injection attacks."""
        from bruno.vast import VastConfig

        with patch.dict("os.environ", {"VAST_API_KEY": "valid-key_123"}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                config = VastConfig.from_env()  # Should not raise
                assert config.api_key == "valid-key_123"

    def test_vast_config_from_env_rejects_malicious_api_key(self):
        """Test VastConfig.from_env rejects API key with shell metacharacters."""
        from bruno.vast import APIKeyValidationError, VastConfig

        with patch.dict("os.environ", {"VAST_API_KEY": "key; rm -rf /"}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(APIKeyValidationError):
                    VastConfig.from_env()


class TestGPUTiers:
    """Test GPU tier configuration."""

    def test_gpu_tiers_contains_expected_tiers(self):
        """Test GPU_TIERS has all expected tier configurations."""
        from bruno.vast import GPU_TIERS

        expected_tiers = [
            "RTX_4090",
            "A6000",
            "A100_40GB",
            "A100_80GB",
            "A100_SXM",
            "H100",
        ]

        for tier in expected_tiers:
            assert tier in GPU_TIERS, f"Missing tier: {tier}"

    def test_gpu_tiers_have_required_fields(self):
        """Test each GPU tier has all required configuration fields."""
        from bruno.vast import GPU_TIERS

        required_fields = [
            "gpu_name",
            "max_price",
            "min_vram",
            "disk_gb",
            "description",
        ]

        for tier_name, tier_config in GPU_TIERS.items():
            for field in required_fields:
                assert field in tier_config, f"Tier {tier_name} missing field: {field}"

    def test_gpu_tiers_vram_ordering(self):
        """Test GPU tiers are ordered by VRAM (generally)."""
        from bruno.vast import GPU_TIERS

        # RTX_4090 should have 24GB
        assert GPU_TIERS["RTX_4090"]["min_vram"] == 24

        # A6000 should have 48GB
        assert GPU_TIERS["A6000"]["min_vram"] == 48

        # A100_80GB and H100 should have 80GB
        assert GPU_TIERS["A100_80GB"]["min_vram"] == 80
        assert GPU_TIERS["H100"]["min_vram"] == 80

    def test_gpu_tiers_price_ordering(self):
        """Test higher tier GPUs have higher max prices."""
        from bruno.vast import GPU_TIERS

        assert GPU_TIERS["RTX_4090"]["max_price"] < GPU_TIERS["A100_80GB"]["max_price"]
        assert GPU_TIERS["A100_80GB"]["max_price"] < GPU_TIERS["H100"]["max_price"]


class TestFindVastaiCLI:
    """Test CLI discovery logic."""

    def test_find_vastai_cli_local_exe(self):
        """Test finding vast.exe in current directory."""
        from bruno.vast import find_vastai_cli

        with patch("pathlib.Path.exists") as mock_exists:
            # First path check (vast.exe in current dir) returns True
            mock_exists.side_effect = [True, False, False, False]

            result = find_vastai_cli()

        # Should return absolute path to local vast.exe
        assert "vast" in result[0].lower()

    def test_find_vastai_cli_in_path(self):
        """Test finding vastai in system PATH."""
        from bruno.vast import find_vastai_cli

        with patch("pathlib.Path.exists", return_value=False):
            with patch("shutil.which") as mock_which:
                mock_which.side_effect = (
                    lambda cmd: "/usr/bin/vastai" if cmd == "vastai" else None
                )

                result = find_vastai_cli()

        assert result == ["vastai"]

    def test_find_vastai_cli_vast_in_path(self):
        """Test finding 'vast' command in PATH."""
        from bruno.vast import find_vastai_cli

        with patch("pathlib.Path.exists", return_value=False):
            with patch("shutil.which") as mock_which:
                # vastai not found, but vast is
                mock_which.side_effect = (
                    lambda cmd: "/usr/bin/vast" if cmd == "vast" else None
                )

                result = find_vastai_cli()

        assert result == ["vast"]

    def test_find_vastai_cli_fallback(self):
        """Test fallback when CLI not found."""
        from bruno.vast import find_vastai_cli

        with patch("pathlib.Path.exists", return_value=False):
            with patch("shutil.which", return_value=None):
                with patch("sys.platform", "linux"):
                    result = find_vastai_cli()

        # Should return default (will fail with clear error)
        assert result == ["vastai"]


class TestRunVastaiCmd:
    """Test running vastai CLI commands."""

    def test_run_vastai_cmd_success(self):
        """Test successful CLI command execution."""
        from bruno.vast import VastConfig, run_vastai_cmd

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.find_vastai_cli", return_value=["vastai"]):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="success output",
                    stderr="",
                )

                code, stdout, stderr = run_vastai_cmd(["show", "instances"], config)

        assert code == 0
        assert stdout == "success output"
        assert stderr == ""

    def test_run_vastai_cmd_with_api_key(self):
        """Test API key is passed via environment."""
        from bruno.vast import VastConfig, run_vastai_cmd

        config = VastConfig(api_key="secret-key")

        with patch("bruno.vast.find_vastai_cli", return_value=["vastai"]):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                run_vastai_cmd(["show", "instances"], config)

        # Verify API key was in environment
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["env"]["VAST_API_KEY"] == "secret-key"

    def test_run_vastai_cmd_missing_api_key(self):
        """Test error when API key is missing."""
        from bruno.vast import VastConfig, run_vastai_cmd

        config = VastConfig(api_key="")

        with patch.dict("os.environ", {}, clear=True):
            code, stdout, stderr = run_vastai_cmd(["show", "instances"], config)

        assert code == 1
        assert "VAST_API_KEY not set" in stderr

    def test_run_vastai_cmd_cli_not_found(self):
        """Test error when CLI executable not found."""
        from bruno.vast import VastConfig, run_vastai_cmd

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.find_vastai_cli", return_value=["vastai"]):
            with patch("subprocess.run", side_effect=FileNotFoundError()):
                code, stdout, stderr = run_vastai_cmd(["show", "instances"], config)

        assert code == 1
        assert "not found" in stderr.lower()

    def test_run_vastai_cmd_timeout(self):
        """Test timeout handling."""
        import subprocess

        from bruno.vast import VastConfig, run_vastai_cmd

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.find_vastai_cli", return_value=["vastai"]):
            with patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 120)
            ):
                code, stdout, stderr = run_vastai_cmd(["show", "instances"], config)

        assert code == 1
        assert "timed out" in stderr.lower()


class TestGetSSHInfo:
    """Test SSH URL parsing for various formats."""

    def test_get_ssh_info_ssh_url_format(self):
        """Test parsing ssh://user@host:port format."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "ssh://root@ssh1.vast.ai:35702", "")

            result = get_ssh_info("12345", config)

        assert result == ("ssh1.vast.ai", 35702)

    def test_get_ssh_info_ssh_p_format_ip(self):
        """Test parsing 'ssh -p PORT user@IP' format."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "ssh -p 35702 root@192.168.1.100", "")

            result = get_ssh_info("12345", config)

        assert result == ("192.168.1.100", 35702)

    def test_get_ssh_info_ssh_p_format_hostname(self):
        """Test parsing 'ssh -p PORT user@hostname' format."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "ssh -p 22222 root@ssh3.vast.ai", "")

            result = get_ssh_info("12345", config)

        assert result == ("ssh3.vast.ai", 22222)

    def test_get_ssh_info_user_host_port_format(self):
        """Test parsing user@host:port format (without ssh://)."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "root@ssh5.vast.ai:44444", "")

            result = get_ssh_info("12345", config)

        assert result == ("ssh5.vast.ai", 44444)

    def test_get_ssh_info_command_failure(self):
        """Test handling of failed ssh-url command."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (1, "", "Instance not found")

            result = get_ssh_info("12345", config)

        assert result is None

    def test_get_ssh_info_unparseable_output(self):
        """Test handling of unparseable SSH URL output."""
        from bruno.vast import VastConfig, get_ssh_info

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "some random output", "")

            result = get_ssh_info("12345", config)

        assert result is None


class TestGetInstances:
    """Test instance listing and filtering."""

    def test_get_instances_success(self):
        """Test successful instance listing."""
        from bruno.vast import VastConfig, get_instances

        config = VastConfig(api_key="test-key")

        instances_json = json.dumps(
            [
                {"id": 1, "actual_status": "running", "gpu_name": "RTX_4090"},
                {"id": 2, "actual_status": "stopped", "gpu_name": "A100"},
            ]
        )

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, instances_json, "")

            result = get_instances(config)

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["actual_status"] == "stopped"

    def test_get_instances_empty(self):
        """Test empty instance list."""
        from bruno.vast import VastConfig, get_instances

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "", "")

            result = get_instances(config)

        assert result == []

    def test_get_instances_api_error(self):
        """Test handling of API errors."""
        from bruno.vast import VastConfig, get_instances

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (1, "", "Unauthorized")
            with patch("bruno.vast.console.print"):  # Suppress output
                result = get_instances(config)

        assert result == []

    def test_get_instances_invalid_json(self):
        """Test handling of invalid JSON response."""
        from bruno.vast import VastConfig, get_instances

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.run_vastai_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "not valid json", "")
            with patch("bruno.vast.console.print"):  # Suppress output
                result = get_instances(config)

        assert result == []


class TestGetRunningInstance:
    """Test finding running instances."""

    def test_get_running_instance_finds_running(self):
        """Test finding first running instance."""
        from bruno.vast import VastConfig, get_running_instance

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.get_instances") as mock_get:
            mock_get.return_value = [
                {"id": 1, "actual_status": "stopped"},
                {"id": 2, "actual_status": "running"},
                {"id": 3, "actual_status": "running"},
            ]

            result = get_running_instance(config)

        # Should return first running instance (id=2)
        assert result["id"] == 2

    def test_get_running_instance_with_status_field(self):
        """Test finding instance using 'status' field instead of 'actual_status'."""
        from bruno.vast import VastConfig, get_running_instance

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.get_instances") as mock_get:
            mock_get.return_value = [
                {"id": 1, "status": "running"},
            ]

            result = get_running_instance(config)

        assert result["id"] == 1

    def test_get_running_instance_none_running(self):
        """Test returning first instance when none running."""
        from bruno.vast import VastConfig, get_running_instance

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.get_instances") as mock_get:
            mock_get.return_value = [
                {"id": 1, "actual_status": "stopped"},
                {"id": 2, "actual_status": "stopped"},
            ]

            result = get_running_instance(config)

        # Returns first instance when none running
        assert result["id"] == 1

    def test_get_running_instance_empty_list(self):
        """Test handling empty instance list."""
        from bruno.vast import VastConfig, get_running_instance

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.get_instances") as mock_get:
            mock_get.return_value = []
            with patch("bruno.vast.find_vastai_cli", return_value=["vastai"]):
                with patch("bruno.vast.console.print"):  # Suppress output
                    result = get_running_instance(config)

        assert result is None


class TestConstants:
    """Test module constants."""

    def test_default_image(self):
        """Test DEFAULT_IMAGE is a valid Docker image."""
        from bruno.vast import DEFAULT_IMAGE

        assert "pytorch" in DEFAULT_IMAGE
        assert "cuda" in DEFAULT_IMAGE

    def test_min_download_speed(self):
        """Test MIN_DOWNLOAD_SPEED is reasonable."""
        from bruno.vast import MIN_DOWNLOAD_SPEED

        assert MIN_DOWNLOAD_SPEED >= 100  # At least 100 Mbps

    def test_models_dir(self):
        """Test MODELS_DIR path."""
        from bruno.vast import MODELS_DIR

        assert MODELS_DIR.startswith("/workspace")


class TestGetConnection:
    """Test SSH connection creation."""

    def test_get_connection_fabric_not_available(self):
        """Test handling when fabric is not installed."""
        from bruno.vast import VastConfig

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.FABRIC_AVAILABLE", False):
            with patch("bruno.vast.console.print"):  # Suppress output
                from bruno.vast import get_connection

                result = get_connection("12345", config)

        assert result is None

    def test_get_connection_no_ssh_info(self):
        """Test handling when SSH info cannot be retrieved."""
        from bruno.vast import VastConfig, get_connection

        config = VastConfig(api_key="test-key")

        with patch("bruno.vast.FABRIC_AVAILABLE", True):
            with patch("bruno.vast.get_ssh_info", return_value=None):
                with patch("bruno.vast.console.print"):  # Suppress output
                    result = get_connection("12345", config)

        assert result is None

    def test_get_connection_success(self):
        """Test successful connection creation."""
        from bruno.vast import VastConfig, get_connection

        config = VastConfig(api_key="test-key")

        mock_connection_cls = MagicMock()
        mock_connection = MagicMock()
        mock_connection_cls.return_value = mock_connection

        with patch("bruno.vast.FABRIC_AVAILABLE", True):
            with patch("bruno.vast.Connection", mock_connection_cls):
                with patch(
                    "bruno.vast.get_ssh_info", return_value=("ssh1.vast.ai", 22222)
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "pathlib.Path.home", return_value=Path("/home/user")
                        ):
                            result = get_connection("12345", config)

        assert result is not None
        mock_connection_cls.assert_called_once()

        # Verify connection parameters
        call_kwargs = mock_connection_cls.call_args[1]
        assert call_kwargs["host"] == "ssh1.vast.ai"
        assert call_kwargs["port"] == 22222
        assert call_kwargs["user"] == "root"


class TestRunSSHCommand:
    """Test SSH command execution."""

    def test_run_ssh_command_success(self):
        """Test successful SSH command execution."""
        from bruno.vast import run_ssh_command

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.stdout = "command output"
        mock_conn.run.return_value = mock_result

        result = run_ssh_command(mock_conn, "nvidia-smi")

        assert result == "command output"
        # The command is wrapped with timeout for reliability
        mock_conn.run.assert_called_once()
        call_args = mock_conn.run.call_args
        assert "nvidia-smi" in call_args[0][0]  # Command is in the wrapped string
        assert call_args[1]["hide"] is True
        assert call_args[1]["warn"] is True

    def test_run_ssh_command_failure(self):
        """Test handling of failed SSH command."""
        from bruno.vast import run_ssh_command

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = False
        mock_conn.run.return_value = mock_result

        result = run_ssh_command(mock_conn, "invalid-command")

        assert result == ""

    def test_run_ssh_command_connection_error(self):
        """Test handling of connection errors (OSError, TimeoutError, EOFError)."""
        from bruno.vast import run_ssh_command

        # Test OSError (covers socket errors)
        mock_conn = MagicMock()
        mock_conn.run.side_effect = OSError("Connection refused")
        assert run_ssh_command(mock_conn, "nvidia-smi") == ""

        # Test TimeoutError
        mock_conn = MagicMock()
        mock_conn.run.side_effect = TimeoutError("Connection timed out")
        assert run_ssh_command(mock_conn, "nvidia-smi") == ""

        # Test EOFError (connection closed)
        mock_conn = MagicMock()
        mock_conn.run.side_effect = EOFError("Connection closed")
        assert run_ssh_command(mock_conn, "nvidia-smi") == ""


@pytest.mark.slow
@pytest.mark.integration
class TestVastIntegration:
    """Integration tests with real Vast.ai API.

    These tests require a valid VAST_API_KEY and network access.
    Marked slow so they don't run in CI by default.
    """

    def test_list_instances_real(self):
        """Integration test with real API call."""
        pytest.skip("Requires VAST_API_KEY - run with -m slow")
