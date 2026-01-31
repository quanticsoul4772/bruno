# SPDX-License-Identifier: AGPL-3.0-or-later
# Vast.ai GPU Cloud Management CLI for Heretic
#
# A clean Python replacement for runpod.ps1 that uses:
# - Fabric for SSH (no shell escaping hell!)
# - Rich for beautiful terminal output
# - Click for CLI structure

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

# Progress imports removed - were unused
from rich.table import Table
from rich.text import Text

from .exceptions import ConfigurationError, SSHError
from .logging import get_logger

logger = get_logger(__name__)

# Lazy imports for fabric to avoid import errors if not installed
try:
    from fabric import Connection

    FABRIC_AVAILABLE = True
except ImportError:
    FABRIC_AVAILABLE = False
    Connection = None

console = Console()

# GPU Tier Configurations
GPU_TIERS = {
    "RTX_4090": {
        "gpu_name": "RTX_4090",
        "max_price": 0.50,
        "min_vram": 24,
        "disk_gb": 50,
        "description": "24GB VRAM - Good for 7B-8B models",
    },
    "A6000": {
        "gpu_name": "RTX_A6000",
        "max_price": 0.80,
        "min_vram": 48,
        "disk_gb": 80,
        "description": "48GB VRAM - Good for 14B-30B models",
    },
    "A100_40GB": {
        "gpu_name": "A100",
        "max_price": 1.00,
        "min_vram": 40,
        "disk_gb": 100,
        "description": "40GB VRAM - Good for 14B-32B models",
    },
    "A100_80GB": {
        "gpu_name": "A100",
        "max_price": 2.00,
        "min_vram": 80,
        "disk_gb": 150,
        "description": "80GB VRAM - Good for 32B-70B models",
    },
    "A100_SXM": {
        "gpu_name": "A100_SXM4",
        "max_price": 2.50,
        "min_vram": 80,
        "disk_gb": 150,
        "description": "80GB VRAM SXM4 - Fastest A100 variant",
    },
    "H100": {
        "gpu_name": "H100",
        "max_price": 4.00,
        "min_vram": 80,
        "disk_gb": 150,
        "description": "80GB VRAM - Fastest, good for 70B+ models",
    },
}

DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
MIN_DOWNLOAD_SPEED = 200
MODELS_DIR = "/workspace/models"


# Regex pattern for valid API keys (alphanumeric, hyphens, underscores only)
# This prevents command injection when passing keys through shell (especially WSL)
API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class APIKeyValidationError(ValueError):
    """Raised when an API key contains invalid characters."""

    pass


def validate_api_key(api_key: str) -> bool:
    """Validate that an API key contains only safe characters.

    This prevents command injection vulnerabilities when the API key
    is passed through shell commands (especially via WSL).

    Args:
        api_key: The API key to validate.

    Returns:
        True if valid, raises APIKeyValidationError if invalid.

    Raises:
        APIKeyValidationError: If the API key contains shell metacharacters.
    """
    if not api_key:
        return True  # Empty keys are allowed (will fail later with clear message)

    if not API_KEY_PATTERN.match(api_key):
        raise APIKeyValidationError(
            "VAST_API_KEY contains invalid characters. "
            "Only alphanumeric characters, hyphens, and underscores are allowed. "
            "This prevents command injection when using WSL."
        )

    return True


@dataclass
class VastConfig:
    """Configuration for Vast.ai connection."""

    api_key: str
    instance_id: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    local_models_dir: str = "./models"

    @classmethod
    def from_env(cls) -> "VastConfig":
        """Load configuration from environment variables and .env file.

        Raises:
            APIKeyValidationError: If VAST_API_KEY contains shell metacharacters.
        """
        # Try to load from .env file first
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value and value != "your_api_key_here":
                        os.environ.setdefault(key, value)

        api_key = os.environ.get("VAST_API_KEY", "")

        # SECURITY: Validate API key format to prevent command injection
        validate_api_key(api_key)

        return cls(
            api_key=api_key,
            local_models_dir=os.environ.get("LOCAL_MODELS_DIR", "./models"),
        )


def find_vastai_cli() -> list[str]:
    """Find the vastai CLI executable. Returns command prefix list."""
    # Check for vast.exe in current directory or script directory
    script_dir = Path(
        __file__
    ).parent.parent.parent  # Go up from src/heretic to project root
    for vast_path in [
        Path("vast.exe"),
        Path("vast"),
        script_dir / "vast.exe",
        script_dir / "vast",
    ]:
        if vast_path.exists():
            return [str(vast_path.absolute())]

    # Check if vastai is in PATH
    if shutil.which("vastai"):
        return ["vastai"]

    if shutil.which("vast"):
        return ["vast"]

    # On Windows, try WSL
    if sys.platform == "win32":
        # Check if WSL is available
        try:
            result = subprocess.run(["wsl", "--status"], capture_output=True, timeout=5)
            if result.returncode == 0:
                # Check if vastai is installed in WSL
                result = subprocess.run(
                    ["wsl", "-e", "which", "vastai"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Return WSL command - we'll handle env vars specially
                    return ["wsl", "-e", "vastai"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Default - will fail but with a clear error
    return ["vastai"]


def run_vastai_cmd(args: list[str], config: VastConfig) -> tuple[int, str, str]:
    """Run a vastai CLI command and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    if config.api_key:
        env["VAST_API_KEY"] = config.api_key
    else:
        # Check if API key might be missing
        if not os.environ.get("VAST_API_KEY"):
            return (
                1,
                "",
                "VAST_API_KEY not set. Add it to .env file or set environment variable.",
            )

    cmd_prefix = find_vastai_cli()

    # If using WSL, we need to pass the API key explicitly since WSL has its own env
    if cmd_prefix and cmd_prefix[0] == "wsl" and config.api_key:
        # Modify command to pass env var through WSL
        # Escape any special characters in the API key
        escaped_key = config.api_key.replace("'", "'\"'\"'")
        cmd = [
            "wsl",
            "-e",
            "bash",
            "-c",
            f"VAST_API_KEY='{escaped_key}' vastai {' '.join(args)}",
        ]
    else:
        cmd = cmd_prefix + args

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=120
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        error_msg = (
            "Vast.ai CLI not found. Install it with:\n"
            "  pip install vastai\n"
            "Or download vast.exe to your project directory."
        )
        return 1, "", error_msg
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"


def get_instances(config: VastConfig) -> list[dict]:
    """Get list of Vast.ai instances."""
    code, stdout, stderr = run_vastai_cmd(["show", "instances", "--raw"], config)
    if code != 0:
        logger.debug(
            "get_instances API call failed",
            exit_code=code,
            stderr=stderr[:200] if stderr else None,
        )
        if stderr:
            console.print(f"[red]API Error: {stderr}[/]")
        return []
    if not stdout.strip():
        logger.debug("get_instances returned empty response")
        return []
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        logger.debug(
            "get_instances JSON parse error",
            error=str(e),
            response_preview=stdout[:200] if stdout else None,
        )
        console.print("[yellow]Warning: Could not parse API response[/]")
        return []


def get_running_instance(config: VastConfig) -> Optional[dict]:
    """Get the first running instance."""
    instances = get_instances(config)
    if not instances:
        # Debug: show what CLI we're using
        cmd_prefix = find_vastai_cli()
        logger.debug(
            "No instances found",
            cli_command=" ".join(cmd_prefix),
        )
        console.print(f"[dim]CLI: {' '.join(cmd_prefix)}[/]")
        return None
    for inst in instances:
        if inst.get("actual_status") == "running" or inst.get("status") == "running":
            return inst
    logger.debug(
        "No running instance found, returning first instance",
        instance_id=instances[0].get("id") if instances else None,
    )
    return instances[0] if instances else None


def get_ssh_info(
    instance_id: str, config: VastConfig, verbose: bool = False
) -> Optional[tuple[str, int]]:
    """Get SSH host and port for an instance.

    Handles multiple SSH URL formats from Vast.ai:
    - ssh://root@ssh1.vast.ai:35702
    - ssh://root@192.168.1.1:35702
    - ssh -p 35702 root@192.168.1.1
    - ssh -p 35702 root@ssh1.vast.ai
    """
    code, stdout, stderr = run_vastai_cmd(["ssh-url", str(instance_id)], config)

    if verbose:
        console.print(f"[dim]SSH URL response (code={code}): {stdout.strip()}[/]")
        if stderr:
            console.print(f"[dim]SSH URL stderr: {stderr.strip()}[/]")

    if code != 0:
        logger.debug(
            "get_ssh_info failed",
            instance_id=instance_id,
            exit_code=code,
            stderr=stderr[:200] if stderr else None,
        )
        if verbose or stderr:
            console.print(f"[red]Failed to get SSH URL for instance {instance_id}[/]")
            if stderr:
                console.print(f"[red]Error: {stderr}[/]")
            if "api-key" in stderr.lower() or "unauthorized" in stderr.lower():
                console.print("[yellow]Hint: Check your VAST_API_KEY in .env file[/]")
        return None

    # Parse multiple SSH URL formats
    # Format 1: ssh://user@host:port (hostname like ssh1.vast.ai)
    match = re.search(r"ssh://([^@]+)@([^:]+):(\d+)", stdout)
    if match:
        return match.group(2), int(match.group(3))

    # Format 2: ssh -p PORT user@host (IP address)
    match = re.search(r"-p\s+(\d+)\s+([^@]+)@([\d.]+)", stdout)
    if match:
        return match.group(3), int(match.group(1))

    # Format 3: ssh -p PORT user@hostname (hostname like ssh1.vast.ai)
    match = re.search(r"-p\s+(\d+)\s+([^@]+)@(\S+)", stdout)
    if match:
        return match.group(3), int(match.group(1))

    # Format 4: user@host:port without ssh:// prefix
    match = re.search(r"([^@]+)@([^:]+):(\d+)", stdout)
    if match:
        return match.group(2), int(match.group(3))

    logger.debug(
        "Could not parse SSH URL format",
        instance_id=instance_id,
        raw_output=stdout.strip()[:200] if stdout else None,
    )
    if verbose:
        console.print(
            f"[yellow]Warning: Could not parse SSH URL format: {stdout.strip()}[/]"
        )
    return None


def get_connection(instance_id: str, config: VastConfig) -> Optional["Connection"]:
    """Create a Fabric SSH connection to the instance."""
    if not FABRIC_AVAILABLE:
        logger.debug("Fabric library not available for SSH connection")
        console.print("[red]Error: fabric not installed. Run: pip install fabric[/]")
        return None

    ssh_info = get_ssh_info(instance_id, config)
    if not ssh_info:
        logger.debug(
            "Cannot create connection - SSH info unavailable",
            instance_id=instance_id,
        )
        console.print("[red]Error: Could not get SSH info for instance[/]")
        return None

    host, port = ssh_info

    # Try common SSH key locations
    ssh_key = None
    for key_path in [
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
    ]:
        if key_path.exists():
            ssh_key = str(key_path)
            break

    connect_kwargs = {
        "look_for_keys": True,
        "allow_agent": True,
    }
    if ssh_key:
        connect_kwargs["key_filename"] = ssh_key

    return Connection(
        host=host,
        port=port,
        user="root",
        connect_kwargs=connect_kwargs,
        connect_timeout=30,
    )


def run_ssh_command(
    conn: "Connection",
    command: str,
    hide: bool = True,
    timeout: int = 60,
    critical: bool = False,
) -> str:
    """Run a command via SSH with timeout and return stdout.

    Args:
        conn: Fabric SSH connection
        command: Command to execute
        hide: Hide command output (default: True)
        timeout: Command timeout in seconds (default: 60)
            Use 60s for quick commands, 300s for long operations
        critical: If True, raise SSHError on failure. If False, return empty string (default)

    Returns:
        Command stdout on success, empty string on non-critical failure

    Raises:
        SSHError: If critical=True and command fails or times out

    Examples:
        # Quick command (60s timeout, non-critical)
        output = run_ssh_command(conn, "ls /workspace")

        # Long operation (300s timeout, critical)
        run_ssh_command(conn, "apt-get update", timeout=300, critical=True)

        # Critical command that must succeed
        run_ssh_command(conn, "mkdir -p /workspace", critical=True)
    """
    try:
        # Wrap command with GNU timeout for time limiting
        # timeout command exits with 124 if command times out
        timeout_command = f"timeout {timeout} bash -c {repr(command)}"
        result = conn.run(timeout_command, hide=hide, warn=True)

        # Check if command timed out (exit code 124)
        if result.return_code == 124:
            if critical:
                raise SSHError(
                    f"SSH command timed out after {timeout}s. "
                    f"Solutions: 1) Increase timeout for long operations "
                    f"2) Check if command is hanging 3) Verify network stability"
                )
            return ""

        # Return stdout if successful, empty string if failed
        return result.stdout if result and result.ok else ""

    except TimeoutError as e:
        # Network/connection timeout (different from command timeout)
        if critical:
            raise SSHError(
                f"SSH connection timeout after {timeout}s. "
                f"Solutions: 1) Check network connection 2) Increase timeout "
                f"3) Verify instance is responsive"
            ) from e
        return ""
    except (OSError, EOFError) as e:
        # Connection errors: socket issues, connection closed
        if critical:
            raise SSHError(
                f"SSH connection error: {e}. "
                f"Solutions: 1) Check instance status 2) Verify network connection "
                f"3) Restart SSH connection"
            ) from e
        return ""
    except KeyboardInterrupt:
        # CRITICAL: Always let user interrupt
        raise


# Input Validation Functions
def validate_instance_id(instance_id: int) -> int:
    """Validate Vast.ai instance ID.

    Args:
        instance_id: Instance ID to validate

    Returns:
        Validated instance ID

    Raises:
        ConfigurationError: If instance ID is invalid
    """
    if not isinstance(instance_id, int):
        raise ConfigurationError(
            f"Instance ID must be an integer, got: {type(instance_id).__name__}"
        )
    if instance_id <= 0:
        raise ConfigurationError(f"Instance ID must be positive, got: {instance_id}")
    if instance_id > 999999999:  # Reasonable upper bound
        raise ConfigurationError(
            f"Instance ID too large (max 999999999), got: {instance_id}"
        )
    return instance_id


def validate_model_name(model_name: str) -> str:
    """Validate and sanitize model name to prevent path traversal and command injection.

    Args:
        model_name: Model name to validate

    Returns:
        Validated model name

    Raises:
        ConfigurationError: If model name contains dangerous patterns
    """
    if not model_name or not isinstance(model_name, str):
        raise ConfigurationError("Model name must be a non-empty string")

    # Check for path traversal attempts
    if ".." in model_name:
        raise ConfigurationError(
            "Invalid model name: path traversal blocked (contains '..')"
        )

    # Check for absolute paths (security risk)
    if model_name.startswith("/") or (len(model_name) > 1 and model_name[1] == ":"):
        raise ConfigurationError("Invalid model name: absolute paths not allowed")

    # Check for shell metacharacters that could enable command injection
    dangerous_chars = [";", "|", "&", "$", "`", "\n", "\r", "<", ">"]
    for char in dangerous_chars:
        if char in model_name:
            raise ConfigurationError(
                f"Invalid model name: contains dangerous character '{char}'"
            )

    # Reasonable length limit
    if len(model_name) > 500:
        raise ConfigurationError(
            f"Model name too long (max 500 chars), got: {len(model_name)}"
        )

    return model_name


def validate_path(
    path: str, must_exist: bool = False, must_be_dir: bool = False
) -> Path:
    """Validate and sanitize file path to prevent traversal attacks.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        must_be_dir: If True, path must be a directory

    Returns:
        Validated Path object

    Raises:
        ConfigurationError: If path is invalid or dangerous
    """
    if not path or not isinstance(path, str):
        raise ConfigurationError("Path must be a non-empty string")

    # Convert to Path object for validation
    try:
        path_obj = Path(path).resolve()
    except (ValueError, OSError) as e:
        raise ConfigurationError(f"Invalid path format: {e}") from e

    # Check for path traversal attempts
    if ".." in path:
        raise ConfigurationError("Invalid path: path traversal blocked (contains '..')")

    # Validate existence if required
    if must_exist and not path_obj.exists():
        raise ConfigurationError(f"Path does not exist: {path_obj}")

    # Validate directory if required
    if must_be_dir and path_obj.exists() and not path_obj.is_dir():
        raise ConfigurationError(f"Path is not a directory: {path_obj}")

    return path_obj


def validate_gpu_tier(tier: str) -> str:
    """Validate GPU tier name.

    Args:
        tier: GPU tier name

    Returns:
        Validated tier name

    Raises:
        ConfigurationError: If tier is invalid
    """
    if not tier or not isinstance(tier, str):
        raise ConfigurationError("GPU tier must be a non-empty string")

    if tier not in GPU_TIERS:
        available = ", ".join(GPU_TIERS.keys())
        raise ConfigurationError(f"Invalid GPU tier: {tier}. Available: {available}")

    return tier


def validate_disk_size(disk_gb: int) -> int:
    """Validate disk size.

    Args:
        disk_gb: Disk size in GB

    Returns:
        Validated disk size

    Raises:
        ConfigurationError: If disk size is invalid
    """
    if not isinstance(disk_gb, int):
        raise ConfigurationError(
            f"Disk size must be an integer, got: {type(disk_gb).__name__}"
        )
    if disk_gb < 50:
        raise ConfigurationError(f"Disk size too small (min 50GB), got: {disk_gb}GB")
    if disk_gb > 10000:  # 10TB reasonable upper bound
        raise ConfigurationError(f"Disk size too large (max 10000GB), got: {disk_gb}GB")
    return disk_gb


def validate_and_convert_instance_id(instance_id_str: str) -> str:
    """Helper to validate instance ID string and return validated string.

    Args:
        instance_id_str: Instance ID as string

    Returns:
        Validated instance ID as string

    Raises:
        ValueError: If not a valid number
        ConfigurationError: If invalid instance ID
    """
    instance_id_int = int(instance_id_str)  # Can raise ValueError
    validate_instance_id(instance_id_int)  # Can raise ConfigurationError
    return str(instance_id_int)


# CLI Commands
@click.group()
@click.pass_context
def cli(ctx):
    """Heretic Vast.ai GPU Cloud Management CLI.

    Manage GPU instances on Vast.ai for abliterating language models.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = VastConfig.from_env()


@cli.command("tiers")
def show_tiers():
    """Show available GPU tiers and pricing."""
    table = Table(title="Available GPU Tiers")
    table.add_column("Tier", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("Max Price", justify="right")
    table.add_column("Description")

    for name, tier in GPU_TIERS.items():
        table.add_row(
            name,
            f"{tier['min_vram']}GB",
            f"${tier['max_price']}/hr",
            tier["description"],
        )

    console.print(table)
    console.print()
    console.print("[yellow]Model size recommendations:[/]")
    console.print("  7B-8B models   -> RTX_4090 (1 GPU)")
    console.print("  14B-30B models -> A6000 or A100_40GB (1 GPU)")
    console.print("  32B models     -> A100_80GB (1 GPU)")
    console.print("  70B-72B models -> A100_80GB (2 GPUs) or H100 (2 GPUs)")


@cli.command("gpus")
@click.argument("tier", default="RTX_4090")
@click.pass_context
def search_gpus(ctx, tier: str):
    """Search available GPU offers."""
    config = ctx.obj["config"]

    if tier not in GPU_TIERS:
        console.print(f"[red]Unknown tier: {tier}[/]")
        console.print(f"Valid tiers: {', '.join(GPU_TIERS.keys())}")
        return

    tier_config = GPU_TIERS[tier]
    console.print(f"Searching Vast.ai GPU offers for [cyan]{tier}[/]...")
    console.print(f"({tier_config['description']})")

    # Build search query
    vram_mb = tier_config["min_vram"] * 1024
    query = (
        f"gpu_name={tier_config['gpu_name']} "
        f"rentable=true num_gpus>=1 "
        f"inet_down>={MIN_DOWNLOAD_SPEED} "
        f"dph<={tier_config['max_price']}"
    )
    if tier_config["min_vram"] >= 40:
        query += f" gpu_ram>={vram_mb}"

    code, stdout, stderr = run_vastai_cmd(
        ["search", "offers", query, "--order", "dph", "--limit", "20"],
        config,
    )
    console.print(stdout)


@cli.command("create")
@click.argument("tier", default="RTX_4090")
@click.argument("num_gpus", default=1, type=int)
@click.pass_context
def create_pod(ctx, tier: str, num_gpus: int):
    """Create a new Vast.ai instance."""
    config = ctx.obj["config"]

    if not config.api_key:
        console.print("[red]Error: VAST_API_KEY not set![/]")
        console.print("Set: export VAST_API_KEY='your-key'")
        return

    # Validate inputs
    try:
        tier = validate_gpu_tier(tier)
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    if not isinstance(num_gpus, int) or num_gpus < 1 or num_gpus > 8:
        console.print(f"[red]Invalid num_gpus: must be 1-8, got {num_gpus}[/]")
        return

    tier_config = GPU_TIERS[tier]
    disk_gb = tier_config["disk_gb"]
    max_price = tier_config["max_price"] * num_gpus

    if num_gpus > 1:
        disk_gb = max(disk_gb, 400)  # 32B+ models need more space for weights + cache

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Creating Vast.ai Instance[/]\n\n"
            f"GPU Tier: [cyan]{tier}[/] ({tier_config['description']})\n"
            f"Num GPUs: [cyan]{num_gpus}[/]\n"
            f"Disk: [cyan]{disk_gb} GB[/]\n"
            f"Max price: [cyan]${max_price}/hr[/]",
            title="Configuration",
        )
    )

    # Find best offer
    console.print("\n[yellow]Finding best GPU offer...[/]")

    vram_mb = tier_config["min_vram"] * 1024
    query = (
        f"gpu_name={tier_config['gpu_name']} "
        f"rentable=true num_gpus>={num_gpus} "
        f"inet_down>={MIN_DOWNLOAD_SPEED} "
        f"dph<={max_price}"
    )
    if tier_config["min_vram"] >= 40:
        query += f" gpu_ram>={vram_mb}"

    code, stdout, stderr = run_vastai_cmd(
        ["search", "offers", query, "--order", "dph", "--limit", "1", "--raw"],
        config,
    )

    if code != 0 or not stdout.strip():
        console.print("[red]Error: No suitable GPU offers found[/]")
        console.print(f"Try: heretic-vast gpus {tier}")
        return

    try:
        offers = json.loads(stdout)
        if not offers:
            console.print("[red]No offers available[/]")
            return
        offer = offers[0]
        offer_id = offer["id"]
        price = round(offer.get("dph_total", 0), 3)
        gpu_name = offer.get("gpu_name", "Unknown")
        offer_gpus = offer.get("num_gpus", 1)
        offer_vram = round(offer.get("gpu_ram", 0) / 1024, 0)
        console.print(
            f"  Found: [green]{offer_gpus}x {gpu_name} ({offer_vram}GB each)[/] "
            f"at [cyan]${price}/hr[/]"
        )
    except json.JSONDecodeError:
        console.print("[red]Error parsing offer response[/]")
        return

    # Create instance
    console.print("\n[yellow]Creating instance...[/]")
    code, stdout, stderr = run_vastai_cmd(
        [
            "create",
            "instance",
            str(offer_id),
            "--image",
            DEFAULT_IMAGE,
            "--disk",
            str(disk_gb),
            "--ssh",
            "--raw",
        ],
        config,
    )

    # Parse instance ID
    instance_id = None
    try:
        result = json.loads(stdout)
        instance_id = result.get("new_contract")
    except json.JSONDecodeError:
        import re

        match = re.search(r"(\d+)", stdout)
        if match:
            instance_id = match.group(1)

    if not instance_id:
        console.print("[red]Error: Could not get instance ID[/]")
        console.print(stdout)
        return

    console.print(f"  Instance ID: [green]{instance_id}[/]")

    # Wait for instance to be ready
    console.print("\n[yellow]Waiting for instance to start...[/]")
    for attempt in range(30):
        time.sleep(5)
        console.print(f"  Checking status (attempt {attempt + 1}/30)...", end="\r")

        ssh_info = get_ssh_info(instance_id, config)
        if ssh_info:
            host, port = ssh_info
            console.print()
            console.print()
            console.print(
                Panel.fit(
                    f"[bold green]Instance Ready![/]\n\n"
                    f"Instance ID: [cyan]{instance_id}[/]\n"
                    f"SSH Host: [cyan]{host}[/]\n"
                    f"SSH Port: [cyan]{port}[/]\n\n"
                    f"Next steps:\n"
                    f"  heretic-vast setup    # Install heretic\n"
                    f"  heretic-vast run MODEL  # Abliterate\n"
                    f"  heretic-vast stop     # Stop when done",
                    title="Success",
                    border_style="green",
                )
            )
            return

    console.print("\n[yellow]Instance created but SSH not ready yet.[/]")
    console.print(f"Instance ID: {instance_id}")
    console.print("Check status: heretic-vast list")


@cli.command("list")
@click.pass_context
def list_instances(ctx):
    """List your Vast.ai instances."""
    config = ctx.obj["config"]
    code, stdout, stderr = run_vastai_cmd(["show", "instances"], config)
    console.print(stdout)


@cli.command("setup")
@click.argument("instance_id", required=False)
@click.pass_context
def setup_instance(ctx, instance_id: Optional[str]):
    """Install heretic on a Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id_int = int(instance_id)
        validate_instance_id(instance_id_int)
        instance_id = str(instance_id_int)  # Use validated value
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    console.print(
        Panel.fit(
            f"Setting up Heretic on Vast.ai\n\nInstance ID: [cyan]{instance_id}[/]",
            title="Setup",
        )
    )

    conn = get_connection(instance_id, config)
    if not conn:
        return

    with console.status("[bold green]Connecting...") as status:
        try:
            conn.open()
        except TimeoutError as e:
            console.print("[red]SSH Connection Timeout[/]")
            console.print("[yellow]Solutions:[/]")
            console.print("  1. Check instance is running (heretic-vast list)")
            console.print("  2. Verify instance SSH port is accessible")
            console.print("  3. Check network connection")
            raise SSHError(f"SSH connection timeout to instance {instance_id}") from e
        except PermissionError as e:
            console.print("[red]SSH Authentication Failed[/]")
            console.print("[yellow]Check SSH key permissions and configuration[/]")
            raise SSHError(
                f"SSH authentication failed to instance {instance_id}"
            ) from e
        except ConnectionRefusedError as e:
            console.print("[red]Connection Refused[/]")
            console.print("[yellow]Instance may not be ready yet - wait and retry[/]")
            raise SSHError(f"Connection refused to instance {instance_id}") from e
        except EOFError as e:
            console.print("[red]SSH Connection Closed Unexpectedly[/]")
            console.print("[yellow]Instance may have stopped or network issue[/]")
            raise SSHError(f"SSH connection closed to instance {instance_id}") from e
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Safety net for unexpected SSH errors
            console.print(f"[red]SSH connection failed: {e}[/]")
            console.print("[yellow]Check instance status and network[/]")
            raise SSHError(
                f"SSH connection failed to instance {instance_id}: {e}"
            ) from e

        status.update("[bold green]Installing git...")
        run_ssh_command(
            conn,
            "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1",
            timeout=300,  # Package installation can be slow
            critical=True,  # Must succeed
        )
        console.print("  [green]✓[/] git installed")

        status.update("[bold green]Configuring workspace...")
        run_ssh_command(
            conn,
            "mkdir -p /workspace/.cache/huggingface /workspace/models",
            critical=True,  # Must succeed
        )
        run_ssh_command(conn, "export HF_HOME=/workspace/.cache/huggingface")
        run_ssh_command(
            conn,
            "grep -q 'HF_HOME' ~/.bashrc || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc",
        )
        console.print("  [green]✓[/] workspace configured")

        status.update("[bold green]Installing heretic...")
        run_ssh_command(
            conn,
            "pip install --quiet git+https://github.com/quanticsoul4772/abliteration-workflow.git",
            timeout=300,  # Git clone + pip install can be slow
            critical=True,  # Must succeed
        )
        console.print("  [green]✓[/] heretic installed (from abliteration-workflow)")

        status.update("[bold green]Checking GPU...")
        gpu_info = run_ssh_command(
            conn,
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            timeout=30,  # Quick command
        )
        console.print(f"  [green]✓[/] GPU: {gpu_info.strip()}")

        conn.close()

    console.print()
    console.print("[bold green]Setup complete![/]")
    console.print("\nNext: heretic-vast run MODEL")


@cli.command("run")
@click.argument("model")
@click.argument("instance_id", required=False)
@click.pass_context
def run_abliteration(ctx, model: str, instance_id: Optional[str]):
    """Run heretic abliteration on a model."""
    config = ctx.obj["config"]

    # Validate model name
    try:
        model = validate_model_name(model)
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id_int = int(instance_id)
        validate_instance_id(instance_id_int)
        instance_id = str(instance_id_int)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    model_name = model.replace("/", "-")
    output_path = f"{MODELS_DIR}/{model_name}-heretic"

    console.print(
        Panel.fit(
            f"Running Heretic Abliteration\n\n"
            f"Model: [cyan]{model}[/]\n"
            f"Output: [cyan]{output_path}[/]\n"
            f"Instance: [cyan]{instance_id}[/]",
            title="Abliteration",
        )
    )

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()
        console.print("\n[yellow]Starting abliteration (this will take a while)...[/]")
        console.print(
            "[dim]Run 'heretic-vast watch' in another terminal to monitor progress[/]"
        )
        console.print()

        # Run heretic with live output
        cmd = (
            f"export HF_HOME=/workspace/.cache/huggingface && "
            f"cd /workspace && "
            f"heretic {model} --auto-select --auto-select-path {output_path}"
        )
        # Stream the output
        conn.run(cmd, hide=False, warn=True)
        conn.close()

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("exec")
@click.argument("command")
@click.argument("instance_id", required=False)
@click.pass_context
def exec_command(ctx, command: str, instance_id: Optional[str]):
    """Execute a command on a Vast.ai instance."""
    config = ctx.obj["config"]

    # Validate command (basic safety check - don't allow dangerous operations)
    if not command or not isinstance(command, str):
        console.print("[red]Command must be a non-empty string[/]")
        return
    if len(command) > 10000:
        console.print("[red]Command too long (max 10000 chars)[/]")
        return

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id_int = int(instance_id)
        validate_instance_id(instance_id_int)
        instance_id = str(instance_id_int)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()
        conn.run(command, hide=False, warn=True)
        conn.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("status")
@click.argument("instance_id", required=False)
@click.pass_context
def show_status(ctx, instance_id: Optional[str]):
    """Show GPU status on a Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()
        output = run_ssh_command(
            conn,
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits",
        )

        table = Table(title="GPU Status")
        table.add_column("GPU", style="cyan")
        table.add_column("Utilization", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Temp", justify="right")

        for i, line in enumerate(output.strip().split("\n")):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        name, util, mem_used, mem_total, temp = parts[:5]
                        mem_used_gb = round(int(mem_used) / 1024, 1)
                        mem_total_gb = round(int(mem_total) / 1024, 1)
                        table.add_row(
                            f"{i}: {name}",
                            f"{util}%",
                            f"{mem_used_gb}/{mem_total_gb} GB",
                            f"{temp}°C",
                        )
                    except ValueError:
                        pass

        console.print(table)
        conn.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("progress")
@click.argument("instance_id", required=False)
@click.pass_context
def show_progress(ctx, instance_id: Optional[str]):
    """Check abliteration progress on a Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()

        # Process status
        console.print("\n[bold]Process Status[/]")
        proc = run_ssh_command(conn, "ps aux | grep '[h]eretic' | head -1")
        if proc.strip():
            console.print("  [green]● Running[/]")
            # Extract model name if possible
            if "--model" in proc or "/" in proc:
                console.print(f"  {proc.strip()[:100]}...")
        else:
            console.print("  [yellow]○ Not running[/]")

        # GPU status
        console.print("\n[bold]GPU Status[/]")
        gpu_info = run_ssh_command(
            conn,
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
        )
        for i, line in enumerate(gpu_info.strip().split("\n")):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        name, util, mem_used, mem_total = parts[:4]
                        mem_used_gb = round(int(mem_used) / 1024, 1)
                        mem_total_gb = round(int(mem_total) / 1024, 1)
                        console.print(
                            f"  GPU {i}: {name} - {util}% util, {mem_used_gb}/{mem_total_gb} GB"
                        )
                    except ValueError:
                        pass

        # Models
        console.print("\n[bold]Output Models[/]")
        models = run_ssh_command(conn, f"ls -1 {MODELS_DIR}/ 2>/dev/null")
        if models.strip():
            for model in models.strip().split("\n"):
                if model.strip():
                    console.print(f"  [green]✓[/] {model.strip()}")
        else:
            console.print("  [dim](No models saved yet)[/]")

        conn.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("watch")
@click.argument("instance_id", required=False)
@click.option("--interval", "-i", default=10, help="Refresh interval in seconds")
@click.pass_context
def watch_dashboard(ctx, instance_id: Optional[str], interval: int):
    """Live dashboard for monitoring abliteration progress."""
    config = ctx.obj["config"]

    # Validate interval
    if not isinstance(interval, int) or interval < 1 or interval > 300:
        console.print("[red]Invalid interval: must be 1-300 seconds[/]")
        return

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()
    except Exception as e:
        console.print(f"[red]SSH connection failed: {e}[/]")
        return

    start_time = time.time()

    def make_dashboard() -> Panel:
        """Generate the dashboard layout."""
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        current_time = time.strftime("%H:%M:%S")

        # Collect data (with error handling for connection drops)
        try:
            proc = run_ssh_command(conn, "ps aux | grep '[h]eretic' | head -1")
            gpu_info = run_ssh_command(
                conn,
                "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits",
            )
            models = run_ssh_command(conn, f"ls -1 {MODELS_DIR}/ 2>/dev/null | head -5")
            disk = run_ssh_command(
                conn,
                'df -h /workspace 2>/dev/null | tail -1 | awk \'{print $3"/"$2" ("$5" used)}\'',
            )
        except (OSError, TimeoutError, EOFError):
            # Connection errors: socket issues, timeouts, connection closed
            proc = ""
            gpu_info = ""
            models = ""
            disk = "(connection error)"

        # Build layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Header
        header_text = Text()
        header_text.append("VAST.AI ABLITERATION DASHBOARD", style="bold cyan")
        header_text.append(
            f"  │  Instance: {instance_id}  │  Time: {current_time}  │  Watch: {elapsed_str}"
        )
        layout["header"].update(Panel(header_text, style="cyan"))

        # Process panel
        process_table = Table(show_header=False, box=None, padding=(0, 1))
        process_table.add_column("Key", style="bold")
        process_table.add_column("Value")

        if proc.strip():
            process_table.add_row("Status", "[green]● RUNNING[/]")

            # Extract model name from command line (e.g., "heretic Qwen/Qwen2.5-72B-Instruct")
            # Look for pattern after "heretic" that looks like a HuggingFace model
            model_match = re.search(
                r"heretic\s+(?:--model\s+)?([A-Za-z0-9_-]+/[A-Za-z0-9._-]+)", proc
            )
            if model_match:
                process_table.add_row("Model", f"[cyan]{model_match.group(1)}[/]")

            # Extract runtime from ps output (format: HH:MM or M:SS in TIME column)
            # ps aux format: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
            time_match = re.search(
                r"\s+(\d+:\d+)\s+(?:/opt/conda/bin/)?(?:python\s+)?(?:/opt/conda/bin/)?heretic",
                proc,
            )
            if time_match:
                process_table.add_row("Runtime", f"[yellow]{time_match.group(1)}[/]")

            # Extract CPU usage
            cpu_match = re.search(r"^\S+\s+\d+\s+([\d.]+)\s+", proc)
            if cpu_match:
                process_table.add_row("CPU", f"{cpu_match.group(1)}%")
        else:
            process_table.add_row("Status", "[yellow]○ NOT RUNNING[/]")
            process_table.add_row("", "[dim]Abliteration complete or not started[/]")

        layout["left"].split_column(
            Layout(
                Panel(process_table, title="Process", border_style="white"),
                name="process",
            ),
            Layout(name="models"),
        )

        # Models panel
        models_content = ""
        if models.strip():
            for m in models.strip().split("\n"):
                if m.strip():
                    models_content += f"[green]✓[/] {m.strip()}\n"
        else:
            models_content = "[dim](No models saved yet)[/]"
        layout["models"].update(
            Panel(models_content.strip(), title="Output Models", border_style="white")
        )

        # GPU panel
        gpu_table = Table(show_header=True, box=None)
        gpu_table.add_column("GPU", style="cyan")
        gpu_table.add_column("Util", justify="right")
        gpu_table.add_column("VRAM", justify="right")
        gpu_table.add_column("Temp", justify="right")
        gpu_table.add_column("Power", justify="right")

        for i, line in enumerate(gpu_info.strip().split("\n")):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    name, util, mem_used, mem_total, temp, power = parts[:6]
                    try:
                        mem_used_gb = round(int(mem_used) / 1024, 1)
                        mem_total_gb = round(int(mem_total) / 1024, 1)
                        mem_pct = round((int(mem_used) / int(mem_total)) * 100)

                        # Color code utilization
                        util_int = int(util)
                        util_style = (
                            "green"
                            if util_int > 80
                            else "yellow"
                            if util_int > 30
                            else "dim"
                        )

                        gpu_table.add_row(
                            f"{i}: {name[:25]}",
                            f"[{util_style}]{util}%[/]",
                            f"{mem_used_gb}/{mem_total_gb}GB ({mem_pct}%)",
                            f"{temp}°C",
                            f"{power}W",
                        )
                    except ValueError:
                        pass

        layout["right"].update(Panel(gpu_table, title="GPUs", border_style="white"))

        # Footer
        footer_text = Text()
        footer_text.append(f"Disk: {disk.strip()}  │  ", style="dim")
        footer_text.append(
            f"Refresh: {interval}s  │  Press Ctrl+C to exit", style="dim"
        )
        layout["footer"].update(Panel(footer_text, style="dim"))

        return Panel(layout, title="[bold]Heretic[/]", border_style="cyan")

    console.print("[dim]Starting live dashboard (Ctrl+C to exit)...[/]")

    try:
        with Live(make_dashboard(), refresh_per_second=0.5, screen=True) as live:
            while True:
                time.sleep(interval)
                live.update(make_dashboard())
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        console.print("\n[dim]Dashboard closed.[/]")


@cli.command("models")
@click.argument("instance_id", required=False)
@click.pass_context
def list_models(ctx, instance_id: Optional[str]):
    """List abliterated models on a Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()
        console.print(f"\n[bold]Models in {MODELS_DIR}:[/]\n")

        models = run_ssh_command(
            conn, f"ls -lh {MODELS_DIR}/ 2>/dev/null | grep -v '^total'"
        )
        if models.strip():
            console.print(models)
        else:
            console.print("[dim](No models found)[/]")

        console.print()
        sizes = run_ssh_command(conn, f"du -sh {MODELS_DIR}/*/ 2>/dev/null")
        if sizes.strip():
            console.print("[bold]Model sizes:[/]")
            console.print(sizes)

        conn.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")

    console.print("\nDownload a model: heretic-vast download MODEL_NAME")


@cli.command("download")
@click.argument("model_name", required=False)
@click.argument("instance_id", required=False)
@click.option(
    "--local-dir", "-d", default="./models", help="Local directory to save model"
)
@click.pass_context
def download_model(
    ctx, model_name: Optional[str], instance_id: Optional[str], local_dir: str
):
    """Download an abliterated model from Vast.ai."""
    config = ctx.obj["config"]

    # Validate local_dir path
    try:
        local_path = validate_path(local_dir, must_exist=False, must_be_dir=False)
    except ConfigurationError as e:
        console.print(f"[red]Invalid local directory: {e}[/]")
        return

    # Validate model_name if provided
    if model_name:
        try:
            model_name = validate_model_name(model_name)
        except ConfigurationError as e:
            console.print(f"[red]Validation Error: {e}[/]")
            return

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    conn = get_connection(instance_id, config)
    if not conn:
        return

    try:
        conn.open()

        # If no model specified, list available and let user choose
        if not model_name:
            console.print("[yellow]Scanning for available models...[/]")
            models_output = run_ssh_command(
                conn, f"ls -1 {MODELS_DIR}/ 2>/dev/null | grep -E '.*-heretic$'"
            )
            models = [m.strip() for m in models_output.strip().split("\n") if m.strip()]

            if not models:
                console.print("[red]No abliterated models found[/]")
                console.print("Run abliteration first: heretic-vast run MODEL")
                return

            if len(models) == 1:
                model_name = models[0]
                console.print(f"Found model: [green]{model_name}[/]")
            else:
                console.print("\nAvailable models:")
                for i, m in enumerate(models, 1):
                    console.print(f"  [{i}] {m}")
                choice = console.input("\nSelect model number: ")
                try:
                    model_name = models[int(choice) - 1]
                except (ValueError, IndexError):
                    model_name = models[-1]  # Default to most recent

        remote_path = f"{MODELS_DIR}/{model_name}"
        local_path = Path(local_dir) / model_name

        # Get model size
        size = run_ssh_command(conn, f"du -sh {remote_path} 2>/dev/null | cut -f1")
        console.print(f"\nModel: [cyan]{model_name}[/]")
        console.print(f"Size: [cyan]{size.strip()}[/]")
        console.print(f"Remote: [dim]{remote_path}[/]")
        console.print(f"Local: [dim]{local_path}[/]")

        console.print(
            "\n[yellow]Note: Large models (70B+) can take 30-60+ minutes to download.[/]"
        )
        if not console.input("\nStart download? [y/N]: ").lower().startswith("y"):
            console.print("Cancelled.")
            return

        conn.close()

        # Get SSH info for scp
        console.print("\n[dim]Getting SSH connection info...[/]")
        ssh_info = get_ssh_info(instance_id, config, verbose=True)
        if not ssh_info:
            console.print("[red]Could not get SSH info for download[/]")
            console.print("\n[yellow]Troubleshooting:[/]")
            console.print("  1. Check VAST_API_KEY is set in .env file")
            console.print("  2. Run: heretic-vast list (to verify API works)")
            console.print(
                "  3. Try PowerShell script: .\\runpod.ps1 vast-download-model"
            )
            return

        host, port = ssh_info
        console.print(f"[dim]SSH target: root@{host}:{port}[/]")

        # Create local directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        console.print("\n[bold green]Starting download...[/]")
        start_time = time.time()

        # Use scp for download with progress
        scp_cmd = [
            "scp",
            "-r",
            "-o",
            "StrictHostKeyChecking=no",
            "-P",
            str(port),
            f"root@{host}:{remote_path}",
            str(local_path),
        ]

        result = subprocess.run(scp_cmd)

        duration = time.time() - start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))

        if result.returncode == 0:
            console.print(
                Panel.fit(
                    f"[bold green]Model Downloaded Successfully![/]\n\n"
                    f"Model: [cyan]{model_name}[/]\n"
                    f"Location: [cyan]{local_path}[/]\n"
                    f"Duration: [cyan]{duration_str}[/]\n\n"
                    f"Don't forget to stop the instance:\n"
                    f"  heretic-vast stop",
                    title="Success",
                    border_style="green",
                )
            )
        else:
            console.print(f"[red]Download failed (exit code: {result.returncode})[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("stop")
@click.argument("instance_id", required=False)
@click.pass_context
def stop_instance(ctx, instance_id: Optional[str]):
    """Stop a Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    console.print(f"Stopping instance [cyan]{instance_id}[/]...")
    code, stdout, stderr = run_vastai_cmd(
        ["stop", "instance", str(instance_id)], config
    )

    console.print(
        Panel.fit(
            f"[bold green]Instance Stopped - Billing Paused[/]\n\n"
            f"Restart with: heretic-vast start {instance_id}",
            title="Stopped",
            border_style="green",
        )
    )


@cli.command("start")
@click.argument("instance_id", required=False)
@click.pass_context
def start_instance(ctx, instance_id: Optional[str]):
    """Start a stopped Vast.ai instance."""
    config = ctx.obj["config"]

    if not instance_id:
        instances = get_instances(config)
        for inst in instances:
            if inst.get("actual_status") != "running":
                instance_id = str(inst["id"])
                break
        if not instance_id:
            console.print("[red]Error: No stopped instance found[/]")
            return

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    console.print(f"Starting instance [cyan]{instance_id}[/]...")
    code, stdout, stderr = run_vastai_cmd(
        ["start", "instance", str(instance_id)], config
    )
    console.print(stdout)
    console.print("\n[green]Instance starting![/]")
    console.print("Wait ~30 seconds, then run: heretic-vast list")


@cli.command("terminate")
@click.argument("instance_id")
@click.pass_context
def terminate_instance(ctx, instance_id: str):
    """Permanently destroy a Vast.ai instance."""
    config = ctx.obj["config"]

    # Validate instance_id BEFORE asking for confirmation
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    if (
        not console.input(
            f"Are you sure you want to [red]DESTROY[/] instance {instance_id}? [y/N]: "
        )
        .lower()
        .startswith("y")
    ):
        console.print("Cancelled.")
        return

    console.print(f"Destroying instance [red]{instance_id}[/]...")
    code, stdout, stderr = run_vastai_cmd(
        ["destroy", "instance", str(instance_id)], config
    )
    console.print("[green]Instance destroyed.[/]")


@cli.command("connect")
@click.argument("instance_id", required=False)
@click.pass_context
def connect_ssh(ctx, instance_id: Optional[str]):
    """SSH to a Vast.ai instance (interactive)."""
    config = ctx.obj["config"]

    if not instance_id:
        inst = get_running_instance(config)
        if not inst:
            console.print("[red]Error: No running instance found[/]")
            return
        instance_id = str(inst["id"])

    # Validate instance_id
    try:
        instance_id = validate_and_convert_instance_id(instance_id)
    except ValueError:
        console.print(
            f"[red]Invalid instance ID: must be a number, got '{instance_id}'[/]"
        )
        return
    except ConfigurationError as e:
        console.print(f"[red]Validation Error: {e}[/]")
        return

    ssh_info = get_ssh_info(instance_id, config)
    if not ssh_info:
        console.print("[red]Could not get SSH info[/]")
        return

    host, port = ssh_info
    console.print(f"Connecting to [cyan]{host}:{port}[/]...")

    # Launch interactive SSH
    os.execvp(
        "ssh",
        ["ssh", "-p", str(port), f"root@{host}", "-o", "StrictHostKeyChecking=no"],
    )


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
