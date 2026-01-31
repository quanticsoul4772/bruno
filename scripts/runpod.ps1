# GPU Cloud Connection Helper for Windows
# Heretic LLM Abliteration Automation
#
# Supports both RunPod and Vast.ai (50% cheaper!)
#
# This script uses a hybrid approach:
# - runpodctl/vastai CLI for pod management and SSH info (fast, reliable)
# - WSL + SSH heredoc for command execution (required for PTY support)
# - GraphQL/REST API as fallback and for advanced operations

param(
    [Parameter(Position=0)]
    [string]$Action = "help",

    [Parameter(Position=1)]
    [string]$Arg1,

    [Parameter(Position=2)]
    [string]$Arg2
)

# ===== LOAD .ENV FILE =====
# Automatically load environment variables from .env file if it exists
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($value -and $value -ne 'your_api_key_here') {
                [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }
    }
}
# Supported .env variables:
#   RUNPOD_API_KEY - RunPod API key
#   VAST_API_KEY   - Vast.ai API key (for 50% cheaper GPU access)
#   HF_TOKEN       - HuggingFace token for gated models

# ===== CONFIGURE THESE =====
# API key loaded from: .env file > environment variable > empty
# Create .env file: copy .env.example to .env and add your key
$RUNPOD_API_KEY = if ($env:RUNPOD_API_KEY) { $env:RUNPOD_API_KEY } else { "" }
$HF_TOKEN = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { "" }
$RUNPOD_HOST = "203.57.40.210"     # Auto-configured by create-pod, or set manually
$RUNPOD_PORT = "10290"     # Auto-configured by create-pod, or set manually
$RUNPOD_USER = "root"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519"  # Your SSH key

# RunPod SSH proxy (more reliable than direct TCP)
# Format: podId-userId@ssh.runpod.io
# IMPORTANT: userId is your RunPod account ID (constant), NOT the API's myself.id
# Find your userId in the RunPod console SSH connection string
$RUNPOD_USER_ID = "64411784"  # Your RunPod account ID - stays constant across pods
$RUNPOD_SSH_PROXY = "znwwcgs2lwcra3-$RUNPOD_USER_ID@ssh.runpod.io"  # Auto-updated by create-pod
$USE_SSH_PROXY = $true  # Using RunPod proxy since direct TCP isn't working
# HF_TOKEN is loaded from .env file above

# runpodctl Configuration
# runpodctl is used for pod management and getting SSH connection info
# Download: https://github.com/runpod/runpodctl/releases
$RUNPODCTL_PATH = "$PSScriptRoot\runpodctl.exe"  # Path to runpodctl executable

# ===== VAST.AI CONFIGURATION =====
# Vast.ai offers 50% cheaper GPU access ($0.16-0.25/hr for RTX 4090 vs $0.34/hr on RunPod)
# API key loaded from: .env file (VAST_API_KEY) > environment variable > empty
$VAST_API_KEY = if ($env:VAST_API_KEY) { $env:VAST_API_KEY } else { "" }
$VASTAI_CLI_PATH = "$PSScriptRoot\vast.exe"  # Path to vast CLI (or 'vast' if in PATH)

# GPU Tier Configurations
# Each tier defines: GPU name filter, max price, min VRAM, recommended models
$GPU_TIERS = @{
    "RTX_4090" = @{
        GpuName = "RTX_4090"
        MaxPrice = 0.50
        MinVram = 24
        DiskGB = 50
        Description = "24GB VRAM - Good for 7B-8B models"
    }
    "A6000" = @{
        GpuName = "RTX_A6000"
        MaxPrice = 0.80
        MinVram = 48
        DiskGB = 80
        Description = "48GB VRAM - Good for 14B-30B models"
    }
    "A100_40GB" = @{
        GpuName = "A100"
        MaxPrice = 1.00
        MinVram = 40
        DiskGB = 100
        Description = "40GB VRAM - Good for 14B-32B models"
    }
    "A100_80GB" = @{
        GpuName = "A100"
        MaxPrice = 2.00
        MinVram = 80
        DiskGB = 150
        Description = "80GB VRAM - Good for 32B-70B models"
    }
    "A100_SXM" = @{
        GpuName = "A100_SXM4"
        MaxPrice = 2.50
        MinVram = 80
        DiskGB = 150
        Description = "80GB VRAM SXM4 - Fastest A100 variant"
    }
    "H100" = @{
        GpuName = "H100"
        MaxPrice = 4.00
        MinVram = 80
        DiskGB = 150
        Description = "80GB VRAM - Fastest, good for 70B+ models"
    }
}

# Vast.ai instance defaults
$VAST_DEFAULT_GPU = "RTX_4090"  # Default GPU tier
$VAST_DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"  # Docker image
$VAST_DEFAULT_DISK_GB = 50  # Disk space in GB (overridden by tier)
$VAST_MIN_DOWNLOAD = 200  # Minimum download speed in Mbps
$VAST_MAX_PRICE = 0.50  # Maximum price per hour (overridden by tier)
$VAST_DEFAULT_NUM_GPUS = 1  # Number of GPUs (for multi-GPU support)

# Vast.ai SSH settings (auto-configured by vast-create-pod)
$VAST_SSH_HOST = ""
$VAST_SSH_PORT = ""
$VAST_INSTANCE_ID = ""

# WSL SSH Configuration
# The RunPod SSH proxy requires a PTY (pseudo-terminal) which Windows SSH can't provide.
# Solution: Use WSL (Windows Subsystem for Linux) with heredoc to allocate a proper PTY.
# Note: runpodctl exec only supports Python scripts, not general shell commands.
$USE_WSL_SSH = $true  # Set to $false if WSL is not available

# Pod creation defaults
# NOTE: RTX 4090 has 24GB VRAM, good for 8B models
$DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
# PyTorch 2.4.0 is compatible with transformers 4.55+ (no manual upgrade needed!)
# Python 3.11, CUDA 12.4.1 - works out of the box with bruno-ai
$DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# IMPORTANT: 20GB container disk is NOT enough (fills with pip packages)
# 40GB minimum; 50GB+ recommended for larger models
$DEFAULT_VOLUME_GB = 50
$DEFAULT_CONTAINER_DISK_GB = 40
# ===========================

$LOCAL_DIR = "C:\Development\Projects\bruno"
$LOCAL_MODELS_DIR = "C:\Development\Projects\bruno\models"  # Local directory for downloaded models
$REMOTE_DIR = "/workspace/bruno"
$VAST_REMOTE_DIR = "/workspace"  # Vast.ai uses /workspace as default
$VAST_MODELS_DIR = "/workspace/models"  # Remote directory for abliterated models
$VLLM_PORT = 8000

# ===== RUNPODCTL FUNCTIONS =====

function Test-RunpodctlAvailable {
    if (Test-Path $RUNPODCTL_PATH) {
        return $true
    }
    # Check if it's in PATH
    $inPath = Get-Command runpodctl -ErrorAction SilentlyContinue
    if ($inPath) {
        Set-Variable -Name RUNPODCTL_PATH -Value "runpodctl" -Scope Script
        return $true
    }
    return $false
}

function Invoke-Runpodctl {
    param([string]$Arguments)
    if (-not (Test-RunpodctlAvailable)) {
        Write-Host "ERROR: runpodctl not found at $RUNPODCTL_PATH" -ForegroundColor Red
        Write-Host "Download from: https://github.com/runpod/runpodctl/releases" -ForegroundColor Yellow
        return $null
    }
    $cmd = "& `"$RUNPODCTL_PATH`" $Arguments"
    return Invoke-Expression $cmd
}

function Get-RunpodctlSSHInfo {
    param([string]$PodId)

    if (-not (Test-RunpodctlAvailable)) {
        return $null
    }

    # runpodctl ssh connect gives us the direct TCP connection info
    $output = if ($PodId) {
        Invoke-Runpodctl "ssh connect $PodId"
    } else {
        Invoke-Runpodctl "ssh connect"
    }

    # Parse: ssh root@IP -p PORT  # pod { id: "xxx", name: "xxx" }
    if ($output -match 'ssh root@([\d.]+) -p (\d+).*id: "([^"]+)"') {
        return @{
            Host = $matches[1]
            Port = $matches[2]
            PodId = $matches[3]
        }
    }
    return $null
}

function Get-FirstRunningPodId {
    if (Test-RunpodctlAvailable) {
        $output = Invoke-Runpodctl "get pod"
        # Parse the table output - look for RUNNING pods
        $lines = $output -split "`n" | Where-Object { $_ -match 'RUNNING' }
        if ($lines) {
            $firstLine = $lines[0]
            if ($firstLine -match '^(\S+)') {
                return $matches[1]
            }
        }
    }
    return $null
}

# ===== VAST.AI CLI FUNCTIONS =====

function Test-VastCliAvailable {
    if (Test-Path $VASTAI_CLI_PATH) {
        return $true
    }
    # Check for .bat wrapper (created by install-vastcli)
    $batPath = "$PSScriptRoot\vast.bat"
    if (Test-Path $batPath) {
        Set-Variable -Name VASTAI_CLI_PATH -Value $batPath -Scope Script
        return $true
    }
    # Check if it's in PATH
    $inPath = Get-Command vast -ErrorAction SilentlyContinue
    if ($inPath) {
        Set-Variable -Name VASTAI_CLI_PATH -Value "vast" -Scope Script
        return $true
    }
    # Check if vastai is available in WSL
    if (Test-WSLAvailable) {
        $wslCheck = wsl -e bash -c "which vastai 2>/dev/null"
        if ($wslCheck) {
            return $true
        }
    }
    return $false
}

function Invoke-VastCli {
    param([string]$Arguments)
    if (-not (Test-VastCliAvailable)) {
        Write-Host "ERROR: vast CLI not found at $VASTAI_CLI_PATH" -ForegroundColor Red
        Write-Host "Install with: .\runpod.ps1 install-vastcli" -ForegroundColor Yellow
        return $null
    }
    # Set API key if available
    if ($VAST_API_KEY) {
        $env:VAST_API_KEY = $VAST_API_KEY
    }
    $cmd = "& `"$VASTAI_CLI_PATH`" $Arguments"
    return Invoke-Expression $cmd
}

function Get-VastSSHInfo {
    param([string]$InstanceId)

    if (-not (Test-VastCliAvailable)) {
        return $null
    }

    # Get SSH URL for the instance
    $output = Invoke-VastCli "ssh-url $InstanceId"

    # Parse: ssh -p PORT root@IP or similar
    if ($output -match '-p\s+(\d+)\s+\S+@([\d.]+)') {
        return @{
            Host = $matches[2]
            Port = $matches[1]
            InstanceId = $InstanceId
        }
    }
    # Alternative format: ssh://root@IP:PORT
    if ($output -match 'ssh://\S+@([\d.]+):(\d+)') {
        return @{
            Host = $matches[1]
            Port = $matches[2]
            InstanceId = $InstanceId
        }
    }
    return $null
}

function Get-FirstVastInstance {
    # Returns the first RUNNING instance, or the configured instance if it's running
    # Priority: 1) $VAST_INSTANCE_ID if running, 2) First running instance, 3) First instance
    if (Test-VastCliAvailable) {
        $output = Invoke-VastCli "show instances --raw"
        if ($output) {
            try {
                $instances = $output | ConvertFrom-Json
                if ($instances -and $instances.Count -gt 0) {
                    # If we have a configured instance ID, check if it's running
                    if ($VAST_INSTANCE_ID) {
                        $configuredInstance = $instances | Where-Object { $_.id -eq [int]$VAST_INSTANCE_ID }
                        if ($configuredInstance -and ($configuredInstance.actual_status -eq "running" -or $configuredInstance.status -eq "running")) {
                            Write-Host "  Using configured instance: $VAST_INSTANCE_ID" -ForegroundColor Gray
                            return $VAST_INSTANCE_ID
                        }
                    }

                    # Find first running instance
                    $runningInstance = $instances | Where-Object { $_.actual_status -eq "running" -or $_.status -eq "running" } | Select-Object -First 1
                    if ($runningInstance) {
                        Write-Host "  Found running instance: $($runningInstance.id)" -ForegroundColor Gray
                        return $runningInstance.id
                    }

                    # Fallback to first instance (might be loading/stopped)
                    Write-Host "  No running instances, using first instance: $($instances[0].id)" -ForegroundColor Gray
                    return $instances[0].id
                }
            } catch {
                # JSON parsing failed, falling back to text parsing
                Write-Host "Note: JSON parsing failed, using text fallback" -ForegroundColor Gray
                $lines = $output -split "`n" | Where-Object { $_ -match '^\d+' }
                if ($lines) {
                    if ($lines[0] -match '^(\d+)') {
                        return $matches[1]
                    }
                }
            }
        }
    }
    return $null
}

function Update-VastScriptConfig {
    param(
        [string]$HostAddress,
        [string]$Port,
        [string]$InstanceId
    )

    $scriptPath = $PSCommandPath
    $content = Get-Content $scriptPath -Raw

    if ($HostAddress) {
        $content = $content -replace '\$VAST_SSH_HOST = "[^"]*"', "`$VAST_SSH_HOST = `"$HostAddress`""
        Set-Variable -Name VAST_SSH_HOST -Value $HostAddress -Scope Script
    }

    if ($Port) {
        $content = $content -replace '\$VAST_SSH_PORT = "[^"]*"', "`$VAST_SSH_PORT = `"$Port`""
        Set-Variable -Name VAST_SSH_PORT -Value $Port -Scope Script
    }

    if ($InstanceId) {
        $content = $content -replace '\$VAST_INSTANCE_ID = "[^"]*"', "`$VAST_INSTANCE_ID = `"$InstanceId`""
        Set-Variable -Name VAST_INSTANCE_ID -Value $InstanceId -Scope Script
    }

    Set-Content $scriptPath -Value $content -NoNewline
}

function Get-VastSSHTarget {
    param([string]$InstanceId)

    if (-not (Test-WSLAvailable)) {
        Write-Host "ERROR: WSL required" -ForegroundColor Red
        return $null
    }

    if ($VAST_API_KEY) {
        $env:VAST_API_KEY = $VAST_API_KEY
    }

    # Get SSH info
    $sshUrl = wsl -e bash -c "vastai ssh-url $InstanceId" 2>$null

    # Format 1: ssh -p PORT root@HOST (IP address)
    if ($sshUrl -match '-p\s+(\d+)\s+\S+@([\d.]+)') {
        return "root@$($matches[2]) -p $($matches[1])"
    }
    # Format 2: ssh://root@HOST:PORT (IP address)
    elseif ($sshUrl -match 'ssh://\S+@([\d.]+):(\d+)') {
        return "root@$($matches[1]) -p $($matches[2])"
    }
    # Format 3: ssh://root@HOSTNAME:PORT (hostname like ssh1.vast.ai)
    elseif ($sshUrl -match 'ssh://([^@]+)@([^:]+):(\d+)') {
        $sshUser = $matches[1]
        $sshHost = $matches[2]
        $sshPort = $matches[3]
        return "$sshUser@$sshHost -p $sshPort"
    }
    # Format 4: ssh -p PORT user@HOSTNAME (hostname)
    elseif ($sshUrl -match '-p\s+(\d+)\s+([^@]+)@(\S+)') {
        $sshPort = $matches[1]
        $sshUser = $matches[2]
        $sshHost = $matches[3]
        return "$sshUser@$sshHost -p $sshPort"
    }

    Write-Host "ERROR: Could not parse SSH URL: $sshUrl" -ForegroundColor Red
    return $null
}

function Invoke-VastSSHCommand {
    param(
        [string]$Commands,
        [string]$InstanceId,
        [int]$TimeoutSeconds = 30,
        [switch]$Quiet
    )

    # Ensure WSL has the SSH key with proper permissions
    Ensure-WSLSSHKey | Out-Null

    if (-not (Test-WSLAvailable)) {
        Write-Host "ERROR: WSL required" -ForegroundColor Red
        return $null
    }

    if ($VAST_API_KEY) {
        $env:VAST_API_KEY = $VAST_API_KEY
    }

    # Get SSH URL from vastai
    $sshUrl = wsl -e bash -c "vastai ssh-url $InstanceId" 2>$null

    # Parse: ssh://root@ssh1.vast.ai:35702 or similar formats
    $sshHost = $null
    $sshPort = $null
    $sshUser = "root"

    if ($sshUrl -match 'ssh://([^@]+)@([^:]+):(\d+)') {
        $sshUser = $matches[1]
        $sshHost = $matches[2]
        $sshPort = $matches[3]
    } elseif ($sshUrl -match '-p\s+(\d+)\s+([^@]+)@(\S+)') {
        $sshPort = $matches[1]
        $sshUser = $matches[2]
        $sshHost = $matches[3]
    } elseif ($sshUrl -match '([^@]+)@([^:]+):(\d+)') {
        $sshUser = $matches[1]
        $sshHost = $matches[2]
        $sshPort = $matches[3]
    }

    if (-not $sshHost -or -not $sshPort) {
        Write-Host "ERROR: Could not parse SSH URL: $sshUrl" -ForegroundColor Red
        return $null
    }

    if (-not $Quiet) {
        Write-Host "Connecting to $sshUser@$sshHost port $sshPort..." -ForegroundColor Gray
    }

    # Escape single quotes in commands for bash
    $escapedCommands = $Commands -replace "'", "'\\''"

    # Simple direct SSH command - no heredoc complexity
    $sshCmd = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=$TimeoutSeconds -p $sshPort $sshUser@$sshHost '$escapedCommands'"

    $output = wsl -e bash -c $sshCmd 2>&1
    return $output
}

function Show-Help {
    Write-Host @"
Heretic GPU Cloud Automation
============================
Supports RunPod and Vast.ai (50% cheaper!)

SETUP (RunPod):
  1. Install WSL: wsl --install (admin PowerShell, then reboot)
  2. Download runpodctl: .\runpod.ps1 install-runpodctl
  3. Set API key: `$env:RUNPOD_API_KEY = 'your-key'` (from runpod.io settings)
  4. Add SSH key to RunPod (Settings -> SSH Keys)
  5. Run: .\runpod.ps1 create-pod
  6. Run: .\runpod.ps1 setup
  7. Run: .\runpod.ps1 test

SETUP (Vast.ai - 50% cheaper!):
  1. Install WSL: wsl --install (admin PowerShell, then reboot)
  2. Download vast CLI: .\runpod.ps1 install-vastcli
  3. Set API key: `$env:VAST_API_KEY = 'your-key'` (from vast.ai account)
  4. Add SSH key to Vast.ai (Account -> SSH Keys)
  5. Run: .\runpod.ps1 vast-create-pod
  6. Run: .\runpod.ps1 vast-setup
  7. Run: .\runpod.ps1 vast-run <model>

RUNPOD MANAGEMENT:
  create-pod [gpu] - Create pod (default: RTX 4090)
  list-pods        - List your pods
  get-ssh [podId]  - Get SSH details for a pod
  stop-pod [podId] - Stop pod (saves volume, stops billing)
  start-pod [podId]- Start a stopped pod
  terminate-pod [podId] - Delete pod permanently
  gpus             - List available GPU types

VAST.AI MANAGEMENT (50% cheaper!):
  vast-create-pod [tier] [num_gpus] - Create Vast.ai instance
                   Tiers: RTX_4090, A6000, A100_40GB, A100_80GB, A100_SXM, H100
                   Examples:
                     vast-create-pod              # RTX 4090, 1 GPU
                     vast-create-pod A100_80GB    # A100 80GB, 1 GPU
                     vast-create-pod A100_80GB 2  # A100 80GB, 2 GPUs (for 70B models)
  vast-list        - List your instances
  vast-gpus [tier] - Search available GPU offers (default: RTX_4090)
  vast-tiers       - List all GPU tiers and pricing
  vast-stop [id]   - Stop instance
  vast-start [id]  - Start instance
  vast-terminate [id] - Destroy instance
  vast-connect [id]- SSH to instance
  vast-setup [id]  - Install bruno on instance
  vast-run <model> - Run bruno on Vast.ai
  vast-exec <cmd>  - Execute command on Vast.ai
  vast-status      - GPU status on Vast.ai
  vast-progress    - Check abliteration progress
  vast-watch       - Live dashboard monitoring (updates every 30s)
  vast-models      - List abliterated models on Vast.ai
  vast-download <remote> [local] - Download file/directory from Vast.ai
  vast-download-model [model]    - Download abliterated model to local machine
  vast-hf-login    - Configure HuggingFace token
  vast-hf-test     - Test HuggingFace authentication

TOOLS:
  install-runpodctl - Download runpodctl CLI
  install-vastcli   - Download vast CLI
  check-tools       - Verify all tools are installed

RUNPOD COMMANDS:
  connect       - SSH to RunPod (interactive)
  setup         - Install bruno on RunPod
  test          - Test with Qwen3-4B (~15 min)
  run <model>   - Run bruno on model
  exec <cmd>    - Execute any command on RunPod
  status        - GPU status (nvidia-smi)

FILE OPERATIONS:
  upload <file> - Upload file to RunPod
  download <file> - Download from RunPod
  sync          - Upload config.toml

VLLM OPERATIONS:
  vllm-install  - Install vLLM on RunPod
  vllm-start <model-path> - Start vLLM server
  vllm-stop     - Stop vLLM server
  vllm-status   - Check vLLM server status
  tunnel        - Create SSH tunnel for vLLM (port 8000)

MONITORING:
  monitor       - Live GPU monitoring
  logs          - View vLLM logs

HUGGINGFACE:
  hf-login      - Configure HuggingFace token
  hf-test       - Test HuggingFace authentication

PRICE COMPARISON (RTX 4090):
  RunPod:   ~`$0.34/hr
  Vast.ai:  ~`$0.16-0.25/hr (50% cheaper!)

QUICK START (RunPod):
  .\runpod.ps1 create-pod                    # 1. Create pod
  .\runpod.ps1 setup                         # 2. Install bruno
  .\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507  # 3. Abliterate
  .\runpod.ps1 stop-pod                      # 4. Stop when done

QUICK START (Vast.ai - cheaper!):
  .\runpod.ps1 vast-create-pod               # 1. Create instance (RTX 4090)
  .\runpod.ps1 vast-setup                    # 2. Install bruno
  .\runpod.ps1 vast-run Qwen/Qwen3-4B-Instruct-2507  # 3. Abliterate
  .\runpod.ps1 vast-download-model           # 4. Download model to local
  .\runpod.ps1 vast-stop                     # 5. Stop when done

LARGE MODEL WORKFLOW (70B+ models):
  .\runpod.ps1 vast-create-pod A100_80GB 2   # 1. Create with 2x A100 80GB (~`$3/hr)
  .\runpod.ps1 vast-setup                    # 2. Install bruno
  .\runpod.ps1 vast-run Qwen/Qwen2.5-72B-Instruct  # 3. Abliterate 72B model
  .\runpod.ps1 vast-download-model           # 4. Download model to local (~1hr for 70B)
  .\runpod.ps1 vast-stop                     # 5. Stop when done

GPU TIERS & PRICING:
  RTX_4090    (24GB)  ~`$0.20-0.40/hr  - 7B-8B models
  A6000       (48GB)  ~`$0.40-0.70/hr  - 14B-30B models
  A100_40GB   (40GB)  ~`$0.50-0.90/hr  - 14B-32B models
  A100_80GB   (80GB)  ~`$1.00-1.80/hr  - 32B-70B models (single), 70B+ (multi)
  H100        (80GB)  ~`$1.50-3.50/hr  - 70B+ models, fastest

EXAMPLES:
  .\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct
  .\runpod.ps1 vast-run mistralai/Mistral-7B-Instruct-v0.3
  .\runpod.ps1 exec 'nvidia-smi'
  .\runpod.ps1 connect                       # Interactive SSH

"@
}

function Check-Config {
    if (-not $RUNPOD_HOST -or -not $RUNPOD_PORT) {
        Write-Host "ERROR: RUNPOD_HOST and RUNPOD_PORT not configured!" -ForegroundColor Red
        Write-Host "Run '.\runpod.ps1 create-pod' to create a pod, or set them manually." -ForegroundColor Yellow
        exit 1
    }
}

function Check-ApiKey {
    if (-not $RUNPOD_API_KEY) {
        Write-Host "ERROR: RUNPOD_API_KEY not set!" -ForegroundColor Red
        Write-Host "Set environment variable: `$env:RUNPOD_API_KEY = 'your-key'" -ForegroundColor Yellow
        Write-Host "Or permanently: [Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'your-key', 'User')" -ForegroundColor Yellow
        Write-Host "Get your API key from: https://runpod.io/console/user/settings" -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-RunPodGraphQL {
    param(
        [string]$Query,
        [hashtable]$Variables = $null
    )

    if (-not $RUNPOD_API_KEY) {
        Write-Host "ERROR: No API key found. Set RUNPOD_API_KEY environment variable." -ForegroundColor Red
        Write-Host "  `$env:RUNPOD_API_KEY = 'your-key'" -ForegroundColor Yellow
        return $null
    }
    $apiKey = $RUNPOD_API_KEY

    $headers = @{
        "Content-Type" = "application/json"
    }

    $uri = "https://api.runpod.io/graphql?api_key=$apiKey"

    $body = @{ query = $Query }
    if ($Variables) {
        $body.variables = $Variables
    }

    try {
        $jsonBody = $body | ConvertTo-Json -Depth 10
        $response = Invoke-RestMethod -Uri $uri -Method POST -Headers $headers -Body $jsonBody

        if ($response.errors) {
            Write-Host "GraphQL Error: $($response.errors[0].message)" -ForegroundColor Red
            return $null
        }

        return $response.data
    } catch {
        # Compatible error handling for both Windows PowerShell and PowerShell Core
        $errorMsg = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
        Write-Host "API Error: $errorMsg" -ForegroundColor Red
        return $null
    }
}

function Update-ScriptConfig {
    param(
        [string]$HostAddress,  # Renamed from $Host (reserved variable in PowerShell)
        [string]$Port,
        [string]$SSHProxy
    )

    $scriptPath = $PSCommandPath
    $content = Get-Content $scriptPath -Raw

    # Update RUNPOD_HOST
    if ($HostAddress) {
        $content = $content -replace '\$RUNPOD_HOST = "[^"]*"', "`$RUNPOD_HOST = `"$HostAddress`""
        Set-Variable -Name RUNPOD_HOST -Value $HostAddress -Scope Script
    }

    # Update RUNPOD_PORT
    if ($Port) {
        $content = $content -replace '\$RUNPOD_PORT = "[^"]*"', "`$RUNPOD_PORT = `"$Port`""
        Set-Variable -Name RUNPOD_PORT -Value $Port -Scope Script
    }

    # Update SSH proxy
    if ($SSHProxy) {
        $content = $content -replace '\$RUNPOD_SSH_PROXY = "[^"]*"', "`$RUNPOD_SSH_PROXY = `"$SSHProxy`""
        Set-Variable -Name RUNPOD_SSH_PROXY -Value $SSHProxy -Scope Script
    }

    Set-Content $scriptPath -Value $content -NoNewline
}

function Get-SSHCommand {
    $cmd = "ssh -o StrictHostKeyChecking=no"
    if ($SSH_KEY) {
        if (-not (Test-Path $SSH_KEY)) {
            Write-Host "WARNING: SSH key not found at $SSH_KEY" -ForegroundColor Yellow
            Write-Host "Using password authentication instead" -ForegroundColor Yellow
        } else {
            $cmd += " -i `"$SSH_KEY`""
        }
    }

    if ($USE_SSH_PROXY -and $RUNPOD_SSH_PROXY) {
        # Use RunPod's SSH proxy (more reliable, no port issues)
        $cmd += " $RUNPOD_SSH_PROXY"
    } else {
        # Use direct TCP connection
        $cmd += " ${RUNPOD_USER}@${RUNPOD_HOST} -p $RUNPOD_PORT"
    }
    return $cmd
}

function Get-SCPPrefix {
    $prefix = "scp"
    if ($SSH_KEY -and (Test-Path $SSH_KEY)) {
        $prefix += " -i `"$SSH_KEY`""
    }
    $prefix += " -P $RUNPOD_PORT"
    return $prefix
}

function Invoke-SSHCommand {
    param([string]$Command)
    $ssh = Get-SSHCommand
    $fullCmd = "$ssh '$Command'"
    Invoke-Expression $fullCmd
}

function Ensure-WSLSSHKey {
    # Copy SSH key to WSL with proper permissions (required because Windows NTFS permissions show as 0777)
    $wslKeyPath = "~/.ssh/id_ed25519"
    $windowsKeyPath = "/mnt/c/Users/$env:USERNAME/.ssh/id_ed25519"

    $setupCmd = "mkdir -p ~/.ssh && cp $windowsKeyPath $wslKeyPath 2>/dev/null && chmod 600 $wslKeyPath"
    $result = wsl -e bash -c $setupCmd 2>&1
    return $true
}

function Invoke-WSLSSHCommand {
    param(
        [string]$Commands,
        [switch]$Quiet
    )

    # Ensure WSL has the SSH key with proper permissions
    Ensure-WSLSSHKey | Out-Null

    # Get the SSH proxy address
    $sshTarget = $RUNPOD_SSH_PROXY

    if (-not $Quiet) {
        Write-Host "Executing on RunPod via WSL..." -ForegroundColor Gray
    }

    # Build and execute the heredoc command directly (proven to work)
    # Using heredoc allows us to:
    # 1. Allocate a proper PTY via -tt
    # 2. Send multiple commands
    # 3. Exit cleanly
    $heredocCmd = @"
wsl -e bash -c 'ssh -tt -o StrictHostKeyChecking=no -o ConnectTimeout=30 $sshTarget <<"SSHEOF"
$Commands
exit
SSHEOF'
"@

    # Execute directly (this is what was proven to work)
    $output = Invoke-Expression $heredocCmd
    return $output
}

function Test-WSLAvailable {
    try {
        $result = wsl --status 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

switch ($Action) {
    "help" { Show-Help }

    "gpus" {
        Check-ApiKey
        Write-Host "Fetching available GPU types..." -ForegroundColor Green

        $query = @"
query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: { gpuCount: 1 }) {
      minimumBidPrice
      uninterruptablePrice
    }
  }
}
"@

        $response = Invoke-RunPodGraphQL -Query $query
        if ($response -and $response.gpuTypes) {
            Write-Host "`nAvailable GPUs:" -ForegroundColor Cyan
            Write-Host ("-" * 70)
            Write-Host ("{0,-35} {1,8} {2,15}" -f "GPU Type", "VRAM", "Price") -ForegroundColor White
            Write-Host ("-" * 70)
            foreach ($gpu in $response.gpuTypes | Sort-Object { if ($_.lowestPrice.minimumBidPrice) { [float]$_.lowestPrice.minimumBidPrice } else { 999 } }) {
                $price = if ($gpu.lowestPrice.minimumBidPrice) { "`$$($gpu.lowestPrice.minimumBidPrice)/hr" } else { "unavailable" }
                $vram = "$($gpu.memoryInGb) GB"
                Write-Host ("{0,-35} {1,8} {2,15}" -f $gpu.id, $vram, $price)
            }
        }
    }

    "list-pods" {
        Write-Host "Fetching your pods..." -ForegroundColor Green

        # Prefer runpodctl for cleaner output
        if (Test-RunpodctlAvailable) {
            Write-Host ""
            Invoke-Runpodctl "get pod"
        } else {
            # Fallback to GraphQL API
            Check-ApiKey
            $query = @"
query Pods {
  myself {
    pods {
      id
      name
      desiredStatus
      machine {
        gpuDisplayName
      }
      runtime {
        ports {
          ip
          isIpPublic
          privatePort
          publicPort
        }
      }
    }
  }
}
"@

            $response = Invoke-RunPodGraphQL -Query $query
            if ($response -and $response.myself) {
                $pods = $response.myself.pods
                if (-not $pods -or $pods.Count -eq 0) {
                    Write-Host "No pods found." -ForegroundColor Yellow
                } else {
                    Write-Host "`nYour Pods:" -ForegroundColor Cyan
                    Write-Host ("-" * 70)
                    foreach ($pod in $pods) {
                        $status = $pod.desiredStatus
                        $statusColor = switch ($status) {
                            "RUNNING" { "Green" }
                            "STOPPED" { "Yellow" }
                            "EXITED" { "Red" }
                            default { "Gray" }
                        }
                        Write-Host "  ID: " -NoNewline
                        Write-Host $pod.id -ForegroundColor White -NoNewline
                        Write-Host " | " -NoNewline
                        Write-Host $status -ForegroundColor $statusColor -NoNewline
                        $gpuName = if ($pod.machine) { $pod.machine.gpuDisplayName } else { "" }
                        Write-Host " | $($pod.name) | $gpuName" -ForegroundColor Gray
                    }
                }
            }
        }
    }

    "create-pod" {
        Check-ApiKey
        $gpuType = if ($Arg1) { $Arg1 } else { $DEFAULT_GPU }
        $podName = "bruno-$(Get-Date -Format 'MMdd-HHmm')"

        # Read SSH public key to include in pod creation
        $sshPubKeyPath = "$SSH_KEY.pub"
        $sshPubKey = ""
        if (Test-Path $sshPubKeyPath) {
            $sshPubKey = (Get-Content $sshPubKeyPath -Raw).Trim()
            Write-Host "Including SSH public key in pod creation..." -ForegroundColor Gray
        } else {
            Write-Host "WARNING: SSH public key not found at $sshPubKeyPath" -ForegroundColor Yellow
            Write-Host "You may need to add your SSH key manually via RunPod console." -ForegroundColor Yellow
        }

        Write-Host "Creating pod with $gpuType..." -ForegroundColor Green

        # Build env array with SSH key if available
        $envArray = if ($sshPubKey) {
            "env: [{ key: `"SSH_PUBLIC_KEY`", value: `"$sshPubKey`" }]"
        } else {
            ""
        }

        $query = @"
mutation CreatePod {
  podFindAndDeployOnDemand(input: {
    cloudType: ALL
    gpuCount: 1
    volumeInGb: $DEFAULT_VOLUME_GB
    containerDiskInGb: $DEFAULT_CONTAINER_DISK_GB
    gpuTypeId: "$gpuType"
    name: "$podName"
    imageName: "$DEFAULT_IMAGE"
    ports: "22/tcp,8888/http"
    volumeMountPath: "/workspace"
    $envArray
  }) {
    id
    name
    desiredStatus
    machine {
      podHostId
    }
  }
}
"@

        $response = Invoke-RunPodGraphQL -Query $query

        if ($response -and $response.podFindAndDeployOnDemand -and $response.podFindAndDeployOnDemand.id) {
            $podId = $response.podFindAndDeployOnDemand.id
            Write-Host "Pod created! ID: $podId" -ForegroundColor Green
            Write-Host "Waiting for pod to start..." -ForegroundColor Yellow

            # Wait for pod to be ready and get SSH details
            $attempts = 0
            $maxAttempts = 30
            $sshDetails = $null

            while ($attempts -lt $maxAttempts) {
                Start-Sleep -Seconds 5
                $attempts++
                Write-Host "  Checking status (attempt $attempts/$maxAttempts)..." -ForegroundColor Gray

                # Use here-string with variable interpolation (same pattern as other queries)
                $statusQuery = @"
query Pod {
  pod(input: { podId: "$podId" }) {
    id
    desiredStatus
    runtime {
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
      }
    }
  }
}
"@
                $podStatus = Invoke-RunPodGraphQL -Query $statusQuery

                if ($podStatus -and $podStatus.pod -and $podStatus.pod.desiredStatus -eq "RUNNING" -and $podStatus.pod.runtime) {
                    # Try to extract SSH details
                    if ($podStatus.pod.runtime.ports) {
                        foreach ($port in $podStatus.pod.runtime.ports) {
                            if ($port.privatePort -eq 22 -and $port.isIpPublic) {
                                $sshDetails = @{
                                    Host = $port.ip
                                    Port = $port.publicPort
                                }
                                break
                            }
                        }
                    }

                    if ($sshDetails) {
                        break
                    }
                }
            }

            if ($sshDetails -and $sshDetails.Host -and $sshDetails.Port) {
                Write-Host "`nPod is ready!" -ForegroundColor Green
                Write-Host "  SSH Host: $($sshDetails.Host)" -ForegroundColor Cyan
                Write-Host "  SSH Port: $($sshDetails.Port)" -ForegroundColor Cyan

                # Build SSH proxy address using the constant user ID
                # IMPORTANT: The userId is your RunPod account ID (found in console SSH strings)
                # It is NOT the same as myself.id from the API (that returns a different format)
                $sshProxyAddr = "$podId-$RUNPOD_USER_ID@ssh.runpod.io"

                # Auto-update script config including SSH proxy
                Update-ScriptConfig -HostAddress $sshDetails.Host -Port $sshDetails.Port -SSHProxy $sshProxyAddr

                Write-Host "  SSH Proxy: $sshProxyAddr" -ForegroundColor Cyan
                Write-Host "`n" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host "  Pod Ready! Script auto-configured." -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Next steps:" -ForegroundColor Yellow
                Write-Host "  .\runpod.ps1 setup    # Install bruno" -ForegroundColor White
                Write-Host "  .\runpod.ps1 test     # Run test" -ForegroundColor White
                Write-Host "  .\runpod.ps1 stop-pod # When done" -ForegroundColor White
            } else {
                Write-Host "`nPod created but SSH details not yet available." -ForegroundColor Yellow
                Write-Host "Run '.\runpod.ps1 get-ssh $podId' in a minute." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Failed to create pod. Check your API key and GPU availability." -ForegroundColor Red
        }
    }

    "get-ssh" {
        $podId = $Arg1

        # Try runpodctl first (faster and cleaner)
        if (Test-RunpodctlAvailable) {
            Write-Host "Fetching SSH details..." -ForegroundColor Green
            $sshInfo = Get-RunpodctlSSHInfo -PodId $podId

            if ($sshInfo) {
                Write-Host ""
                Write-Host "SSH Connection Details:" -ForegroundColor Cyan
                Write-Host "  Host: $($sshInfo.Host)" -ForegroundColor White
                Write-Host "  Port: $($sshInfo.Port)" -ForegroundColor White
                Write-Host "  User: root" -ForegroundColor White
                Write-Host "  Pod:  $($sshInfo.PodId)" -ForegroundColor Gray
                Write-Host ""
                Write-Host "Direct TCP:  ssh root@$($sshInfo.Host) -p $($sshInfo.Port)" -ForegroundColor Yellow
                Write-Host "SSH Proxy:   ssh $($sshInfo.PodId)-<userid>@ssh.runpod.io" -ForegroundColor Yellow

                # Offer to update config
                $update = Read-Host "`nUpdate script config with these details? (y/n)"
                if ($update -eq "y") {
                    Update-ScriptConfig -HostAddress $sshInfo.Host -Port $sshInfo.Port
                    Write-Host "Config updated!" -ForegroundColor Green
                }
                return
            }
        }

        # Fallback to GraphQL API
        Check-ApiKey

        if (-not $podId) {
            # Get first running pod
            $listQuery = @"
query Pods {
  myself {
    pods {
      id
      desiredStatus
    }
  }
}
"@
            $response = Invoke-RunPodGraphQL -Query $listQuery
            if ($response -and $response.myself -and $response.myself.pods) {
                $runningPod = $response.myself.pods | Where-Object { $_.desiredStatus -eq "RUNNING" } | Select-Object -First 1
                if ($runningPod) {
                    $podId = $runningPod.id
                    Write-Host "Using pod: $podId" -ForegroundColor Gray
                }
            }
            if (-not $podId) {
                Write-Host "ERROR: No running pods found. Specify a pod ID." -ForegroundColor Red
                exit 1
            }
        }

        Write-Host "Fetching SSH details for pod $podId..." -ForegroundColor Green

        $query = @"
query Pod {
  pod(input: { podId: "$podId" }) {
    id
    desiredStatus
    runtime {
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
      }
    }
  }
}
"@

        $response = Invoke-RunPodGraphQL -Query $query

        if ($response -and $response.pod -and $response.pod.runtime -and $response.pod.runtime.ports) {
            foreach ($port in $response.pod.runtime.ports) {
                if ($port.privatePort -eq 22 -and $port.isIpPublic) {
                    Write-Host "`nSSH Connection Details:" -ForegroundColor Cyan
                    Write-Host "  Host: $($port.ip)" -ForegroundColor White
                    Write-Host "  Port: $($port.publicPort)" -ForegroundColor White
                    Write-Host "  User: root" -ForegroundColor White
                    Write-Host "`nConnect with: ssh root@$($port.ip) -p $($port.publicPort)" -ForegroundColor Yellow

                    # Offer to update config
                    $update = Read-Host "`nUpdate script config with these details? (y/n)"
                    if ($update -eq "y") {
                        Update-ScriptConfig -HostAddress $port.ip -Port $port.publicPort
                        Write-Host "Config updated!" -ForegroundColor Green
                    }
                    break
                }
            }
        } else {
            Write-Host "Could not get SSH details. Pod may still be starting." -ForegroundColor Yellow
            if ($response -and $response.pod) {
                Write-Host "Pod status: $($response.pod.desiredStatus)" -ForegroundColor Gray
            }
        }
    }

    "stop-pod" {
        $podId = $Arg1

        # Try runpodctl first
        if (Test-RunpodctlAvailable) {
            if (-not $podId) {
                $podId = Get-FirstRunningPodId
            }
            if (-not $podId) {
                Write-Host "ERROR: No running pods found." -ForegroundColor Red
                exit 1
            }

            Write-Host "Stopping pod $podId..." -ForegroundColor Yellow
            Invoke-Runpodctl "stop pod $podId"

            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "  Pod Stopped - Billing Paused" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Volume data preserved." -ForegroundColor Cyan
            Write-Host "Restart with: .\runpod.ps1 start-pod" -ForegroundColor Yellow
        } else {
            # Fallback to GraphQL API
            Check-ApiKey

            if (-not $podId) {
                $listQuery = @"
query Pods {
  myself {
    pods {
      id
      desiredStatus
    }
  }
}
"@
                $response = Invoke-RunPodGraphQL -Query $listQuery
                if ($response -and $response.myself -and $response.myself.pods) {
                    $runningPod = $response.myself.pods | Where-Object { $_.desiredStatus -eq "RUNNING" } | Select-Object -First 1
                    if ($runningPod) {
                        $podId = $runningPod.id
                    }
                }
                if (-not $podId) {
                    Write-Host "ERROR: No running pods found." -ForegroundColor Red
                    exit 1
                }
            }

            Write-Host "Stopping pod $podId..." -ForegroundColor Yellow

            $query = @"
mutation StopPod {
  podStop(input: { podId: "$podId" }) {
    id
    desiredStatus
  }
}
"@

            $response = Invoke-RunPodGraphQL -Query $query

            if ($response -and $response.podStop) {
                Write-Host ""
                Write-Host "========================================" -ForegroundColor Green
                Write-Host "  Pod Stopped - Billing Paused" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Volume data preserved." -ForegroundColor Cyan
                Write-Host "Restart with: .\runpod.ps1 start-pod" -ForegroundColor Yellow
            }
        }
    }

    "start-pod" {
        $podId = $Arg1

        # Try runpodctl first
        if (Test-RunpodctlAvailable) {
            if (-not $podId) {
                # Get first stopped pod from runpodctl output
                $output = Invoke-Runpodctl "get pod"
                $lines = $output -split "`n" | Where-Object { $_ -match 'EXITED|STOPPED' }
                if ($lines) {
                    $firstLine = $lines[0]
                    if ($firstLine -match '^(\S+)') {
                        $podId = $matches[1]
                    }
                }
            }
            if (-not $podId) {
                Write-Host "ERROR: No stopped pods found." -ForegroundColor Red
                exit 1
            }

            Write-Host "Starting pod $podId..." -ForegroundColor Green
            Invoke-Runpodctl "start pod $podId"
            Write-Host "Pod starting!" -ForegroundColor Green
            Write-Host "Wait ~30 seconds, then run '.\runpod.ps1 get-ssh $podId'" -ForegroundColor Yellow
        } else {
            # Fallback to GraphQL API
            Check-ApiKey

            if (-not $podId) {
                $listQuery = @"
query Pods {
  myself {
    pods {
      id
      desiredStatus
    }
  }
}
"@
                $response = Invoke-RunPodGraphQL -Query $listQuery
                if ($response -and $response.myself -and $response.myself.pods) {
                    $stoppedPod = $response.myself.pods | Where-Object { $_.desiredStatus -eq "STOPPED" -or $_.desiredStatus -eq "EXITED" } | Select-Object -First 1
                    if ($stoppedPod) {
                        $podId = $stoppedPod.id
                    }
                }
                if (-not $podId) {
                    Write-Host "ERROR: No stopped pods found." -ForegroundColor Red
                    exit 1
                }
            }

            Write-Host "Starting pod $podId..." -ForegroundColor Green

            $query = @"
mutation ResumePod {
  podResume(input: { podId: "$podId", gpuCount: 1 }) {
    id
    desiredStatus
  }
}
"@

            $response = Invoke-RunPodGraphQL -Query $query

            if ($response -and $response.podResume) {
                Write-Host "Pod starting!" -ForegroundColor Green
                Write-Host "Wait ~30 seconds, then run '.\runpod.ps1 get-ssh $podId'" -ForegroundColor Yellow
            }
        }
    }

    "terminate-pod" {
        Check-ApiKey

        $podId = $Arg1
        if (-not $podId) {
            Write-Host "ERROR: Specify pod ID to terminate." -ForegroundColor Red
            Write-Host "Use '.\runpod.ps1 list-pods' to see your pods." -ForegroundColor Yellow
            exit 1
        }

        $confirm = Read-Host "Are you sure you want to PERMANENTLY DELETE pod $podId? (yes/no)"
        if ($confirm -ne "yes") {
            Write-Host "Cancelled." -ForegroundColor Yellow
            exit 0
        }

        Write-Host "Terminating pod $podId..." -ForegroundColor Red

        $query = @"
mutation TerminatePod {
  podTerminate(input: { podId: "$podId" })
}
"@

        $response = Invoke-RunPodGraphQL -Query $query

        Write-Host "Pod terminated." -ForegroundColor Green
    }

    "connect" {
        Check-Config
        Write-Host "Connecting to RunPod..." -ForegroundColor Green
        Invoke-Expression (Get-SSHCommand)
    }

    "setup" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Setting up Heretic on RunPod" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""

        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            Write-Host "[1/4] Connecting via WSL SSH..." -ForegroundColor Yellow

            $setupCommands = @"
echo '[2/4] Configuring HuggingFace cache...'
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/huggingface
grep -q 'HF_HOME' ~/.bashrc || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc

echo '[3/4] Installing bruno-ai...'
pip install --quiet bruno-ai

echo '[4/4] Preparing GPU...'
fuser -k /dev/nvidia* 2>/dev/null || true
echo ''
echo 'GPU Status:'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ''
echo '========================================'
echo '  SETUP COMPLETE!'
echo '========================================'
echo ''
echo 'Next: .\runpod.ps1 test'
echo '  or: .\\runpod.ps1 run [model-name]'
"@

            $output = Invoke-WSLSSHCommand -Commands $setupCommands -TimeoutSeconds 300
            Write-Host $output

            Write-Host ""
            Write-Host "Ready! Run: .\runpod.ps1 test" -ForegroundColor Green
        } else {
            Write-Host "ERROR: WSL not available!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Install WSL:" -ForegroundColor Yellow
            Write-Host "  1. Open PowerShell as Admin" -ForegroundColor White
            Write-Host "  2. Run: wsl --install" -ForegroundColor White
            Write-Host "  3. Reboot" -ForegroundColor White
            Write-Host ""
            Write-Host "Or connect manually:" -ForegroundColor Yellow
            Write-Host "  ssh $RUNPOD_SSH_PROXY -i ~/.ssh/id_ed25519" -ForegroundColor White
        }
    }

    "test" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Running Heretic Test (Qwen3-4B)" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Estimated time: 15-20 min on RTX 4090" -ForegroundColor Yellow
        Write-Host ""

        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $testCommands = @"
export HF_HOME=/workspace/.cache/huggingface
cd /workspace
echo 'Starting Heretic abliteration...'
echo ''
bruno Qwen/Qwen3-4B-Instruct-2507
"@

            # Long timeout for model download + abliteration
            $output = Invoke-WSLSSHCommand -Commands $testCommands -TimeoutSeconds 1800
            Write-Host $output
        } else {
            Write-Host "ERROR: WSL required. Run 'wsl --install' first." -ForegroundColor Red
        }
    }

    "run" {
        if (-not $Arg1) {
            Write-Host "ERROR: Specify model name" -ForegroundColor Red
            Write-Host ""
            Write-Host "Examples:" -ForegroundColor Yellow
            Write-Host "  .\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507" -ForegroundColor White
            Write-Host "  .\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct" -ForegroundColor White
            Write-Host "  .\runpod.ps1 run mistralai/Mistral-7B-Instruct-v0.3" -ForegroundColor White
            exit 1
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Running Heretic Abliteration" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Model: $Arg1" -ForegroundColor Yellow
        Write-Host ""

        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $runCommands = @"
export HF_HOME=/workspace/.cache/huggingface
cd /workspace
echo 'Starting Heretic abliteration...'
echo 'Model: $Arg1'
echo ''
bruno $Arg1
"@

            # Long timeout for model download + abliteration
            $output = Invoke-WSLSSHCommand -Commands $runCommands -TimeoutSeconds 3600
            Write-Host $output
        } else {
            Write-Host "ERROR: WSL required. Run 'wsl --install' first." -ForegroundColor Red
        }
    }

    "upload" {
        Check-Config
        if (-not $Arg1) {
            Write-Host "ERROR: Specify file to upload" -ForegroundColor Red
            exit 1
        }

        $localPath = "$LOCAL_DIR\$Arg1"
        if (-not (Test-Path $localPath)) {
            Write-Host "ERROR: File not found: $localPath" -ForegroundColor Red
            exit 1
        }

        Write-Host "Uploading $Arg1..." -ForegroundColor Green
        $scp = Get-SCPPrefix
        Invoke-Expression "$scp `"$localPath`" ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"
    }

    "download" {
        Check-Config
        if (-not $Arg1) {
            Write-Host "ERROR: Specify file/directory to download" -ForegroundColor Red
            exit 1
        }

        Write-Host "Downloading $Arg1..." -ForegroundColor Green
        $scp = Get-SCPPrefix
        Invoke-Expression "$scp -r ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/$Arg1 `"$LOCAL_DIR\`""
    }

    "sync" {
        Check-Config
        Write-Host "Syncing config.toml..." -ForegroundColor Green
        $scp = Get-SCPPrefix
        Invoke-Expression "$scp `"$LOCAL_DIR\config.toml`" ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"
    }

    "status" {
        Write-Host "Checking GPU status..." -ForegroundColor Cyan
        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $output = Invoke-WSLSSHCommand -Commands "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader" -TimeoutSeconds 30
            Write-Host $output
        } else {
            Check-Config
            Invoke-SSHCommand "nvidia-smi"
        }
    }

    "exec" {
        if (-not $Arg1) {
            Write-Host "ERROR: Specify command to run" -ForegroundColor Red
            Write-Host "Example: .\runpod.ps1 exec 'nvidia-smi'" -ForegroundColor Yellow
            exit 1
        }

        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            Write-Host "Executing: $Arg1" -ForegroundColor Cyan
            $output = Invoke-WSLSSHCommand -Commands $Arg1 -TimeoutSeconds 60
            Write-Host $output
        } else {
            Check-Config
            Invoke-SSHCommand $Arg1
        }
    }

    "monitor" {
        Check-Config
        Write-Host "GPU monitoring (Ctrl+C to exit)..." -ForegroundColor Green
        Invoke-SSHCommand "watch -n 1 nvidia-smi"
    }

    "vllm-install" {
        Write-Host "Installing vLLM..." -ForegroundColor Green
        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $output = Invoke-WSLSSHCommand -Commands "pip install vllm && echo 'vLLM installed!'"
            Write-Host $output
        } else {
            Write-Host "ERROR: WSL required. Run 'wsl --install' first." -ForegroundColor Red
        }
        Write-Host "Next: .\runpod.ps1 vllm-start <model-path>" -ForegroundColor Cyan
    }

    "vllm-start" {
        Check-Config
        if (-not $Arg1) {
            Write-Host "ERROR: Specify model path" -ForegroundColor Red
            Write-Host "Example: .\runpod.ps1 vllm-start /workspace/bruno/models/llama-8b" -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Starting vLLM server for $Arg1..." -ForegroundColor Green
        Write-Host "Server will run on port $VLLM_PORT" -ForegroundColor Cyan
        Write-Host "Use 'tunnel' command to access from Windows" -ForegroundColor Cyan

        $vllmCmd = @"
cd $REMOTE_DIR && \
nohup python -m vllm.entrypoints.openai.api_server \
  --model $Arg1 \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --gpu-memory-utilization 0.95 \
  --dtype auto \
  > vllm.log 2>&1 &
echo \`$! > vllm.pid
echo 'vLLM server starting...'
sleep 5
tail -20 vllm.log
"@

        Invoke-SSHCommand $vllmCmd
        Write-Host "`nvLLM started! Check status with: .\runpod.ps1 vllm-status" -ForegroundColor Green
        Write-Host "Create tunnel with: .\runpod.ps1 tunnel" -ForegroundColor Cyan
    }

    "vllm-stop" {
        Check-Config
        Write-Host "Stopping vLLM server..." -ForegroundColor Yellow
        Invoke-SSHCommand "cd $REMOTE_DIR && if [ -f vllm.pid ]; then kill \`$(cat vllm.pid) 2>/dev/null; rm vllm.pid; echo 'vLLM stopped'; else echo 'No vLLM PID file found'; fi"
    }

    "vllm-status" {
        Check-Config
        Write-Host "Checking vLLM status..." -ForegroundColor Green
        Invoke-SSHCommand "cd $REMOTE_DIR && if [ -f vllm.pid ]; then ps -p \`$(cat vllm.pid) > /dev/null && echo 'vLLM running (PID: '\`$(cat vllm.pid)')' || echo 'vLLM not running (stale PID file)'; else echo 'vLLM not running'; fi; curl -s http://localhost:$VLLM_PORT/health 2>/dev/null && echo 'vLLM API responding' || echo 'vLLM API not responding'"
    }

    "logs" {
        Check-Config
        Write-Host "vLLM logs (Ctrl+C to exit):" -ForegroundColor Green
        Invoke-SSHCommand "cd $REMOTE_DIR && tail -f vllm.log"
    }

    "tunnel" {
        Check-Config
        Write-Host "Creating SSH tunnel for vLLM..." -ForegroundColor Green
        Write-Host "Local: http://localhost:$VLLM_PORT" -ForegroundColor Cyan
        Write-Host "Keep this window open. Press Ctrl+C to close tunnel." -ForegroundColor Yellow

        $tunnelCmd = "ssh"
        if ($SSH_KEY -and (Test-Path $SSH_KEY)) {
            $tunnelCmd += " -i `"$SSH_KEY`""
        }
        $tunnelCmd += " -L ${VLLM_PORT}:localhost:${VLLM_PORT} ${RUNPOD_USER}@${RUNPOD_HOST} -p $RUNPOD_PORT -N"

        Invoke-Expression $tunnelCmd
    }

    "hf-login" {
        if ($HF_TOKEN) {
            Write-Host "Using token from script configuration..." -ForegroundColor Green
            if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
                $output = Invoke-WSLSSHCommand -Commands "huggingface-cli login --token $HF_TOKEN"
                Write-Host $output
            }
        } else {
            Write-Host "Enter HuggingFace token:" -ForegroundColor Yellow
            $token = Read-Host
            if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
                $output = Invoke-WSLSSHCommand -Commands "huggingface-cli login --token $token"
                Write-Host $output
            }
        }

        Write-Host "HuggingFace login complete!" -ForegroundColor Green
    }

    "hf-test" {
        Write-Host "Testing HuggingFace authentication..." -ForegroundColor Green
        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $output = Invoke-WSLSSHCommand -Commands "huggingface-cli whoami"
            Write-Host $output
        } else {
            Write-Host "ERROR: WSL required." -ForegroundColor Red
        }
    }

    "install-runpodctl" {
        Write-Host "Installing runpodctl..." -ForegroundColor Green

        $downloadUrl = "https://github.com/runpod/runpodctl/releases/download/v1.14.15/runpodctl-windows-amd64.zip"
        $zipPath = "$PSScriptRoot\runpodctl.zip"
        $exePath = "$PSScriptRoot\runpodctl.exe"

        try {
            Write-Host "Downloading from GitHub..." -ForegroundColor Gray
            Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing

            Write-Host "Extracting..." -ForegroundColor Gray
            Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force
            Remove-Item $zipPath -Force

            if (Test-Path $exePath) {
                Write-Host ""
                Write-Host "========================================" -ForegroundColor Green
                Write-Host "  runpodctl installed successfully!" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Location: $exePath" -ForegroundColor Cyan

                # Configure API key
                Write-Host "Configuring API key..." -ForegroundColor Gray
                $configDir = "$env:USERPROFILE\.runpod"
                if (-not (Test-Path $configDir)) {
                    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
                }
                if ($RUNPOD_API_KEY) {
                    "apiKey: $RUNPOD_API_KEY" | Out-File -FilePath "$configDir\.runpod.yaml" -Encoding utf8
                } else {
                    Write-Host "WARNING: No API key set. Configure runpodctl manually." -ForegroundColor Yellow
                }
                Write-Host "API key configured." -ForegroundColor Green

                # Test it
                Write-Host ""
                Write-Host "Testing..." -ForegroundColor Gray
                & $exePath get pod
            } else {
                Write-Host "ERROR: Installation failed - executable not found" -ForegroundColor Red
            }
        } catch {
            Write-Host "ERROR: Download failed - $($_.Exception.Message)" -ForegroundColor Red
            Write-Host ""
            Write-Host "Manual install:" -ForegroundColor Yellow
            Write-Host "  1. Download: $downloadUrl" -ForegroundColor White
            Write-Host "  2. Extract runpodctl.exe to: $PSScriptRoot" -ForegroundColor White
        }
    }

    "check-tools" {
        Write-Host ""
        Write-Host "Checking tools..." -ForegroundColor Cyan
        Write-Host ("-" * 50)

        # Check WSL
        Write-Host "WSL: " -NoNewline
        if (Test-WSLAvailable) {
            Write-Host "OK" -ForegroundColor Green
        } else {
            Write-Host "NOT INSTALLED" -ForegroundColor Red
            Write-Host "  Install: wsl --install (admin PowerShell)" -ForegroundColor Yellow
        }

        # Check runpodctl
        Write-Host "runpodctl: " -NoNewline
        if (Test-RunpodctlAvailable) {
            Write-Host "OK ($RUNPODCTL_PATH)" -ForegroundColor Green
        } else {
            Write-Host "NOT INSTALLED" -ForegroundColor Red
            Write-Host "  Install: .\runpod.ps1 install-runpodctl" -ForegroundColor Yellow
        }

        # Check vast CLI
        Write-Host "vast CLI: " -NoNewline
        if (Test-VastCliAvailable) {
            Write-Host "OK ($VASTAI_CLI_PATH)" -ForegroundColor Green
        } else {
            Write-Host "NOT INSTALLED" -ForegroundColor Yellow
            Write-Host "  Install: .\runpod.ps1 install-vastcli" -ForegroundColor Yellow
        }

        # Check SSH key
        Write-Host "SSH Key: " -NoNewline
        if (Test-Path $SSH_KEY) {
            Write-Host "OK ($SSH_KEY)" -ForegroundColor Green
        } else {
            Write-Host "NOT FOUND" -ForegroundColor Red
            Write-Host "  Create: ssh-keygen -t ed25519" -ForegroundColor Yellow
        }

        # Check RunPod API key
        Write-Host "RunPod API Key: " -NoNewline
        if ($RUNPOD_API_KEY -and $RUNPOD_API_KEY.Length -gt 10) {
            Write-Host "OK" -ForegroundColor Green
        } else {
            Write-Host "NOT CONFIGURED" -ForegroundColor Yellow
            Write-Host "  Set: `$env:RUNPOD_API_KEY = 'your-key'" -ForegroundColor Yellow
        }

        # Check Vast.ai API key
        Write-Host "Vast.ai API Key: " -NoNewline
        if ($VAST_API_KEY -and $VAST_API_KEY.Length -gt 10) {
            Write-Host "OK" -ForegroundColor Green
        } else {
            Write-Host "NOT CONFIGURED" -ForegroundColor Yellow
            Write-Host "  Set: `$env:VAST_API_KEY = 'your-key'" -ForegroundColor Yellow
        }

        Write-Host ("-" * 50)
        Write-Host ""
    }

    "progress" {
        Write-Host ""
        Write-Host "Checking abliteration progress..." -ForegroundColor Cyan
        Write-Host ""

        if ($USE_WSL_SSH -and (Test-WSLAvailable)) {
            $progressCommands = @"
echo '=== Process Status ==='
ps aux | grep bruno | head -3
echo ''
echo '=== GPU Status ==='
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ''
echo '=== Workspace ==='
ls -la /workspace/*.db 2>/dev/null; ls -la /workspace/models/ 2>/dev/null; echo 'Done'
"@

            $output = Invoke-WSLSSHCommand -Commands $progressCommands
            Write-Host $output
        } else {
            Write-Host "ERROR: WSL required for progress check." -ForegroundColor Red
        }
    }

    # ===== VAST.AI COMMANDS =====

    "install-vastcli" {
        Write-Host "Installing Vast.ai CLI..." -ForegroundColor Green

        # Vast CLI is a Python package - install via pip in WSL
        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required to install vast CLI" -ForegroundColor Red
            Write-Host "Install WSL: wsl --install (admin PowerShell)" -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Installing via pip in WSL..." -ForegroundColor Gray
        $result = wsl -e bash -c "pip install vastai --upgrade && which vastai"
        Write-Host $result

        # Create a wrapper script for Windows
        $wrapperContent = @"
@echo off
wsl -e vastai %*
"@
        $wrapperPath = "$PSScriptRoot\vast.bat"
        Set-Content -Path $wrapperPath -Value $wrapperContent

        # Also try to get standalone binary
        Write-Host ""
        Write-Host "Creating Windows wrapper at: $wrapperPath" -ForegroundColor Cyan

        # Configure API key if available
        if ($VAST_API_KEY) {
            Write-Host "Configuring API key..." -ForegroundColor Gray
            wsl -e bash -c "vastai set api-key $VAST_API_KEY"
        } else {
            Write-Host ""
            Write-Host "NOTE: Set your Vast.ai API key:" -ForegroundColor Yellow
            Write-Host "  `$env:VAST_API_KEY = 'your-key'" -ForegroundColor White
            Write-Host "  Or add VAST_API_KEY=your-key to .env file" -ForegroundColor White
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  Vast.ai CLI installed!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Test with: .\runpod.ps1 vast-gpus" -ForegroundColor Cyan
    }

    "vast-tiers" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Available GPU Tiers" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host ("{0,-15} {1,8} {2,12} {3,-35}" -f "TIER", "VRAM", "MAX PRICE", "RECOMMENDED FOR") -ForegroundColor White
        Write-Host ("-" * 75)

        foreach ($tierName in @("RTX_4090", "A6000", "A100_40GB", "A100_80GB", "A100_SXM", "H100")) {
            $tier = $GPU_TIERS[$tierName]
            $vram = "$($tier.MinVram)GB"
            $price = "`$$($tier.MaxPrice)/hr"
            Write-Host ("{0,-15} {1,8} {2,12} {3,-35}" -f $tierName, $vram, $price, $tier.Description)
        }

        Write-Host ""
        Write-Host "MODEL SIZE RECOMMENDATIONS:" -ForegroundColor Yellow
        Write-Host "  7B-8B models   -> RTX_4090 (1 GPU)" -ForegroundColor White
        Write-Host "  14B-30B models -> A6000 or A100_40GB (1 GPU)" -ForegroundColor White
        Write-Host "  32B models     -> A100_80GB (1 GPU)" -ForegroundColor White
        Write-Host "  70B-72B models -> A100_80GB (2 GPUs) or H100 (2 GPUs)" -ForegroundColor White
        Write-Host ""
        Write-Host "MULTI-GPU EXAMPLES:" -ForegroundColor Yellow
        Write-Host "  .\runpod.ps1 vast-create-pod A100_80GB 2   # ~`$3-4/hr, good for 70B" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-create-pod H100 2        # ~`$4-7/hr, fastest for 70B" -ForegroundColor White
        Write-Host ""
    }

    "vast-gpus" {
        # Get GPU tier from argument or use default
        $gpuTier = if ($Arg1 -and $GPU_TIERS.ContainsKey($Arg1)) { $Arg1 } else { $VAST_DEFAULT_GPU }
        $tierConfig = $GPU_TIERS[$gpuTier]

        Write-Host "Searching Vast.ai GPU offers..." -ForegroundColor Green
        Write-Host "Tier: $gpuTier ($($tierConfig.Description))" -ForegroundColor Cyan
        Write-Host "(Filtering: $($tierConfig.GpuName), >$VAST_MIN_DOWNLOAD Mbps, <`$$($tierConfig.MaxPrice)/hr)" -ForegroundColor Gray
        Write-Host ""

        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required" -ForegroundColor Red
            exit 1
        }

        # Build search query based on tier
        # For A100 tiers, we need to filter by VRAM to distinguish 40GB vs 80GB
        # Vast.ai reports gpu_ram in MB, so multiply GB by 1024
        $gpuFilter = $tierConfig.GpuName
        $vramFilterMB = $tierConfig.MinVram * 1024
        $vramFilter = if ($tierConfig.MinVram -ge 40) { "gpu_ram>=$vramFilterMB" } else { "" }

        $searchQuery = "gpu_name=$gpuFilter rentable=true num_gpus>=1 inet_down>=$VAST_MIN_DOWNLOAD dph<=$($tierConfig.MaxPrice)"
        if ($vramFilter) {
            $searchQuery += " $vramFilter"
        }

        $searchCmd = "vastai search offers '$searchQuery' --order 'dph' --limit 20"
        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        $output = wsl -e bash -c $searchCmd
        Write-Host $output

        Write-Host ""
        Write-Host "Available GPU tiers: RTX_4090, A6000, A100_40GB, A100_80GB, A100_SXM, H100" -ForegroundColor Yellow
        Write-Host "Search other tiers: .\runpod.ps1 vast-gpus A100_80GB" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To create an instance: .\runpod.ps1 vast-create-pod $gpuTier" -ForegroundColor Cyan
    }

    "vast-create-pod" {
        if (-not $VAST_API_KEY) {
            Write-Host "ERROR: VAST_API_KEY not set!" -ForegroundColor Red
            Write-Host "Set: `$env:VAST_API_KEY = 'your-key'" -ForegroundColor Yellow
            Write-Host "Get key from: https://cloud.vast.ai/account/" -ForegroundColor Yellow
            exit 1
        }

        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required" -ForegroundColor Red
            exit 1
        }

        # Parse GPU tier argument
        if ($Arg1 -and -not $GPU_TIERS.ContainsKey($Arg1)) {
            Write-Host "WARNING: Unknown tier '$Arg1', using $VAST_DEFAULT_GPU" -ForegroundColor Yellow
            Write-Host "Valid tiers: RTX_4090, A6000, A100_40GB, A100_80GB, A100_SXM, H100" -ForegroundColor Yellow
            Write-Host ""
        }
        $gpuTier = if ($Arg1 -and $GPU_TIERS.ContainsKey($Arg1)) { $Arg1 } else { $VAST_DEFAULT_GPU }
        $numGpus = if ($Arg2 -and $Arg2 -match '^\d+$') { [int]$Arg2 } else { $VAST_DEFAULT_NUM_GPUS }

        # Get tier configuration
        $tierConfig = $GPU_TIERS[$gpuTier]
        $diskGB = $tierConfig.DiskGB
        $maxPrice = $tierConfig.MaxPrice * $numGpus  # Scale price by number of GPUs

        # For multi-GPU, increase disk space
        if ($numGpus -gt 1) {
            $diskGB = [math]::Max($diskGB, 200)
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Creating Vast.ai Instance" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "GPU Tier: $gpuTier ($($tierConfig.Description))" -ForegroundColor Yellow
        Write-Host "Num GPUs: $numGpus" -ForegroundColor Yellow
        Write-Host "Image: $VAST_DEFAULT_IMAGE" -ForegroundColor Yellow
        Write-Host "Disk: $diskGB GB" -ForegroundColor Yellow
        Write-Host "Max price: `$$maxPrice/hr ($($tierConfig.MaxPrice) x $numGpus GPUs)" -ForegroundColor Yellow
        Write-Host ""

        # Find best offer
        Write-Host "[1/3] Finding best GPU offer..." -ForegroundColor Yellow
        $env:VAST_API_KEY = $VAST_API_KEY

        # Build search query based on tier
        # For A100 tiers, filter by VRAM to distinguish 40GB vs 80GB
        # Vast.ai reports gpu_ram in MB, so multiply GB by 1024
        $gpuFilter = $tierConfig.GpuName
        $vramFilterMB = $tierConfig.MinVram * 1024
        $vramFilter = if ($tierConfig.MinVram -ge 40) { "gpu_ram>=$vramFilterMB" } else { "" }

        $searchQuery = "gpu_name=$gpuFilter rentable=true num_gpus>=$numGpus inet_down>=$VAST_MIN_DOWNLOAD dph<=$maxPrice"
        if ($vramFilter) {
            $searchQuery += " $vramFilter"
        }

        $searchCmd = "vastai search offers '$searchQuery' --order 'dph' --limit 1 --raw"
        $offerJson = wsl -e bash -c $searchCmd 2>$null

        if (-not $offerJson -or $offerJson -match "error" -or $offerJson -match "No offers") {
            Write-Host "ERROR: No suitable GPU offers found" -ForegroundColor Red
            Write-Host "Try: .\runpod.ps1 vast-gpus to see available offers" -ForegroundColor Yellow
            exit 1
        }

        try {
            $offers = $offerJson | ConvertFrom-Json
            if (-not $offers -or $offers.Count -eq 0) {
                Write-Host "ERROR: No offers returned" -ForegroundColor Red
                Write-Host ""
                Write-Host "Suggestions:" -ForegroundColor Yellow
                Write-Host "  - Try a different GPU tier: .\runpod.ps1 vast-gpus A100_80GB" -ForegroundColor White
                Write-Host "  - Check available offers: .\runpod.ps1 vast-gpus $gpuTier" -ForegroundColor White
                Write-Host "  - For 70B models, try: .\runpod.ps1 vast-create-pod A100_80GB 2" -ForegroundColor White
                exit 1
            }
            $bestOffer = $offers[0]
            $offerId = $bestOffer.id
            $price = [math]::Round($bestOffer.dph_total, 3)
            $gpuName = $bestOffer.gpu_name
            $offerNumGpus = if ($bestOffer.num_gpus) { $bestOffer.num_gpus } else { 1 }
            $offerVram = if ($bestOffer.gpu_ram) { [math]::Round($bestOffer.gpu_ram / 1024, 0) } else { "?" }

            Write-Host "  Found: $offerNumGpus x $gpuName ($($offerVram)GB each) at `$$price/hr (offer ID: $offerId)" -ForegroundColor Green
        } catch {
            # Try parsing as simple text
            if ($offerJson -match '^\s*(\d+)') {
                $offerId = $matches[1]
                Write-Host "  Found offer ID: $offerId" -ForegroundColor Green
            } else {
                Write-Host "ERROR: Could not parse offer response" -ForegroundColor Red
                Write-Host $offerJson
                exit 1
            }
        }

        # Create the instance
        Write-Host "[2/3] Creating instance..." -ForegroundColor Yellow
        $createCmd = "vastai create instance $offerId --image '$VAST_DEFAULT_IMAGE' --disk $diskGB --ssh --raw"
        $createResult = wsl -e bash -c $createCmd 2>&1

        Write-Host $createResult -ForegroundColor Gray

        # Parse instance ID from result
        $instanceId = $null
        if ($createResult -match '"new_contract":\s*(\d+)') {
            $instanceId = $matches[1]
        } elseif ($createResult -match 'instance\s+(\d+)') {
            $instanceId = $matches[1]
        } elseif ($createResult -match '^\s*(\d+)') {
            $instanceId = $matches[1]
        }

        if (-not $instanceId) {
            Write-Host "ERROR: Could not get instance ID from response" -ForegroundColor Red
            exit 1
        }

        Write-Host "  Instance ID: $instanceId" -ForegroundColor Green

        # Wait for instance to be ready
        Write-Host "[3/3] Waiting for instance to start..." -ForegroundColor Yellow
        $attempts = 0
        $maxAttempts = 30
        $sshInfo = $null

        while ($attempts -lt $maxAttempts) {
            Start-Sleep -Seconds 5
            $attempts++
            Write-Host "  Checking status (attempt $attempts/$maxAttempts)..." -ForegroundColor Gray

            # Check instance status
            $statusCmd = "vastai show instance $instanceId --raw"
            $statusJson = wsl -e bash -c $statusCmd 2>$null

            if ($statusJson -match '"actual_status":\s*"running"' -or $statusJson -match '"status":\s*"running"') {
                # Try to get SSH info
                $sshCmd = "vastai ssh-url $instanceId"
                $sshUrl = wsl -e bash -c $sshCmd 2>$null

                if ($sshUrl -match '-p\s+(\d+).*@([\d.]+)') {
                    $sshInfo = @{
                        Port = $matches[1]
                        Host = $matches[2]
                    }
                    break
                } elseif ($sshUrl -match '@([\d.]+):(\d+)') {
                    $sshInfo = @{
                        Host = $matches[1]
                        Port = $matches[2]
                    }
                    break
                }
            }
        }

        if ($sshInfo) {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "  Instance Ready!" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "  Instance ID: $instanceId" -ForegroundColor Cyan
            Write-Host "  SSH Host: $($sshInfo.Host)" -ForegroundColor Cyan
            Write-Host "  SSH Port: $($sshInfo.Port)" -ForegroundColor Cyan
            Write-Host ""

            # Update script config
            Update-VastScriptConfig -HostAddress $sshInfo.Host -Port $sshInfo.Port -InstanceId $instanceId

            Write-Host "Script auto-configured!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Next steps:" -ForegroundColor Yellow
            Write-Host "  .\runpod.ps1 vast-setup    # Install bruno" -ForegroundColor White
            Write-Host "  .\runpod.ps1 vast-run <model>  # Abliterate" -ForegroundColor White
            Write-Host "  .\runpod.ps1 vast-stop     # Stop when done" -ForegroundColor White
        } else {
            Write-Host ""
            Write-Host "Instance created but SSH not ready yet." -ForegroundColor Yellow
            Write-Host "Instance ID: $instanceId" -ForegroundColor Cyan
            Write-Host "Check status: .\runpod.ps1 vast-list" -ForegroundColor Yellow
        }
    }

    "vast-list" {
        Write-Host "Fetching your Vast.ai instances..." -ForegroundColor Green
        Write-Host ""

        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required" -ForegroundColor Red
            exit 1
        }

        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        $output = wsl -e bash -c "vastai show instances"
        Write-Host $output
    }

    "vast-stop" {
        $instanceId = if ($Arg1) { $Arg1 } else { $VAST_INSTANCE_ID }

        if (-not $instanceId) {
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID specified or found" -ForegroundColor Red
            Write-Host "Usage: .\runpod.ps1 vast-stop <instance-id>" -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Stopping Vast.ai instance $instanceId..." -ForegroundColor Yellow

        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        $output = wsl -e bash -c "vastai stop instance $instanceId"
        Write-Host $output

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  Instance Stopped - Billing Paused" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Restart with: .\runpod.ps1 vast-start $instanceId" -ForegroundColor Yellow
    }

    "vast-start" {
        $instanceId = if ($Arg1) { $Arg1 } else { $VAST_INSTANCE_ID }

        if (-not $instanceId) {
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID specified" -ForegroundColor Red
            Write-Host "Usage: .\runpod.ps1 vast-start <instance-id>" -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Starting Vast.ai instance $instanceId..." -ForegroundColor Green

        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        $output = wsl -e bash -c "vastai start instance $instanceId"
        Write-Host $output

        Write-Host ""
        Write-Host "Instance starting! Wait ~30 seconds, then run '.\runpod.ps1 vast-list'" -ForegroundColor Yellow
    }

    "vast-terminate" {
        $instanceId = if ($Arg1) { $Arg1 } else { $VAST_INSTANCE_ID }

        if (-not $instanceId) {
            Write-Host "ERROR: Specify instance ID to terminate" -ForegroundColor Red
            Write-Host "Usage: .\runpod.ps1 vast-terminate <instance-id>" -ForegroundColor Yellow
            Write-Host "Use '.\runpod.ps1 vast-list' to see your instances" -ForegroundColor Yellow
            exit 1
        }

        $confirm = Read-Host "Are you sure you want to DESTROY instance $instanceId? (yes/no)"
        if ($confirm -ne "yes") {
            Write-Host "Cancelled." -ForegroundColor Yellow
            exit 0
        }

        Write-Host "Destroying Vast.ai instance $instanceId..." -ForegroundColor Red

        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        $output = wsl -e bash -c "vastai destroy instance $instanceId"
        Write-Host $output

        Write-Host "Instance destroyed." -ForegroundColor Green
    }

    "vast-connect" {
        $instanceId = if ($Arg1) { $Arg1 } else { $VAST_INSTANCE_ID }

        if (-not $instanceId) {
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID specified" -ForegroundColor Red
            Write-Host "Usage: .\runpod.ps1 vast-connect <instance-id>" -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Connecting to Vast.ai instance $instanceId..." -ForegroundColor Green

        if ($VAST_API_KEY) {
            $env:VAST_API_KEY = $VAST_API_KEY
        }

        # Get SSH URL
        $sshUrl = wsl -e bash -c "vastai ssh-url $instanceId"

        if ($sshUrl -match 'ssh\s+(.+)') {
            $sshCmd = $matches[0]
            Write-Host "Running: $sshCmd" -ForegroundColor Gray
            wsl -e bash -c $sshCmd
        } else {
            Write-Host "ERROR: Could not get SSH URL" -ForegroundColor Red
            Write-Host "Response: $sshUrl" -ForegroundColor Gray
        }
    }

    "vast-setup" {
        $instanceId = if ($Arg1) { $Arg1 } else { $VAST_INSTANCE_ID }

        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Setting up Heretic on Vast.ai" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Instance ID: $instanceId" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[1/5] Connecting via SSH..." -ForegroundColor Yellow

        # Note: PyTorch Docker image doesn't have git installed, so we need to install it first
        $setupCommands = @"
echo '[2/5] Installing git (required for pip install from GitHub)...'
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 || { echo 'ERROR: Failed to install git'; exit 1; }
echo '  git installed'

echo '[3/5] Configuring workspace...'
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/models
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
grep -q 'HF_HOME' ~/.bashrc || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc

echo '[4/5] Installing bruno from GitHub...'
pip install --quiet git+https://github.com/quanticsoul4772/bruno.git
echo '  bruno installed'

echo '[5/5] Checking GPU...'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ''
echo '========================================'
echo '  SETUP COMPLETE!'
echo '========================================'
echo ''
echo 'Next: .\\runpod.ps1 vast-run [model]'
"@

        $output = Invoke-VastSSHCommand -Commands $setupCommands -InstanceId $instanceId -TimeoutSeconds 180
        Write-Host $output

        Write-Host ""
        Write-Host "Ready! Run: .\runpod.ps1 vast-run <model>" -ForegroundColor Green
    }

    "vast-run" {
        if (-not $Arg1) {
            Write-Host "ERROR: Specify model name" -ForegroundColor Red
            Write-Host ""
            Write-Host "Examples:" -ForegroundColor Yellow
            Write-Host "  .\runpod.ps1 vast-run Qwen/Qwen3-4B-Instruct-2507" -ForegroundColor White
            Write-Host "  .\runpod.ps1 vast-run meta-llama/Llama-3.1-8B-Instruct" -ForegroundColor White
            Write-Host "  .\runpod.ps1 vast-run Qwen/Qwen2.5-72B-Instruct  # Requires A100 80GB x2" -ForegroundColor White
            Write-Host ""
            Write-Host "GPU requirements by model size:" -ForegroundColor Yellow
            Write-Host "  7B-8B   -> RTX 4090 (24GB)" -ForegroundColor White
            Write-Host "  14B-32B -> A100 40-80GB" -ForegroundColor White
            Write-Host "  70B-72B -> A100 80GB x2 or H100 x2" -ForegroundColor White
            exit 1
        }

        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Running Heretic on Vast.ai" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Model: $Arg1" -ForegroundColor Yellow
        Write-Host "Instance: $instanceId" -ForegroundColor Yellow
        Write-Host ""

        # Determine output path based on model name
        $modelName = $Arg1 -replace '/', '-'
        $outputPath = "/workspace/models/$modelName-bruno"

        $runCommands = @"
export HF_HOME=/workspace/.cache/huggingface
cd /workspace
echo 'Starting Heretic abliteration...'
echo 'Model: $Arg1'
echo 'Output: $outputPath'
echo ''
bruno $Arg1 --auto-select --auto-select-path $outputPath
"@

        $output = Invoke-VastSSHCommand -Commands $runCommands -InstanceId $instanceId -TimeoutSeconds 60
        Write-Host $output
    }

    "vast-exec" {
        if (-not $Arg1) {
            Write-Host "ERROR: Specify command to run" -ForegroundColor Red
            Write-Host "Example: .\runpod.ps1 vast-exec 'nvidia-smi'" -ForegroundColor Yellow
            exit 1
        }

        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host "Executing on Vast.ai instance $instanceId..." -ForegroundColor Cyan

        $output = Invoke-VastSSHCommand -Commands $Arg1 -InstanceId $instanceId
        Write-Host $output
    }

    "vast-status" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host "Checking GPU status on Vast.ai instance $instanceId..." -ForegroundColor Cyan

        $output = Invoke-VastSSHCommand -Commands "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader" -InstanceId $instanceId -Quiet
        Write-Host $output
    }

    "vast-progress" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "Checking abliteration progress on Vast.ai..." -ForegroundColor Cyan
        Write-Host ""

        $progressCommands = @"
echo '=== Process Status ==='
ps aux | grep bruno | head -3
echo ''
echo '=== GPU Status ==='
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ''
echo '=== Workspace ==='
ls -la /workspace/*.db 2>/dev/null; ls -la /workspace/models/ 2>/dev/null; echo 'Done'
"@

        $output = Invoke-VastSSHCommand -Commands $progressCommands -InstanceId $instanceId -Quiet
        Write-Host $output
    }

    "vast-watch" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID" -ForegroundColor Red
            exit 1
        }

        # Get interval from argument or default to 30 seconds
        $interval = if ($Arg1 -and $Arg1 -match '^\d+$') { [int]$Arg1 } else { 30 }

        # Track start time for elapsed time calculation
        $watchStartTime = Get-Date
        $lastProcessRunning = $true

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Vast.ai Live Dashboard" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Instance: $instanceId" -ForegroundColor Yellow
        Write-Host "Refresh: Every $interval seconds" -ForegroundColor Gray
        Write-Host "Press Ctrl+C to exit" -ForegroundColor Gray
        Write-Host ""

        while ($true) {
            # Clear screen and redraw
            Clear-Host

            $currentTime = Get-Date
            $elapsed = $currentTime - $watchStartTime
            $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed

            # Header
            Write-Host ""
            Write-Host "  ╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
            Write-Host "  ║              VAST.AI ABLITERATION DASHBOARD                  ║" -ForegroundColor Cyan
            Write-Host "  ╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  Instance: $instanceId    Time: $(Get-Date -Format 'HH:mm:ss')    Watch: $elapsedStr" -ForegroundColor Gray
            Write-Host ""

            # Get all status info in one SSH call for efficiency
            $statusCommands = 'PROC=$(ps aux | grep "[h]eretic" | head -1); if [ -n "$PROC" ]; then MODEL=$(echo "$PROC" | awk "{for(i=1;i<=NF;i++) if(\$i ~ /\\//) print \$i}" | head -1); RUNTIME=$(echo "$PROC" | awk "{print \$10}"); CPU=$(echo "$PROC" | awk "{print \$3}"); MEM=$(echo "$PROC" | awk "{print \$4}"); echo "PROCESS:RUNNING"; echo "MODEL:$MODEL"; echo "RUNTIME:$RUNTIME"; echo "CPU:$CPU"; echo "MEM:$MEM"; else echo "PROCESS:STOPPED"; fi; echo "---"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null; echo "---"; if [ -d /workspace/models ]; then ls -1 /workspace/models/ 2>/dev/null | head -5; else echo "(none)"; fi; echo "---"; df -h /workspace 2>/dev/null | tail -1 | awk "{print \$3\"/\"\$2\" (\"\$5\" used)"}"'

            $output = Invoke-VastSSHCommand -Commands $statusCommands -InstanceId $instanceId -Quiet -TimeoutSeconds 15

            # Parse the output
            $sections = ($output -join "`n") -split '---'
            $processSection = $sections[0] -split "`n" | Where-Object { $_ -match '^(PROCESS|MODEL|RUNTIME|CPU|MEM):' }
            $gpuSection = $sections[1] -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch 'Welcome to' -and $_ -notmatch 'authentication' }
            $modelsSection = $sections[2] -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch 'Welcome to' }
            $diskSection = $sections[3] -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch 'Welcome to' } | Select-Object -First 1

            # Parse process info
            $processRunning = $false
            $modelName = "N/A"
            $runtime = "N/A"
            $cpuUsage = "N/A"
            $memUsage = "N/A"

            foreach ($line in $processSection) {
                if ($line -match '^PROCESS:(.+)') { $processRunning = ($matches[1] -eq "RUNNING") }
                if ($line -match '^MODEL:(.+)') { $modelName = $matches[1].Trim() }
                if ($line -match '^RUNTIME:(.+)') { $runtime = $matches[1].Trim() }
                if ($line -match '^CPU:(.+)') { $cpuUsage = $matches[1].Trim() + "%" }
                if ($line -match '^MEM:(.+)') { $memUsage = $matches[1].Trim() + "%" }
            }

            # Process Status Section
            Write-Host "  ┌─ PROCESS ───────────────────────────────────────────────────┐" -ForegroundColor White
            if ($processRunning) {
                Write-Host "  │  Status:  " -NoNewline -ForegroundColor White
                Write-Host "● RUNNING" -NoNewline -ForegroundColor Green
                Write-Host "                                             │" -ForegroundColor White
                Write-Host "  │  Model:   $($modelName.PadRight(50))│" -ForegroundColor White
                Write-Host "  │  Runtime: $($runtime.PadRight(50))│" -ForegroundColor White
                Write-Host "  │  CPU:     $($cpuUsage.PadRight(20))Mem: $($memUsage.PadRight(22))│" -ForegroundColor White
            } else {
                Write-Host "  │  Status:  " -NoNewline -ForegroundColor White
                Write-Host "○ NOT RUNNING" -NoNewline -ForegroundColor Yellow
                Write-Host "                                         │" -ForegroundColor White
                Write-Host "  │                                                              │" -ForegroundColor White

                # Check if process just finished
                if ($lastProcessRunning -and -not $processRunning) {
                    Write-Host "  │  " -NoNewline -ForegroundColor White
                    Write-Host "★ ABLITERATION COMPLETE! ★" -NoNewline -ForegroundColor Green
                    Write-Host "                                  │" -ForegroundColor White
                }
            }
            Write-Host "  └──────────────────────────────────────────────────────────────┘" -ForegroundColor White
            Write-Host ""

            # GPU Status Section
            Write-Host "  ┌─ GPUs ────────────────────────────────────────────────────────┐" -ForegroundColor White
            $gpuIndex = 0
            foreach ($gpuLine in $gpuSection) {
                # Match: GPU_NAME, UTIL%, MEM_USED, MEM_TOTAL, TEMP, POWER
                $gpuPattern = '^\s*(.+?),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+\.?\d*)'
                if ($gpuLine -match $gpuPattern) {
                    $gpuName = $matches[1].Trim()
                    $gpuUtil = [int]$matches[2]
                    $memUsed = [int]$matches[3]
                    $memTotal = [int]$matches[4]
                    $temp = $matches[5]
                    $power = $matches[6]

                    $memPercent = [math]::Round(($memUsed / $memTotal) * 100)
                    $memUsedGB = [math]::Round($memUsed / 1024, 1)
                    $memTotalGB = [math]::Round($memTotal / 1024, 1)

                    # Create progress bar for GPU util
                    $barLen = 20
                    $filledLen = [math]::Round(($gpuUtil / 100) * $barLen)
                    $bar = ("█" * $filledLen) + ("░" * ($barLen - $filledLen))

                    $utilColor = if ($gpuUtil -gt 80) { "Green" } elseif ($gpuUtil -gt 30) { "Yellow" } else { "Gray" }

                    Write-Host "  │  GPU $gpuIndex : $($gpuName.Substring(0, [Math]::Min(25, $gpuName.Length)).PadRight(25))" -NoNewline -ForegroundColor White
                    Write-Host "$temp°C" -NoNewline -ForegroundColor Cyan
                    Write-Host "  ${power}W" -NoNewline -ForegroundColor Gray
                    Write-Host "       │" -ForegroundColor White

                    Write-Host "  │         Util: " -NoNewline -ForegroundColor White
                    Write-Host "$bar" -NoNewline -ForegroundColor $utilColor
                    Write-Host " $($gpuUtil.ToString().PadLeft(3))%" -NoNewline -ForegroundColor $utilColor
                    Write-Host "                  │" -ForegroundColor White

                    Write-Host "  │         VRAM: $memUsedGB / $memTotalGB GB ($memPercent%)" -NoNewline -ForegroundColor White
                    $padding = 62 - "         VRAM: $memUsedGB / $memTotalGB GB ($memPercent%)".Length
                    Write-Host (" " * [Math]::Max(1, $padding)) -NoNewline
                    Write-Host "│" -ForegroundColor White

                    $gpuIndex++
                }
            }
            if ($gpuIndex -eq 0) {
                Write-Host "  │  (No GPU data available)                                    │" -ForegroundColor Gray
            }
            Write-Host "  └──────────────────────────────────────────────────────────────┘" -ForegroundColor White
            Write-Host ""

            # Output Models Section
            Write-Host "  ┌─ OUTPUT MODELS ────────────────────────────────────────────────┐" -ForegroundColor White
            $modelCount = 0
            foreach ($model in $modelsSection) {
                $modelTrimmed = $model.Trim()
                if ($modelTrimmed -and $modelTrimmed -ne "(none)") {
                    Write-Host "  │    ✓ $($modelTrimmed.PadRight(56))│" -ForegroundColor Green
                    $modelCount++
                }
            }
            if ($modelCount -eq 0) {
                Write-Host "  │    (No models saved yet - will appear when complete)        │" -ForegroundColor Gray
            }
            Write-Host "  └──────────────────────────────────────────────────────────────┘" -ForegroundColor White
            Write-Host ""

            # Disk Usage
            if ($diskSection) {
                Write-Host "  Disk: $diskSection" -ForegroundColor Gray
            }

            Write-Host ""
            Write-Host "  ────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
            Write-Host "  Refreshing in $interval seconds... (Ctrl+C to exit)" -ForegroundColor DarkGray
            Write-Host ""

            # Commands hint
            Write-Host "  Quick commands:" -ForegroundColor DarkGray
            Write-Host "    .\runpod.ps1 vast-download-model  - Download completed model" -ForegroundColor DarkGray
            Write-Host "    .\runpod.ps1 vast-stop            - Stop instance (save money!)" -ForegroundColor DarkGray

            $lastProcessRunning = $processRunning

            # Sleep for interval
            Start-Sleep -Seconds $interval
        }
    }

    "vast-hf-login" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        if ($HF_TOKEN) {
            Write-Host "Configuring HuggingFace token on Vast.ai instance..." -ForegroundColor Green
            $output = Invoke-VastSSHCommand -Commands "huggingface-cli login --token $HF_TOKEN" -InstanceId $instanceId
            Write-Host $output
        } else {
            Write-Host "Enter HuggingFace token:" -ForegroundColor Yellow
            $token = Read-Host
            $output = Invoke-VastSSHCommand -Commands "huggingface-cli login --token $token" -InstanceId $instanceId
            Write-Host $output
        }

        Write-Host "HuggingFace login complete!" -ForegroundColor Green
    }

    "vast-hf-test" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host "Testing HuggingFace authentication on Vast.ai..." -ForegroundColor Green
        $output = Invoke-VastSSHCommand -Commands "huggingface-cli whoami" -InstanceId $instanceId -Quiet
        Write-Host $output
    }

    "vast-models" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "Listing abliterated models on Vast.ai..." -ForegroundColor Cyan
        Write-Host ""

        $listCommands = 'echo "Models in /workspace/models:"; echo ""; if [ -d /workspace/models ]; then ls -lh /workspace/models/ 2>/dev/null | grep -v "^total"; echo ""; echo "Model sizes:"; du -sh /workspace/models/*/ 2>/dev/null; else echo "  (models directory does not exist yet)"; fi'

        $output = Invoke-VastSSHCommand -Commands $listCommands -InstanceId $instanceId -Quiet
        Write-Host $output

        Write-Host ""
        Write-Host "Download a model: .\runpod.ps1 vast-download-model [model-name]" -ForegroundColor Yellow
    }

    "vast-download" {
        if (-not $Arg1) {
            Write-Host "ERROR: Specify remote path to download" -ForegroundColor Red
            Write-Host ""
            Write-Host "Usage: .\runpod.ps1 vast-download [remote-path] [local-path]" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Examples:" -ForegroundColor Yellow
            Write-Host "  .\runpod.ps1 vast-download /workspace/models/my-model" -ForegroundColor White
            Write-Host "  .\runpod.ps1 vast-download /workspace/output.txt C:\Downloads\" -ForegroundColor White
            exit 1
        }

        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required for file download" -ForegroundColor Red
            exit 1
        }

        $remotePath = $Arg1
        $localPath = if ($Arg2) { $Arg2 } else { $LOCAL_DIR }

        # Convert Windows path to WSL path
        $wslLocalPath = ($localPath -replace '^([A-Za-z]):', '/mnt/$1' -replace '\\', '/').ToLower()

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Downloading from Vast.ai" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Remote: $remotePath" -ForegroundColor Yellow
        Write-Host "Local:  $localPath" -ForegroundColor Yellow
        Write-Host ""

        # Ensure WSL has SSH key
        Ensure-WSLSSHKey | Out-Null

        # Get SSH target
        $sshTarget = Get-VastSSHTarget -InstanceId $instanceId
        if (-not $sshTarget) {
            Write-Host "ERROR: Could not get SSH connection info" -ForegroundColor Red
            exit 1
        }

        # Parse host and port from sshTarget (format: root@HOST -p PORT)
        if ($sshTarget -match 'root@([\d.]+)\s+-p\s+(\d+)') {
            $sshHost = $matches[1]
            $sshPort = $matches[2]
        } else {
            Write-Host "ERROR: Could not parse SSH target: $sshTarget" -ForegroundColor Red
            exit 1
        }

        # Ensure local directory exists
        if (-not (Test-Path $localPath)) {
            Write-Host "Creating local directory: $localPath" -ForegroundColor Gray
            New-Item -ItemType Directory -Path $localPath -Force | Out-Null
        }

        Write-Host "Starting download (this may take a while for large models)..." -ForegroundColor Yellow
        Write-Host ""

        # Use SCP via WSL with progress
        $scpCmd = "scp -r -o StrictHostKeyChecking=no -P $sshPort root@${sshHost}:$remotePath $wslLocalPath"

        Write-Host "Running: $scpCmd" -ForegroundColor Gray
        Write-Host ""

        # Execute SCP - use wsl directly for real-time output
        wsl -e bash -c $scpCmd

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "  Download Complete!" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Files saved to: $localPath" -ForegroundColor Cyan
        } else {
            Write-Host ""
            Write-Host "ERROR: Download failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }

    "vast-download-model" {
        $instanceId = $VAST_INSTANCE_ID
        if (-not $instanceId) {
            Write-Host "Looking for running instance..." -ForegroundColor Gray
            $instanceId = Get-FirstVastInstance
        }

        if (-not $instanceId) {
            Write-Host "ERROR: No instance ID. Run vast-create-pod first." -ForegroundColor Red
            exit 1
        }

        if (-not (Test-WSLAvailable)) {
            Write-Host "ERROR: WSL required for model download" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Download Abliterated Model" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""

        # First, list available models if no specific model requested
        $modelName = $Arg1

        if (-not $modelName) {
            Write-Host "Scanning for available models..." -ForegroundColor Yellow
            Write-Host ""

            $listCmd = "ls -1 $VAST_MODELS_DIR/ 2>/dev/null | grep -E '.*-bruno$'"
            $models = Invoke-VastSSHCommand -Commands $listCmd -InstanceId $instanceId -Quiet

            if (-not $models -or $models -match "No such file") {
                Write-Host "ERROR: No abliterated models found in $VAST_MODELS_DIR" -ForegroundColor Red
                Write-Host ""
                Write-Host "Run abliteration first: .\runpod.ps1 vast-run [model]" -ForegroundColor Yellow
                exit 1
            }

            # Parse models into array
            $modelList = $models -split "`n" | Where-Object { $_.Trim() -ne "" -and $_ -notmatch "^exit" -and $_ -notmatch "^Connection" }

            if ($modelList.Count -eq 0) {
                Write-Host "ERROR: No abliterated models found" -ForegroundColor Red
                exit 1
            } elseif ($modelList.Count -eq 1) {
                $modelName = $modelList[0].Trim()
                Write-Host "Found model: $modelName" -ForegroundColor Green
            } else {
                Write-Host "Available models:" -ForegroundColor Cyan
                $i = 1
                foreach ($m in $modelList) {
                    $mName = $m.Trim()
                    if ($mName) {
                        Write-Host "  [$i] $mName" -ForegroundColor White
                        $i++
                    }
                }
                Write-Host ""
                $selection = Read-Host "Select model number (or press Enter for most recent)"

                if ($selection -match '^\d+$' -and [int]$selection -le $modelList.Count -and [int]$selection -gt 0) {
                    $modelName = $modelList[[int]$selection - 1].Trim()
                } else {
                    # Default to last (most recent) model
                    $modelName = $modelList[-1].Trim()
                }
            }
        }

        $remotePath = "$VAST_MODELS_DIR/$modelName"
        $localPath = $LOCAL_MODELS_DIR

        # Convert Windows path to WSL path
        $wslLocalPath = ($localPath -replace '^([A-Za-z]):', '/mnt/$1' -replace '\\', '/').ToLower()

        Write-Host ""
        Write-Host "Model: $modelName" -ForegroundColor Yellow
        Write-Host "Remote: $remotePath" -ForegroundColor Yellow
        Write-Host "Local:  $localPath\$modelName" -ForegroundColor Yellow
        Write-Host ""

        # Get model size
        Write-Host "Checking model size..." -ForegroundColor Gray
        $sizeCmd = "du -sh $remotePath 2>/dev/null | cut -f1"
        $modelSize = Invoke-VastSSHCommand -Commands $sizeCmd -InstanceId $instanceId -Quiet
        $modelSize = ($modelSize -split "`n" | Where-Object { $_ -match '^[0-9]' } | Select-Object -First 1)
        if ($modelSize) {
            Write-Host "Model size: $($modelSize.Trim())" -ForegroundColor Cyan
        }

        Write-Host ""
        Write-Host "NOTE: Large models (70B+) can take 30-60+ minutes to download." -ForegroundColor Yellow
        Write-Host "      Make sure you have enough disk space locally." -ForegroundColor Yellow
        Write-Host ""

        $confirm = Read-Host "Start download? (y/n)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "Cancelled." -ForegroundColor Yellow
            exit 0
        }

        # Ensure WSL has SSH key
        Ensure-WSLSSHKey | Out-Null

        # Get SSH target
        $sshTarget = Get-VastSSHTarget -InstanceId $instanceId
        if (-not $sshTarget) {
            Write-Host "ERROR: Could not get SSH connection info" -ForegroundColor Red
            exit 1
        }

        # Parse host and port
        if ($sshTarget -match 'root@([\d.]+)\s+-p\s+(\d+)') {
            $sshHost = $matches[1]
            $sshPort = $matches[2]
        } else {
            Write-Host "ERROR: Could not parse SSH target" -ForegroundColor Red
            exit 1
        }

        # Ensure local models directory exists
        if (-not (Test-Path $localPath)) {
            Write-Host "Creating local models directory: $localPath" -ForegroundColor Gray
            New-Item -ItemType Directory -Path $localPath -Force | Out-Null
        }

        Write-Host ""
        Write-Host "Starting download..." -ForegroundColor Green
        Write-Host "(Progress will be shown. Press Ctrl+C to cancel.)" -ForegroundColor Gray
        Write-Host ""

        $startTime = Get-Date

        # Use rsync for better progress indication (if available) or fallback to scp
        # Check if rsync is available
        $rsyncCheck = wsl -e bash -c "which rsync 2>/dev/null"

        if ($rsyncCheck) {
            # Use rsync with progress
            $rsyncCmd = "rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no -p $sshPort' root@${sshHost}:$remotePath $wslLocalPath/"
            Write-Host "Using rsync for transfer..." -ForegroundColor Gray
            wsl -e bash -c $rsyncCmd
        } else {
            # Fallback to scp
            $scpCmd = "scp -r -o StrictHostKeyChecking=no -P $sshPort root@${sshHost}:$remotePath $wslLocalPath/"
            Write-Host "Using scp for transfer..." -ForegroundColor Gray
            wsl -e bash -c $scpCmd
        }

        $endTime = Get-Date
        $duration = $endTime - $startTime

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "  Model Downloaded Successfully!" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "Model: $modelName" -ForegroundColor Cyan
            Write-Host "Location: $localPath\$modelName" -ForegroundColor Cyan
            Write-Host "Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "You can now use this model locally in your chat app!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Don't forget to stop the Vast.ai instance:" -ForegroundColor Yellow
            Write-Host "  .\runpod.ps1 vast-stop" -ForegroundColor White
        } else {
            Write-Host ""
            Write-Host "ERROR: Download failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
            Write-Host ""
            Write-Host "Troubleshooting:" -ForegroundColor Yellow
            Write-Host "  - Check disk space: Get-PSDrive C" -ForegroundColor White
            Write-Host "  - Verify model exists: .\runpod.ps1 vast-models" -ForegroundColor White
            Write-Host "  - Try manual download: .\runpod.ps1 vast-download $remotePath" -ForegroundColor White
        }
    }

    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host "Run '.\runpod.ps1 help' for usage" -ForegroundColor Yellow
    }
}
