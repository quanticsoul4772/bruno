# RunPod Connection Helper for Windows
# Heretic LLM Abliteration Automation
#
# This script uses a hybrid approach:
# - runpodctl CLI for pod management and SSH info (fast, reliable)
# - WSL + SSH heredoc for command execution (required for PTY support)
# - GraphQL API as fallback and for advanced operations

param(
    [Parameter(Position=0)]
    [string]$Action = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1,
    
    [Parameter(Position=2)]
    [string]$Arg2
)

# ===== CONFIGURE THESE =====
$RUNPOD_API_KEY = ""  # Your RunPod API key (get from runpod.io/console/user/settings)
$RUNPOD_HOST = "103.196.86.69"     # Auto-configured by create-pod, or set manually
$RUNPOD_PORT = "19214"     # Auto-configured by create-pod, or set manually
$RUNPOD_USER = "root"
$SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519"  # Your SSH key

# RunPod SSH proxy (more reliable than direct TCP)
$RUNPOD_SSH_PROXY = "hjnmjr0qk1af9j-64411a7d@ssh.runpod.io"  # Auto-updated by create-pod
$USE_SSH_PROXY = $true  # Using RunPod proxy since direct TCP isn't working
$HF_TOKEN = "" # Optional: Set HuggingFace token here or use hf-login command

# runpodctl Configuration
# runpodctl is used for pod management and getting SSH connection info
# Download: https://github.com/runpod/runpodctl/releases
$RUNPODCTL_PATH = "$PSScriptRoot\runpodctl.exe"  # Path to runpodctl executable

# WSL SSH Configuration
# The RunPod SSH proxy requires a PTY (pseudo-terminal) which Windows SSH can't provide.
# Solution: Use WSL (Windows Subsystem for Linux) with heredoc to allocate a proper PTY.
# Note: runpodctl exec only supports Python scripts, not general shell commands.
$USE_WSL_SSH = $true  # Set to $false if WSL is not available

# Pod creation defaults
# NOTE: RTX 4090 has 24GB VRAM, good for 8B models
$DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
# PyTorch 2.4.0 is compatible with transformers 4.55+ (no manual upgrade needed!)
# Python 3.11, CUDA 12.4.1 - works out of the box with heretic-llm
$DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# IMPORTANT: 20GB container disk is NOT enough (fills with pip packages)
# 40GB minimum; 50GB+ recommended for larger models
$DEFAULT_VOLUME_GB = 50
$DEFAULT_CONTAINER_DISK_GB = 40
# ===========================

$LOCAL_DIR = "C:\Development\Projects\heretic"
$REMOTE_DIR = "/workspace/heretic"
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

function Show-Help {
    Write-Host @"
Heretic RunPod Automation
=========================

SETUP:
  1. Install WSL: wsl --install (admin PowerShell, then reboot)
  2. Download runpodctl: .\runpod.ps1 install-runpodctl
  3. Set RUNPOD_API_KEY in this script (get from runpod.io settings)
  4. Add SSH key to RunPod (Settings -> SSH Keys)
  5. Run: .\runpod.ps1 create-pod
  6. Run: .\runpod.ps1 setup
  7. Run: .\runpod.ps1 test

POD MANAGEMENT:
  create-pod [gpu] - Create pod (default: RTX 4090)
  list-pods        - List your pods (uses runpodctl)
  get-ssh [podId]  - Get SSH details for a pod
  stop-pod [podId] - Stop pod (saves volume, stops billing)
  start-pod [podId]- Start a stopped pod
  terminate-pod [podId] - Delete pod permanently
  gpus             - List available GPU types
  
TOOLS:
  install-runpodctl - Download runpodctl CLI
  check-tools       - Verify all tools are installed

CORE COMMANDS:
  connect       - SSH to RunPod (interactive)
  setup         - Install heretic on RunPod (uses WSL)
  test          - Test with Qwen3-4B (~15 min)
  run <model>   - Run heretic on model
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
  hf-login      - Configure HuggingFace token on RunPod
  hf-test       - Test HuggingFace authentication
  
QUICK START:
  .\runpod.ps1 create-pod                    # 1. Create pod
  .\runpod.ps1 setup                         # 2. Install heretic
  .\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507  # 3. Abliterate
  .\runpod.ps1 stop-pod                      # 4. Stop when done

EXAMPLES:
  .\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct
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
    $apiKey = if ($RUNPOD_API_KEY) { $RUNPOD_API_KEY } else { $env:RUNPOD_API_KEY }
    if (-not $apiKey) {
        Write-Host "ERROR: RUNPOD_API_KEY not set!" -ForegroundColor Red
        Write-Host "Set it in this script, or as environment variable: `$env:RUNPOD_API_KEY = 'your-key'" -ForegroundColor Yellow
        Write-Host "Get your API key from: https://runpod.io/console/user/settings" -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-RunPodGraphQL {
    param(
        [string]$Query,
        [hashtable]$Variables = $null
    )
    
    # Support API key from environment variable as fallback
    $apiKey = if ($RUNPOD_API_KEY) { $RUNPOD_API_KEY } else { $env:RUNPOD_API_KEY }
    if (-not $apiKey) {
        Write-Host "ERROR: No API key found. Set RUNPOD_API_KEY in script or environment." -ForegroundColor Red
        return $null
    }
    
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
        [string]$Host,
        [string]$Port,
        [string]$SSHProxy
    )
    
    $scriptPath = $PSCommandPath
    $content = Get-Content $scriptPath -Raw
    
    # Update RUNPOD_HOST
    if ($Host) {
        $content = $content -replace '\$RUNPOD_HOST = "[^"]*"', "`$RUNPOD_HOST = `"$Host`""
        Set-Variable -Name RUNPOD_HOST -Value $Host -Scope Script
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
        $podName = "heretic-$(Get-Date -Format 'MMdd-HHmm')"
        
        Write-Host "Creating pod with $gpuType..." -ForegroundColor Green
        
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
                
                # Get the RunPod user ID from the API response or use default
                # The SSH proxy format is: podId-userId@ssh.runpod.io
                # We need to fetch the user ID from the myself query
                $userQuery = @"
query Me {
  myself {
    id
  }
}
"@
                $userResponse = Invoke-RunPodGraphQL -Query $userQuery
                $userId = if ($userResponse -and $userResponse.myself) { $userResponse.myself.id } else { "" }
                
                # Build SSH proxy address
                $sshProxyAddr = "$podId-$userId@ssh.runpod.io"
                
                # Auto-update script config including SSH proxy
                Update-ScriptConfig -Host $sshDetails.Host -Port $sshDetails.Port -SSHProxy $sshProxyAddr
                
                Write-Host "  SSH Proxy: $sshProxyAddr" -ForegroundColor Cyan
                Write-Host "`n" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host "  Pod Ready! Script auto-configured." -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Next steps:" -ForegroundColor Yellow
                Write-Host "  .\runpod.ps1 setup    # Install heretic" -ForegroundColor White
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
                    Update-ScriptConfig -Host $sshInfo.Host -Port $sshInfo.Port
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
                        Update-ScriptConfig -Host $port.ip -Port $port.publicPort
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

echo '[3/4] Installing heretic-llm...'
pip install --quiet heretic-llm

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
echo '  or: .\runpod.ps1 run <model-name>'
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
heretic Qwen/Qwen3-4B-Instruct-2507
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
heretic $Arg1
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
            Write-Host "Example: .\runpod.ps1 vllm-start /workspace/heretic/models/llama-8b" -ForegroundColor Yellow
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
                "apiKey: $RUNPOD_API_KEY" | Out-File -FilePath "$configDir\.runpod.yaml" -Encoding utf8
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
        
        # Check SSH key
        Write-Host "SSH Key: " -NoNewline
        if (Test-Path $SSH_KEY) {
            Write-Host "OK ($SSH_KEY)" -ForegroundColor Green
        } else {
            Write-Host "NOT FOUND" -ForegroundColor Red
            Write-Host "  Create: ssh-keygen -t ed25519" -ForegroundColor Yellow
        }
        
        # Check API key
        Write-Host "API Key: " -NoNewline
        if ($RUNPOD_API_KEY -and $RUNPOD_API_KEY.Length -gt 10) {
            Write-Host "OK (configured)" -ForegroundColor Green
        } else {
            Write-Host "NOT CONFIGURED" -ForegroundColor Red
            Write-Host "  Set RUNPOD_API_KEY in this script" -ForegroundColor Yellow
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
ps aux | grep heretic | head -3
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
    
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host "Run '.\runpod.ps1 help' for usage" -ForegroundColor Yellow
    }
}
