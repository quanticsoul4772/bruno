# RunPod Connection Helper for Windows
# Heretic LLM Abliteration Automation

param(
    [Parameter(Position=0)]
    [string]$Action = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1,
    
    [Parameter(Position=2)]
    [string]$Arg2
)

# ===== CONFIGURE THESE =====
$RUNPOD_HOST = "your-pod-id.runpod.io"  # UPDATE THIS
$RUNPOD_PORT = "12345"                   # UPDATE THIS
$RUNPOD_USER = "root"
$SSH_KEY = ""  # Leave empty for password, or set to key path like: "$env:USERPROFILE\.ssh\id_ed25519"
$HF_TOKEN = "" # Optional: Set HuggingFace token here or use hf-login command
# ===========================

$LOCAL_DIR = "C:\Development\Projects\heretic"
$REMOTE_DIR = "/workspace/heretic"
$VLLM_PORT = 8000

function Show-Help {
    Write-Host @"
Heretic RunPod Automation
=========================

SETUP:
  Edit this script and set RUNPOD_HOST and RUNPOD_PORT

CORE COMMANDS:
  connect       - SSH to RunPod
  setup         - Install heretic-llm on RunPod
  test          - Test with Qwen3-4B (~15 min)
  run <model>   - Run heretic on model
  
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
  status        - GPU status
  monitor       - Live GPU monitoring
  logs          - View vLLM logs
  
HUGGINGFACE:
  hf-login      - Configure HuggingFace token on RunPod
  hf-test       - Test HuggingFace authentication
  
EXAMPLES:
  .\runpod.ps1 setup
  .\runpod.ps1 test
  .\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct
  .\runpod.ps1 vllm-install
  .\runpod.ps1 vllm-start /workspace/heretic/models/llama-8b
  .\runpod.ps1 tunnel

"@
}

function Check-Config {
    if ($RUNPOD_HOST -eq "your-pod-id.runpod.io") {
        Write-Host "ERROR: Update RUNPOD_HOST in this script first!" -ForegroundColor Red
        exit 1
    }
}

function Get-SSHCommand {
    $cmd = "ssh"
    if ($SSH_KEY) { 
        if (-not (Test-Path $SSH_KEY)) {
            Write-Host "WARNING: SSH key not found at $SSH_KEY" -ForegroundColor Yellow
            Write-Host "Using password authentication instead" -ForegroundColor Yellow
        } else {
            $cmd += " -i `"$SSH_KEY`""
        }
    }
    $cmd += " ${RUNPOD_USER}@${RUNPOD_HOST} -p $RUNPOD_PORT"
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

switch ($Action) {
    "help" { Show-Help }
    
    "connect" {
        Check-Config
        Write-Host "Connecting to RunPod..." -ForegroundColor Green
        Invoke-Expression (Get-SSHCommand)
    }
    
    "setup" {
        Check-Config
        Write-Host "Setting up Heretic on RunPod..." -ForegroundColor Green
        
        # Install heretic and create directory
        Invoke-SSHCommand "pip install heretic-llm && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && echo 'Heretic installed'"
        
        # Upload config
        if (Test-Path "$LOCAL_DIR\config.toml") {
            Write-Host "Uploading config.toml..." -ForegroundColor Green
            $scp = Get-SCPPrefix
            Invoke-Expression "$scp `"$LOCAL_DIR\config.toml`" ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"
        }
        
        Write-Host "Setup complete!" -ForegroundColor Green
        Write-Host "Next: .\runpod.ps1 test" -ForegroundColor Cyan
    }
    
    "test" {
        Check-Config
        Write-Host "Testing with Qwen3-4B (15-20 min on RTX 5090)..." -ForegroundColor Green
        Invoke-SSHCommand "cd $REMOTE_DIR && heretic Qwen/Qwen3-4B-Instruct-2507"
    }
    
    "run" {
        Check-Config
        if (-not $Arg1) { 
            Write-Host "ERROR: Specify model name" -ForegroundColor Red
            Write-Host "Example: .\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "Running heretic on $Arg1..." -ForegroundColor Green
        
        $configArg = ""
        if (Test-Path "$REMOTE_DIR\config.toml") {
            $configArg = "--config config.toml"
        }
        
        Invoke-SSHCommand "cd $REMOTE_DIR && heretic $configArg $Arg1"
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
        Check-Config
        Invoke-SSHCommand "nvidia-smi"
    }
    
    "monitor" {
        Check-Config
        Write-Host "GPU monitoring (Ctrl+C to exit)..." -ForegroundColor Green
        Invoke-SSHCommand "watch -n 1 nvidia-smi"
    }
    
    "vllm-install" {
        Check-Config
        Write-Host "Installing vLLM..." -ForegroundColor Green
        Invoke-SSHCommand "pip install vllm"
        Write-Host "vLLM installed!" -ForegroundColor Green
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
        Check-Config
        
        if ($HF_TOKEN) {
            Write-Host "Using token from script configuration..." -ForegroundColor Green
            Invoke-SSHCommand "huggingface-cli login --token $HF_TOKEN"
        } else {
            Write-Host "Enter HuggingFace token (will be sent to RunPod):" -ForegroundColor Yellow
            $token = Read-Host -AsSecureString
            $tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))
            Invoke-SSHCommand "huggingface-cli login --token $tokenPlain"
        }
        
        Write-Host "HuggingFace login complete!" -ForegroundColor Green
    }
    
    "hf-test" {
        Check-Config
        Write-Host "Testing HuggingFace authentication..." -ForegroundColor Green
        Invoke-SSHCommand "huggingface-cli whoami"
    }
    
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host "Run '.\runpod.ps1 help' for usage" -ForegroundColor Yellow
    }
}
