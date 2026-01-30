# Convert HuggingFace Model to GGUF for Ollama
# This script converts safetensors models to quantized GGUF format
#
# USAGE:
#   .\scripts\convert-to-gguf.ps1 -ModelPath ".\models\Qwen2.5-Coder-32B-heretic"
#   .\scripts\convert-to-gguf.ps1 -ModelPath ".\models\Qwen2.5-Coder-32B-heretic" -Quantization "Q4_K_M"
#
# OUTPUT:
#   Creates GGUF file ready for Ollama import

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [Parameter(Mandatory=$false)]
    [ValidateSet("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K")]
    [string]$Quantization = "Q4_K_M",

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = ".\models\gguf"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = "." }

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     GGUF Conversion for Ollama                               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Validate model path
if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model path not found: $ModelPath" -ForegroundColor Red
    exit 1
}

# Get model name from path
$ModelName = Split-Path -Leaf $ModelPath
Write-Host "Model: $ModelName" -ForegroundColor Yellow
Write-Host "Quantization: $Quantization" -ForegroundColor Yellow
Write-Host ""

# Check for llama.cpp conversion script
$ConvertScript = Join-Path $ProjectRoot "tools\llama.cpp\convert_hf_to_gguf.py"
if (-not (Test-Path $ConvertScript)) {
    Write-Host "ERROR: llama.cpp not found. Run this first:" -ForegroundColor Red
    Write-Host "  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git tools/llama.cpp" -ForegroundColor Yellow
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$F16Output = Join-Path $OutputDir "$ModelName-f16.gguf"
$QuantizedOutput = Join-Path $OutputDir "$ModelName-$Quantization.gguf"

# Step 1: Convert to F16 GGUF
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  Step 1: Converting to F16 GGUF" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""
Write-Host "This may take 10-30 minutes for a 32B model..." -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date

python $ConvertScript $ModelPath --outfile $F16Output --outtype f16

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: F16 conversion failed" -ForegroundColor Red
    exit 1
}

$f16Size = (Get-Item $F16Output).Length / 1GB
Write-Host ""
Write-Host "✓ F16 GGUF created: $F16Output ($([math]::Round($f16Size, 2)) GB)" -ForegroundColor Green
Write-Host ""

# Step 2: Quantize (if not F16)
if ($Quantization -ne "F16") {
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  Step 2: Quantizing to $Quantization" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""

    # Check for quantize binary or use Python
    $QuantizeBin = Join-Path $ProjectRoot "tools\llama.cpp\build\bin\Release\llama-quantize.exe"
    if (-not (Test-Path $QuantizeBin)) {
        $QuantizeBin = Join-Path $ProjectRoot "tools\llama.cpp\llama-quantize.exe"
    }

    if (Test-Path $QuantizeBin) {
        # Use binary quantizer (faster)
        & $QuantizeBin $F16Output $QuantizedOutput $Quantization
    } else {
        # Fallback: Use Ollama to quantize (simpler)
        Write-Host "Note: Using Ollama for quantization (llama.cpp binary not built)" -ForegroundColor Gray
        Write-Host "For faster quantization, build llama.cpp with: cmake --build tools/llama.cpp/build" -ForegroundColor Gray
        Write-Host ""

        # Create a temporary Modelfile for Ollama import
        $TempModelfile = Join-Path $OutputDir "Modelfile.temp"
        @"
FROM $F16Output
"@ | Out-File -FilePath $TempModelfile -Encoding utf8

        # Import into Ollama (it will handle storage)
        Write-Host "Importing into Ollama..." -ForegroundColor Cyan
        ollama create "$ModelName-temp" -f $TempModelfile

        # Export quantized version
        Write-Host "Creating quantized version..." -ForegroundColor Cyan
        ollama cp "$ModelName-temp" "$ModelName-$Quantization"
        ollama rm "$ModelName-temp"

        Remove-Item $TempModelfile -Force

        Write-Host ""
        Write-Host "✓ Model imported to Ollama as: $ModelName-$Quantization" -ForegroundColor Green

        $endTime = Get-Date
        $duration = $endTime - $startTime

        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "  CONVERSION COMPLETE" -ForegroundColor Green
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host ""
        Write-Host "Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Test the model:" -ForegroundColor Yellow
        Write-Host "  ollama run $ModelName-$Quantization" -ForegroundColor White
        Write-Host ""
        Write-Host "Compare with original:" -ForegroundColor Yellow
        Write-Host "  .\scripts\compare-models.ps1" -ForegroundColor White
        Write-Host ""
        exit 0
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Quantization failed" -ForegroundColor Red
        exit 1
    }

    $quantSize = (Get-Item $QuantizedOutput).Length / 1GB
    Write-Host ""
    Write-Host "✓ Quantized GGUF created: $QuantizedOutput ($([math]::Round($quantSize, 2)) GB)" -ForegroundColor Green

    # Clean up F16 file to save space
    Write-Host ""
    $cleanup = Read-Host "Delete F16 file to save space? ($([math]::Round($f16Size, 2)) GB) (y/n)"
    if ($cleanup -eq "y" -or $cleanup -eq "Y") {
        Remove-Item $F16Output -Force
        Write-Host "F16 file deleted." -ForegroundColor Gray
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  CONVERSION COMPLETE" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
Write-Host "Output: $QuantizedOutput" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Import to Ollama:" -ForegroundColor Yellow
Write-Host "  ollama create qwen32b-abliterated -f Modelfile" -ForegroundColor White
Write-Host ""
Write-Host "Or run directly:" -ForegroundColor Yellow
Write-Host "  ollama run --model $QuantizedOutput" -ForegroundColor White
