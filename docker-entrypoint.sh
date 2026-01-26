#!/bin/bash
# Heretic Docker Entrypoint
# Handles environment setup and passes commands through

set -e

# =============================================================================
# Environment Setup
# =============================================================================

# Ensure HuggingFace cache directory exists
mkdir -p "${HF_HOME:-/workspace/.cache/huggingface}"

# Configure HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    echo "Configuring HuggingFace authentication..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# =============================================================================
# GPU Check
# =============================================================================

echo "========================================"
echo "  Heretic LLM Abliteration Container"
echo "========================================"
echo ""

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# =============================================================================
# Command Execution
# =============================================================================

# If no arguments, show usage
if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  docker run --gpus all quanticsoul4772/heretic heretic <model>"
    echo ""
    echo "Examples:"
    echo "  heretic Qwen/Qwen3-4B-Instruct-2507"
    echo "  heretic Qwen/Qwen3-4B-Instruct-2507 --auto-select"
    echo "  heretic meta-llama/Llama-3.1-8B-Instruct --auto-select --hf-upload user/repo"
    echo ""
    echo "Environment Variables:"
    echo "  HF_TOKEN     - HuggingFace token for gated models and uploads"
    echo "  HF_HOME      - HuggingFace cache directory (default: /workspace/.cache/huggingface)"
    echo ""
    exit 0
fi

# Execute the command
exec "$@"
