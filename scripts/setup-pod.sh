#!/bin/bash
# Heretic RunPod Setup Script
# Run with: curl -sSL https://raw.githubusercontent.com/YOUR_REPO/setup-pod.sh | bash
# Or upload and run: bash setup-pod.sh

set -e  # Exit on error

echo "========================================"
echo "  Heretic RunPod Setup Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Set up HuggingFace cache on volume
echo -e "${YELLOW}[1/5] Setting up HuggingFace cache on volume...${NC}"
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/huggingface

# Add to bashrc for persistence
if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Heretic - HuggingFace cache on volume" >> ~/.bashrc
    echo "export HF_HOME=/workspace/.cache/huggingface" >> ~/.bashrc
    echo "export TRANSFORMERS_CACHE=/workspace/.cache/huggingface" >> ~/.bashrc
    echo -e "${GREEN}  Added HF_HOME to ~/.bashrc${NC}"
else
    echo -e "${GREEN}  HF_HOME already in ~/.bashrc${NC}"
fi
echo -e "${GREEN}  HF cache: $HF_HOME${NC}"

# Step 2: Check disk space
echo ""
echo -e "${YELLOW}[2/5] Checking disk space...${NC}"
ROOT_AVAIL=$(df -h / | awk 'NR==2 {print $4}')
WORKSPACE_AVAIL=$(df -h /workspace 2>/dev/null | awk 'NR==2 {print $4}' || echo "N/A")
echo -e "  Container disk available: ${GREEN}$ROOT_AVAIL${NC}"
echo -e "  Volume (/workspace) available: ${GREEN}$WORKSPACE_AVAIL${NC}"

# Step 3: Check PyTorch version
echo ""
echo -e "${YELLOW}[3/5] Checking PyTorch version...${NC}"
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo -e "  PyTorch version: ${GREEN}$PYTORCH_VERSION${NC}"

# Check if PyTorch needs upgrade (version < 2.2)
if [[ "$PYTORCH_VERSION" == "2.1"* ]] || [[ "$PYTORCH_VERSION" == "2.0"* ]] || [[ "$PYTORCH_VERSION" == "1."* ]]; then
    echo -e "${RED}  WARNING: PyTorch $PYTORCH_VERSION may be incompatible with transformers 4.55+${NC}"
    echo -e "${YELLOW}  Upgrading PyTorch...${NC}"
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}  Upgraded to PyTorch $PYTORCH_VERSION${NC}"
else
    echo -e "${GREEN}  PyTorch version is compatible${NC}"
fi

# Step 4: Install heretic-llm
echo ""
echo -e "${YELLOW}[4/5] Installing heretic-llm...${NC}"
pip install --quiet heretic-llm
HERETIC_VERSION=$(pip show heretic-llm 2>/dev/null | grep Version | awk '{print $2}' || echo "unknown")
echo -e "${GREEN}  Installed heretic-llm v$HERETIC_VERSION${NC}"

# Step 5: Clear GPU memory
echo ""
echo -e "${YELLOW}[5/5] Clearing GPU memory...${NC}"
# Kill any processes using the GPU (except this shell)
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || echo "")
if [ -n "$GPU_PROCS" ]; then
    echo -e "  Found GPU processes: $GPU_PROCS"
    for PID in $GPU_PROCS; do
        if [ "$PID" != "$$" ]; then
            kill $PID 2>/dev/null && echo -e "  ${GREEN}Killed process $PID${NC}" || true
        fi
    done
else
    echo -e "${GREEN}  No GPU processes to kill${NC}"
fi

# Show GPU status
echo ""
echo -e "${YELLOW}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Create workspace directory
mkdir -p /workspace/heretic

# Summary
echo ""
echo "========================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run a test:     heretic Qwen/Qwen3-4B-Instruct-2507"
echo "  2. Or full model:  heretic meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "Output will be saved to: /workspace/heretic/"
echo ""
