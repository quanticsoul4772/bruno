#!/bin/bash
# Remote setup script for Vast.ai instance
# Run this ON the instance after SSH is available
# Usage: bash /workspace/remote_setup.sh

set -e

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 3

echo "=== Installing CrewAI ==="
pip install crewai crewai-tools

echo "=== Cleaning partial uploads ==="
rm -rf /workspace/Frontend-3B /workspace/Backend-3B /workspace/Test-3B \
       /workspace/Security-3B /workspace/Docs-3B /workspace/DevOps-3B \
       /workspace/Orchestrator-14B

echo "=== Ready for model uploads ==="
echo "Upload models now with: bash scripts/upload_and_run.sh <HOST> <PORT>"
