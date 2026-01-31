# Utility Scripts

Collection of utility scripts for testing, deployment, and cloud GPU management.

> **Note:** Most cloud GPU operations should use the `heretic-vast` CLI instead of these scripts.
> These scripts are maintained for reference and specific use cases.

## Cloud GPU Management

**Recommended: Use `heretic-vast` CLI**
```bash
uv run heretic-vast create RTX_4090 4
uv run heretic-vast setup
uv run heretic-vast watch
```

**Legacy Scripts (for reference):**
- `runpod.ps1` - PowerShell RunPod management (use heretic-vast instead)
- `setup_runpod_a100.ps1` - A100 setup automation
- `start-abliteration.ps1` - Start abliteration on cloud
- `vast.bat` - Batch wrapper for Vast.ai
- `monitor_a100.ps1` - A100 monitoring
- `test-on-cloud.ps1` - Cloud testing

## Model Management

- `download-model.ps1` - Download models from cloud (use `heretic-vast download` instead)
- `compare-models.ps1` - Compare original vs abliterated models
- `convert-to-gguf.ps1` - Convert models to GGUF format

## Utilities

- `generate_checksums.py` - Generate SHA256 checksums for model files
- `verify_checksums.py` - Verify model file integrity
- `patch_heretic.py` - Apply patches to heretic installation
- `test_model.py` - Quick model testing script

## Setup Instructions

- `setup_runpod_a100.md` - Detailed RunPod A100 80GB setup guide
- `setup_pod.sh` / `setup-pod.sh` - Shell setup scripts
- `start_abliteration.sh` - Start abliteration in background

## Usage Notes

**PowerShell Scripts:**
- Run from project root
- May require execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Shell Scripts:**
- Make executable: `chmod +x scripts/*.sh`
- Run from project root: `./scripts/setup_pod.sh`

**Python Scripts:**
- Run with uv: `uv run python scripts/test_model.py`
