# ⛔ STOP: Read RUNBOOK.md FIRST ⛔

## ⛔ CRITICAL - MEMORIZE THESE ⛔

### Git: ALWAYS push to `fork` (abliteration-workflow)
```powershell
git push fork master   # ✅ CORRECT
git push heretic-fork  # ❌ WRONG - causes split repos
```

### Resume Support: ALWAYS use these flags
```
--storage sqlite:////workspace/heretic_study.db
--study-name qwen32b-abliteration
```

### Quick Start
```powershell
cd C:\Development\Projects\heretic
.\start-abliteration.ps1
```

### For 32B+ Models: MUST use `--cache-weights false`
```bash
heretic --model MODEL --cache-weights false --storage sqlite:///study.db --study-name NAME
```

### Current Instance (June 2025)
```bash
ssh -p 37522 root@ssh4.vast.ai 'tail -f /workspace/heretic.log'
```

---

## ⛔ STOP - READ THIS FIRST ⛔

**BEFORE YOU DO ANYTHING ELSE:**

1. **READ THIS ENTIRE FILE** - Not skim. READ.
2. **CHECK what already exists** - Models downloaded? Experiments run? Tools available?
3. **USE EXISTING TOOLS** - `heretic-vast` CLI, `runpod.ps1` script
4. **ASK if unsure** - Don't guess. Don't assume. ASK.

**YOU HAVE FAILED MULTIPLE TIMES BY:**
- Ignoring this documentation after writing it
- Using manual SSH instead of `heretic-vast` CLI
- Destroying running instances without checking status
- Pattern-matching "cloud task" → "manual SSH commands"

**THE FIX:** Read `knowledge.md` and `WORKFLOW.md` FIRST. Every. Single. Time.

---

This file gives Codebuff context about your project: goals, commands, conventions, and gotchas.

## Vision

Heretic is a **neural behavior modification workbench**, not just an "uncensoring tool." The abliteration technique is general and can extract/modify any behavioral direction encoded in model weights - verbosity, hedging, sycophancy, and more. See [ROADMAP.md](ROADMAP.md) for full vision.

## Quickstart
- Setup: `uv sync --all-extras --dev`
- Dev: `uv run heretic <model-name>`
- Test: No test suite - CI runs `ruff format` and `ruff check`
- Chat UI: `python chat_app.py`
- Cloud (Vast.ai): `heretic-vast create A100_80GB 2 && heretic-vast setup && heretic-vast run MODEL`

## Performance Optimizations (NEW)

Heretic includes several performance optimizations:

| Optimization | Speedup | Flag/Config |
|--------------|---------|-------------|
| In-memory weight caching | ~5-10x faster model reset | Automatic |
| torch.compile() | ~1.5-2x inference speedup | `--compile` |
| Early stopping for refusals | ~40-60% faster evaluation | `--refusal-check-tokens 30` |
| Parallel evaluation | ~20-30% faster per trial | Automatic |
| Resume support | Can resume interrupted runs | `--storage sqlite:///study.db` |

**Example with all optimizations:**
```bash
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --compile \
  --storage sqlite:///heretic_study.db \
  --study-name qwen32b \
  --auto-select true
```

## Architecture
- Key directories:
  - `src/heretic/` - Core abliteration logic
  - `src/heretic/vast.py` - Vast.ai cloud CLI
  - `experiments/` - Behavioral direction experiments
  - `models/` - Local modified models (gitignored)
  - `chat_history/` - Saved chat sessions (gitignored)
- Data flow: Load model -> Extract behavioral directions -> Optuna optimization -> Save/upload

## Key Components
- `heretic` - Main CLI for abliteration
- `heretic-vast` - Vast.ai GPU cloud management CLI (Rich dashboard, Fabric SSH)
- `chat_app.py` - Gradio chat interface for testing models
- `experiments/verbosity/` - Spike experiment for verbosity direction extraction

## Conventions
- Formatting/linting: `ruff format .` and `ruff check --extend-select I .`
- Patterns to follow: Use existing Model/Evaluator/Settings classes
- Things to avoid:
  - No emojis or unicode symbols in output
  - No `torch_dtype` (use `dtype`)
  - No type casting to `any` - keep proper types
  - No print statements - use `logging` module instead
- Type hints: Use comprehensive type hints for all functions/methods
- Error handling: Use custom exception classes with descriptive messages
- CSS: Use Gradio CSS variables (e.g., `--body-text-color`) for theme compatibility

## Key Files
- `chat_app.py` - Gradio chat interface for modified models
  - Uses `ModelManager` class for model loading/caching
  - Custom exceptions: `HereticError`, `ModelNotFoundError`, `CUDAOutOfMemoryError`, etc.
  - Logger: `heretic_chat`
- `src/heretic/vast.py` - Vast.ai CLI
  - `VastConfig`: Configuration from env/.env
  - `GPU_TIERS`: Presets for RTX_4090, A6000, A100_40GB, A100_80GB, H100
  - Uses Fabric for SSH, Rich for terminal UI
- `runpod.ps1` - PowerShell automation for RunPod deployment
- `config.default.toml` - Default configuration template

## Current Experiments
- `experiments/verbosity/` - Test extraction of "verbosity direction"
  - Uses naturally verbose vs concise prompts (not explicit instructions)
  - Goal: Validate that behavioral directions beyond refusal can be extracted
  - **STATUS: COMPLETE** - Spike experiment finished successfully
  - Run with: `python experiments/verbosity/load_local_dataset.py` then `heretic --model MODEL --auto-select true`

### Verbosity Spike Results (Qwen 7B)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Trials | 50 Optuna optimization trials |
| Datasets | 200 concise + 200 verbose prompts |
| Model Saved | `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB) |

**Test Results (RTX 4070, 4-bit quantization):**

| Prompt Type | Example | Words |
|-------------|---------|-------|
| Factual | "What is the capital of France?" | 6 |
| Factual | "What is 24 multiplied by 7?" | 8 |
| Factual | "How many continents are there?" | 30 |
| Open-ended | "What do you think about AI?" | 203 |
| Open-ended | "Should I learn Python or JavaScript?" | 195 |

**Conclusions:**
- ✅ **Factual questions**: Concise responses (6-30 words)
- ⚠️ **Open-ended questions**: Still verbose (195-203 words)
- The ablation appears to reduce verbosity for factual Q&A but doesn't affect elaboration on open-ended topics
- This is expected: open-ended questions naturally invite longer responses

**Analysis of Results:**

The spike extracted a **"padding direction"** not an **"elaboration direction"**:
- We removed unnecessary filler on simple questions ✅
- We preserved appropriate depth on complex questions ✅

**Why the prompts were flawed:**
- Concise prompts = simple factual questions
- Verbose prompts = complex open-ended questions
- This conflates **question complexity** with **verbosity behavior**

**Better approach for v2:**
- Use SAME questions with different verbosity instructions
- Example: "What is 2+2?" vs "What is 2+2? Explain in detail."
- This isolates the padding behavior from question complexity

See `experiments/verbosity_v2/` for improved prompts.

## Active Experiments

| Experiment | Status | Goal | Location |
|------------|--------|------|----------|
| **verbosity** | ✅ Complete | Extract padding direction | `experiments/verbosity/` |
| **verbosity_v2** | ✅ Complete | Isolate padding from complexity | `experiments/verbosity_v2/` |
| **hedging** | 🔄 Ready | Extract hedging markers | `experiments/hedging/` |
| **sycophancy** | 📋 Planned | Extract excessive praise | Future |
| **meta-commentary** | 📋 Planned | Extract "Let me think..." | Future |

### Quick Start for Experiments

```bash
# Hedging (recommended next)
cd experiments/hedging
python load_local_dataset.py  
cp config.hedging.toml ../../config.toml
heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true
```

### Verbosity v2 Results (Qwen 7B)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Trials | 200 Optuna optimization trials |
| Datasets | 70 concise + 70 verbose prompts (same questions, different instructions) |
| Instance | Vast.ai RTX 4090 (24GB) |
| Runtime | ~40 minutes |
| Model Saved | `./experiments/verbosity_v2/Qwen2.5-7B-Instruct-heretic/` (15 GB) |

**Approach:** Used identical questions with explicit verbosity instructions:
- Concise: "What is 2+2? Answer briefly."
- Verbose: "What is 2+2? Explain in detail."

This isolates the padding behavior from question complexity (unlike v1 which conflated them).

## Chat App Patterns
- Model validation: Check for config.json, model weights, and tokenizer files
- GPU monitoring: Use `torch.cuda.memory_reserved()` for accurate usage
- Tokenization: Use `apply_chat_template(tokenize=True, return_tensors="pt")` for cross-model compatibility
- Message validation: Ensure all message content is string (not None) for Qwen compatibility

## Cloud Deployment
- **Vast.ai** (preferred for large models): Use `heretic-vast` CLI
- **RunPod**: Use `runpod.ps1` PowerShell script
- Model size guidelines:
  - 7B-8B: 1x RTX_4090 (24GB)
  - 14B-32B: 1x A100_40GB/80GB
  - 70B+: 2x A100_80GB or H100

### Downloading Models from Vast.ai

**THREE OPTIONS - USE WHAT WORKS:**

1. **Direct rsync from WSL (MOST RELIABLE)**:
   ```bash
   # From WSL Ubuntu terminal:
   cd /mnt/c/Development/Projects/heretic
   mkdir -p ./experiments/verbosity_v2/Qwen2.5-7B-Instruct-heretic
   rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p PORT" root@ssh1.vast.ai:/workspace/MODEL/ ./LOCAL_PATH/
   ```
   - Uses WSL's SSH key (copy from Windows if needed: `cp /mnt/c/Users/USERNAME/.ssh/id_rsa ~/.ssh/`)
   - Shows progress, resumes on disconnect
   - ~15GB model takes ~35 min at 7MB/s

2. **PowerShell script**: `.\runpod.ps1 vast-download-model`
   - More robust SSH handling with multiple URL format support
   - Better progress display with rsync
   - Works reliably on Windows

3. **Python CLI**: `heretic-vast download`
   - May have SSH connection issues on Windows/WSL
   - If it fails, use rsync or PowerShell script instead

**SSH Key Issues from WSL:**
WSL has its own SSH keys separate from Windows. If you get "Permission denied (publickey)":
```bash
# Copy Windows SSH key to WSL
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/.ssh/id_rsa ~/.ssh/
chmod 600 ~/.ssh/id_rsa
```

## CRITICAL GOTCHAS (Lessons Learned)

### MANDATORY PRE-ACTION CHECKLIST

**STOP. Before taking ANY action, complete this checklist:**

```
□ 1. Have I read knowledge.md COMPLETELY? (Not skimmed - READ)
□ 2. Have I read WORKFLOW.md? (Contains heretic-vast commands)
□ 3. What does the user ACTUALLY want? (Write it down)
□ 4. What do I ALREADY have? (Check conversation, local files, downloaded models)
□ 5. Do existing tools handle this? (heretic-vast CLI, runpod.ps1 script)
□ 6. Can local hardware do this? (RTX 4070, 8GB VRAM - use 4-bit quantization)
□ 7. Is cloud NECESSARY? (Default answer: NO)
□ 8. What's the SIMPLEST approach?
```

### CLOUD TASK WORKFLOW (MANDATORY)

**When user asks ANYTHING about cloud/Vast.ai/experiments:**

1. `heretic-vast list` - Check existing instances
2. `heretic-vast status ID` - Check if experiment running
3. `heretic-vast progress ID` - Check experiment progress
4. **ONLY THEN** decide what to do

**NEVER:**
- Use raw SSH commands when `heretic-vast` exists
- Destroy/stop instances without explicit user permission
- Start new instances when one is already running
- Assume - always CHECK first

**ANTI-PATTERNS TO RECOGNIZE:**

| Trigger | WRONG Response | RIGHT Response |
|---------|----------------|----------------|
| "Run experiment" | Manual SSH commands | Use `heretic-vast run MODEL` |
| "Check progress" | SSH and grep logs | Use `heretic-vast progress ID` |
| "Test model" | Start Vast.ai | Check if results/model already exist locally |
| "Need GPU" | Rent cloud GPU | Check if local 4070 can handle it (4-bit) |
| "Error occurred" | Try workaround | Fix the actual error |
| "Compare models" | Load both at once | Sequential loading with memory clearing |
| User gives instruction | Treat as suggestion | Treat as HARD CONSTRAINT |

### ROOT CAUSE OF REPEATED FAILURES

1. **Pattern matching instead of thinking** - Seeing "test model" and jumping to "start Vast.ai" without checking what exists
2. **Not reading documentation** - Creating docs then ignoring them
3. **Prioritizing action over understanding** - Looking productive vs being effective
4. **Treating user instructions as suggestions** - They are constraints, not options

### Local Hardware
- **User has RTX 4070 with 8GB VRAM** - Use this for testing when possible
- **7B models need 4-bit quantization** to fit in 8GB (use BitsAndBytesConfig)
- **Cannot load two 7B models simultaneously** - Run comparisons sequentially with memory clearing
- **PyTorch CUDA issue with uv**: `uv run` uses lockfile which may revert to CPU-only torch
  - Solution 1: Use system Python directly: `python script.py` instead of `uv run python script.py`
  - Solution 2: Force reinstall: `uv pip install --reinstall torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

### Tokenizer Issues
- **Heretic saves tokenizer_config.json with `extra_special_tokens` as a list** instead of dict
- This causes transformers to fail when loading the model
- **Fix**: Convert to empty dict: `c['extra_special_tokens'] = {}`
- Python one-liner fix:
  ```python
  import json; p='models/MODEL/tokenizer_config.json'; c=json.load(open(p,encoding='utf-8')); c['extra_special_tokens']={}; json.dump(c,open(p,'w',encoding='utf-8'),indent=2)
  ```

### Heretic CLI Flags
- **`--auto-select` REQUIRES a boolean value**: Use `--auto-select true`, NOT `--auto-select`
- **`--compile`**: Enable torch.compile() for ~1.5-2x inference speedup (longer initial compilation)
- **`--storage`**: Optuna storage URL for resume support (e.g., `sqlite:///study.db`)
- **`--study-name`**: Name for the Optuna study (default: `heretic_study`)
- **`--refusal-check-tokens`**: Tokens to generate for refusal detection (default: 30, lower = faster)
- **`--cache-weights`**: Enable/disable in-memory weight caching (default: true). **SET TO FALSE FOR 32B+ MODELS** to avoid OOM.
- Example (7B): `heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true --n-trials 20 --compile`
- Example (32B): `heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --cache-weights false --storage sqlite:///study.db`

### Dataset Loading
- **heretic uses `load_dataset()` from HuggingFace**, NOT `load_from_disk()`
- `DatasetSpecification` only supports: `dataset`, `split`, `column` - NO `data_files` field
- For local datasets, you must either:
  1. Use HuggingFace Hub datasets (e.g., `mlabonne/harmless_alpaca`)
  2. Push local datasets to HuggingFace Hub first
  3. Modify heretic source to support `load_from_disk()` for local paths

### Gated Models
- **Llama models are gated** - require HuggingFace authentication
- Use `huggingface-cli login` or set `HF_TOKEN` environment variable
- **Qwen models are NOT gated** - use these for quick testing

### Vast.ai exec Command
- Complex shell commands with quotes/escaping often fail through `heretic-vast exec`
- Use **base64 encoding** for complex scripts:
  ```bash
  heretic-vast exec 'echo BASE64_ENCODED_SCRIPT | base64 -d > script.py && python script.py'
  ```
- Single quotes in the command get parsed incorrectly - avoid nested quotes

### Instance Stop vs Process Kill
- **Stopping a Vast.ai instance KILLS running processes** - no graceful save
- If abliteration hasn't saved the model yet, you lose ALL progress
- Check `heretic-vast progress` for "Models Saved" before stopping

### PowerShell Scripts Fail from Bash/Codebuff

**Problem:** Running `.ps1` scripts from bash (like Codebuff does) often fails with syntax errors like "The string is missing the terminator".

**Root Cause:** Bash can't properly execute PowerShell scripts. The error message is misleading.

**Solution:** Use direct SSH commands instead:
```bash
# Get instance info
vastai show instances --raw | grep -E '"id":|"ssh_host":|"ssh_port"'

# Run commands directly
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'COMMAND'
```

**For Codebuff:** Always use direct `ssh` and `vastai` CLI commands instead of relying on the PowerShell wrapper scripts.

### pip install --force-reinstall Breaks Environment

**Problem:** After running `pip install --force-reinstall git+https://...`, heretic fails with "Could not import module 'Qwen2ForCausalLM'".

**Root Cause:** `--force-reinstall` upgrades PyTorch and transformers to incompatible versions.

**Solution:** Fix the environment:
```bash
pip install transformers>=4.55.2
```

**Prevention:** Never use `--force-reinstall`. Use `--upgrade` instead:
```bash
pip install --upgrade git+https://github.com/quanticsoul4772/abliteration-workflow.git
```

### SSH Authentication Failures on Vast.ai

**Problem:** `heretic-vast` commands fail with "Authentication failed" even though `vastai ssh-keys` shows the key is registered.

**Root Causes & Solutions:**

1. **SSH agent not running** - The key exists but isn't loaded:
   ```bash
   eval $(ssh-agent -s)
   ssh-add ~/.ssh/id_ed25519
   ```

2. **Key permissions wrong** - Must be 600:
   ```bash
   chmod 600 ~/.ssh/id_ed25519
   ```

3. **Key not attached to instance** - Even if registered on account:
   ```bash
   vastai attach ssh INSTANCE_ID "$(cat ~/.ssh/id_ed25519.pub)"
   ```

4. **Instance needs reboot** - After attaching SSH key:
   ```bash
   vastai reboot instance INSTANCE_ID
   # Wait 30-60 seconds for reboot
   ```

5. **Wrong instance/port** - After reboot, instance ID and port may change:
   ```bash
   uv run heretic-vast list  # Get new SSH details
   ```

6. **Fabric vs direct SSH** - If `heretic-vast` fails but direct SSH works:
   ```bash
   # Use direct SSH as fallback
   ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p PORT root@ssh1.vast.ai "command"
   ```

**Debugging SSH Issues:**
```bash
# Check if key is loaded
ssh-add -l

# Check keys registered on Vast.ai
vastai show ssh-keys

# Test direct SSH with verbose output
ssh -vvv -i ~/.ssh/id_ed25519 -p PORT root@ssh1.vast.ai "echo test" 2>&1 | tail -30

# Compare key fingerprints
ssh-keygen -lf ~/.ssh/id_ed25519.pub  # Local key
vastai show ssh-keys                   # Remote key
```

**Full Recovery Workflow (when SSH completely broken):**
```bash
# 1. Start SSH agent and add key
eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519

# 2. Verify key is on Vast.ai (add if not)
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)" 2>/dev/null || echo "Key exists"

# 3. Get instance ID
INSTANCE_ID=$(uv run heretic-vast list 2>/dev/null | grep -oP '\d{8}' | head -1)

# 4. Attach key to instance
vastai attach ssh $INSTANCE_ID "$(cat ~/.ssh/id_ed25519.pub)"

# 5. Reboot instance to pick up new key
vastai reboot instance $INSTANCE_ID

# 6. Wait and get new SSH details
sleep 45
uv run heretic-vast list

# 7. Test connection
ssh -o StrictHostKeyChecking=no -p NEW_PORT root@ssh1.vast.ai "echo SUCCESS"
```
