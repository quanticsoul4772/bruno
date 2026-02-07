# Bruno Feature Roadmap: Competing with Top Abliterated Models

## Executive Summary

Bruno is **technically the most sophisticated abliteration tool** on HuggingFace — Optuna optimization, MPOA, sacred directions, MoE-aware abliteration, neural refusal detection — none of the top competitors have anything close. But we're losing on **packaging, presentation, and distribution**.

This document is the implementation plan to close that gap.

**Total estimated time: 2-3 weeks of focused work.**

---

## Implementation Phases

| Phase | Features | Time | Why This Order |
|-------|----------|------|----------------|
| **Phase 1** | #1 Rich model card, #2 Tags, #3 Usage examples, #7 Bruno Score | 1-2 days | Highest effort-to-impact. No new deps. All touch `utils.py` + `model_saving.py`. |
| **Phase 2** | #4 GGUF conversion, #5 Multi-quant, #12 One-command pipeline | 2-3 days | GGUF is the biggest download driver. llama.cpp already a submodule. |
| **Phase 3** | #6 Extended benchmarks, #8 Competitor comparison | 2-3 days | Quality differentiation. Extends existing validation framework. |
| **Phase 4** | #9 imatrix quantization | 2-3 days | Premium quality GGUF. Extension of Phase 2. |
| **Phase 5** | #10 AWQ/GPTQ/EXL2 formats | 3-4 days | New dependencies, GPU-specific. Lower priority. |
| **Phase 6** | #11 Post-abliteration DPO | 1 week | Most complex. Research-heavy. Do last. |

---

## Feature 1: Rich Model Card + Benchmarks

**Priority:** 🔥 1 | **Effort:** 4 hrs | **Impact:** Very High

### The Problem

`get_readme_intro()` currently generates a minimal model card with only:
- Title + Bruno version
- Abliteration parameters table
- KL divergence value
- Refusal count

Meanwhile, `AbliterationValidator` already computes MMLU scores (12 categories), refusal rates, KL divergence, and generates a full `ValidationReport` — but **none of this data reaches the model card**.

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/utils.py` | Expand `get_readme_intro()` to accept `ValidationReport` and active features |
| `src/bruno/phases/model_saving.py` | Thread `ValidationReport` through `upload_model_huggingface()` → `_upload_model_card()` |
| `src/bruno/main.py` | Pass validator report to save/upload calls (~lines 1516, 1619, 1683) |
| `tests/unit/test_utils.py` | Update `TestGetReadmeIntro` tests |

### Implementation Steps

**Step 1: Expand `get_readme_intro()` signature**

```python
def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
    validation_report: ValidationReport | None = None,
    features_active: list[str] | None = None,
    repo_id: str | None = None,
) -> str:
```

**Step 2: Add rich content sections**

The expanded template should include:

1. **Title + Bruno badge** — "Made with Bruno v{version} (Optuna-optimized abliteration)"
2. **Technique description** — Pull from `features_active` list (from `feature_tracker.get_summary()["active"]`) to describe what Bruno did:
   - "This model was abliterated using Optuna-optimized multi-phase direction extraction with MPOA (Norm-Preserving Biprojected Abliteration), sacred direction preservation, and neural refusal detection."
3. **Benchmark table** — If `validation_report` has MMLU scores:
   ```markdown
   ## Benchmarks

   | Benchmark | This Model | Original Model | Delta |
   | :-------- | :--------: | :------------: | :---: |
   | **MMLU (average)** | 48.7% | 48.9% | -0.4% |
   | abstract_algebra | 35.0% | 35.0% | 0.0% |
   | formal_logic | 42.0% | 43.0% | -1.0% |
   ...
   ```
4. **Performance summary** — Existing KL divergence + refusals table, plus capability_preserved badge
5. **Abliteration parameters** — Existing parameters table
6. **Disclaimer** — Standard uncensored model warning

**Step 3: Wire data through model_saving.py**

Add `validation_report` parameter to `upload_model_huggingface()` and `_upload_model_card()`:

```python
def upload_model_huggingface(
    model: Model,
    settings: Settings,
    trial: Trial,
    evaluator: Evaluator,
    repo_id: str | None = None,
    private: bool | None = None,
    token: str | None = None,
    validation_report: ValidationReport | None = None,  # NEW
) -> bool:
```

**Step 4: Wire from main.py**

The `validator` object already exists in `run()`. After `validator.measure_post_abliteration()`, the report is generated (~line 1458). Pass it through:

```python
# In auto-select path (~line 1516):
upload_model_huggingface(
    model, settings, trial, evaluator,
    validation_report=validator.get_report() if validator.post_abliteration else None,
)

# Get active features for model card:
features_active = [name for name in feature_tracker.get_summary()["active"]]
```

### New Config Settings

None — all data sources already exist.

### Dependencies

None — uses existing `ValidationReport` dataclass.

---

## Feature 2: Better HuggingFace Tags

**Priority:** 🔥 2 | **Effort:** 1 hr | **Impact:** Medium

### The Problem

Tags are hardcoded to just `["heretic", "uncensored", "decensored", "abliterated"]`. Missing: architecture tags, `text-generation`, `conversational`, `base_model`, `pipeline_tag`, `language`, `library_name`.

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/phases/model_saving.py` | Expand `_upload_model_card()` with dynamic tags + YAML metadata |
| `src/bruno/main.py` | Update duplicate tag code at ~lines 1677-1681 (interactive upload path) |

### Implementation Steps

**Step 1: Add helper function for dynamic tags**

```python
def _get_model_tags(model_name: str, is_moe: bool = False) -> list[str]:
    tags = ["heretic", "uncensored", "decensored", "abliterated", "text-generation"]
    name_lower = model_name.lower()

    # Architecture tags
    if "qwen" in name_lower:
        tags.append("qwen2")
    if "llama" in name_lower:
        tags.append("llama")
    if "deepseek" in name_lower:
        tags.append("deepseek")
    if "gemma" in name_lower:
        tags.append("gemma")
    if "phi" in name_lower:
        tags.append("phi")
    if "mistral" in name_lower:
        tags.append("mistral")

    # Capability tags
    if "coder" in name_lower or "code" in name_lower:
        tags.append("code")
    if "instruct" in name_lower or "chat" in name_lower:
        tags.append("conversational")
    if is_moe or "moe" in name_lower or "moonlight" in name_lower:
        tags.append("moe")

    # Bruno-specific
    tags.append("optuna-optimized")
    return tags
```

**Step 2: Set YAML frontmatter metadata**

```python
card.data.base_model = settings.model
card.data.pipeline_tag = "text-generation"
card.data.language = ["en"]
card.data.library_name = "transformers"
card.data.tags = _get_model_tags(settings.model, model.is_moe_model())
```

**Step 3: Refactor duplicate tag code**

The interactive upload path in `main.py` (~lines 1677-1681) duplicates tag logic. Refactor to call `_upload_model_card()` from `model_saving.py` instead.

### New Config Settings

None.

---

## Feature 3: Usage Examples in Model Card

**Priority:** 🔥 3 | **Effort:** 2 hrs | **Impact:** Medium

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/utils.py` | Add `_get_usage_examples()` helper, include in `get_readme_intro()` |

### Implementation Steps

**Step 1: Create usage example generator**

```python
def _get_usage_examples(repo_id: str, model_name: str) -> str:
    is_chat = any(kw in model_name.lower() for kw in ["instruct", "chat"])

    sections = []

    # Transformers example
    if is_chat:
        sections.append(f'''### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Hello, how are you?"}},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```''')
    else:
        sections.append(f'''### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```''')

    # vLLM example
    sections.append(f'''### Using vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_id}")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```''')

    return "\n\n".join(sections)
```

**Step 2: Include in `get_readme_intro()`**

Add the usage section after the benchmark table but before the parameters table.

### New Config Settings

None.

---

## Feature 4: Integrated GGUF Conversion + Upload

**Priority:** 🔥 4 | **Effort:** 1-2 days | **Impact:** Very High

### The Problem

GGUF models get **10-100x more downloads** than safetensors-only models. Bruno has `convert-to-gguf.ps1` and llama.cpp as a submodule, but they're not integrated into the pipeline.

### New Files

| File | Purpose |
|------|---------|
| `src/bruno/phases/gguf_conversion.py` | GGUF conversion + quantization + upload logic |

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/config.py` | Add GGUF settings |
| `src/bruno/phases/__init__.py` | Export new functions |
| `src/bruno/main.py` | Integrate GGUF step after model save |

### New Config Settings

```python
enable_gguf_conversion: bool = Field(
    default=False,
    description="Convert saved model to GGUF format after saving.",
)
gguf_quantizations: list[str] = Field(
    default=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    description="GGUF quantization types to generate.",
)
gguf_output_dir: str = Field(
    default="./models/gguf",
    description="Directory to store GGUF files.",
)
gguf_upload: bool = Field(
    default=False,
    description="Upload GGUF files to a companion HuggingFace repo (adds -GGUF suffix).",
)
gguf_imatrix: bool = Field(
    default=False,
    description="Use importance matrix for higher quality quantization.",
)
gguf_imatrix_dataset: str = Field(
    default="",
    description="Path to calibration text file for imatrix generation.",
)
gguf_cleanup_f16: bool = Field(
    default=True,
    description="Delete intermediate F16 GGUF file after quantization to save disk space.",
)
```

### Implementation Steps

**Step 1: Create `gguf_conversion.py`**

Key functions:

```python
import subprocess
from pathlib import Path

LLAMA_CPP_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

def find_quantize_binary() -> Path | None:
    """Find llama-quantize binary in known locations."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",  # Windows
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",  # Windows alt
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",  # Linux
        LLAMA_CPP_DIR / "llama-quantize",  # Root
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

def convert_to_gguf_f16(model_path: str, output_path: str) -> str:
    """Convert HuggingFace model to F16 GGUF using convert_hf_to_gguf.py."""
    cmd = ["python", str(CONVERT_SCRIPT), model_path, "--outfile", output_path, "--outtype", "f16"]
    subprocess.run(cmd, check=True)
    return output_path

def quantize_gguf(input_gguf: str, output_gguf: str, quant_type: str, imatrix: str | None = None) -> str:
    """Quantize F16 GGUF to specified quantization type."""
    quantize_bin = find_quantize_binary()
    if not quantize_bin:
        raise FileNotFoundError("llama-quantize binary not found. Build llama.cpp first.")
    cmd = [str(quantize_bin)]
    if imatrix:
        cmd.extend(["--imatrix", imatrix])
    cmd.extend([input_gguf, output_gguf, quant_type])
    subprocess.run(cmd, check=True)
    return output_gguf

def convert_and_quantize(model_path: str, settings: Settings) -> list[Path]:
    """Full pipeline: HF → F16 GGUF → multiple quantized GGUFs."""
    ...

def upload_gguf_files(gguf_files: list[Path], base_repo_id: str, token: str, model_card_text: str) -> bool:
    """Upload GGUF files to {base_repo_id}-GGUF companion repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{base_repo_id}-GGUF"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    for gguf_file in gguf_files:
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
            token=token,
        )
    # Upload GGUF-specific README
    ...
    return True
```

**Step 2: GGUF file naming convention**

```
{ModelName}-{QuantType}.gguf
Example: Moonlight-16B-A3B-Instruct-bruno-Q4_K_M.gguf
```

**Step 3: GGUF repo model card**

Generate a separate README for the GGUF repo with:
- Quant table (type, file size, description)
- Download links
- Usage with llama.cpp, Ollama, and llama-server
- Link back to the main safetensors repo

**Step 4: Integration in main.py**

After `save_model_local()` succeeds and `settings.enable_gguf_conversion`:
```python
if settings.enable_gguf_conversion:
    gguf_files = convert_and_quantize(save_directory, settings)
    if settings.gguf_upload and settings.hf_upload:
        upload_gguf_files(gguf_files, settings.hf_upload, token, gguf_readme)
```

### Dependencies

- `subprocess` (stdlib) for calling llama.cpp tools
- llama.cpp submodule must be present
- Quantize binary must be built (provide clear error + build instructions if missing)

---

## Feature 5: Multi-quant GGUF (Q4/Q5/Q6/Q8)

**Priority:** 🔥 5 | **Effort:** 1 day | **Impact:** Very High

This is an extension of Feature 4 — the `gguf_quantizations` config list already handles multiple types.

### Additional Work Beyond Feature 4

1. **Parallel quantization** — Each quant is CPU-bound and independent:
   ```python
   from concurrent.futures import ProcessPoolExecutor
   with ProcessPoolExecutor(max_workers=2) as executor:
       futures = {
           executor.submit(quantize_gguf, f16_path, out_path, qt): qt
           for qt, out_path in quant_jobs.items()
       }
   ```

2. **Quant info table** for the GGUF repo README:
   ```markdown
   | Quantization | File Size | Bits/Weight | Quality | Use Case |
   | :----------- | :-------: | :---------: | :-----: | :------- |
   | Q4_K_M | 4.5 GB | 4.83 | ★★★☆ | Good balance, recommended for most users |
   | Q5_K_M | 5.2 GB | 5.69 | ★★★★ | Higher quality, moderate VRAM |
   | Q6_K | 6.1 GB | 6.56 | ★★★★★ | High quality |
   | Q8_0 | 8.0 GB | 8.50 | ★★★★★ | Near-lossless |
   ```

3. **Size estimation** — After each quant, record file size and include in model card.

4. **Auto-cleanup** — `gguf_cleanup_f16` config to delete F16 intermediate.

---

## Feature 6: Extended Benchmarks (HellaSwag, ARC, TruthfulQA, GSM8K)

**Priority:** 6 | **Effort:** 2-3 days | **Impact:** High

### New Files

| File | Purpose |
|------|---------|
| `src/bruno/benchmarks.py` | Extended benchmark evaluators (HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval) |

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/validation.py` | Add `extended_benchmarks` to `ValidationMetrics`, wire into `AbliterationValidator` |
| `src/bruno/config.py` | Add benchmark settings |

### New Config Settings

```python
run_extended_benchmarks: bool = Field(
    default=False,
    description="Run extended benchmarks during validation. Slower but provides comprehensive capability data.",
)
benchmark_suites: list[str] = Field(
    default=["hellaswag", "arc_challenge", "truthfulqa"],
    description="Which extended benchmarks to run.",
)
benchmark_samples: int = Field(
    default=100,
    description="Number of samples per benchmark.",
)
```

### Implementation Steps

**Step 1: Create evaluator classes for each benchmark**

Each evaluator follows the same pattern as `MMLUEvaluator`:

| Benchmark | Dataset | Format | Answer Extraction |
|-----------|---------|--------|-------------------|
| **HellaSwag** | `Rowan/hellaswag` | 4-way completion selection | Letter A-D |
| **ARC-Challenge** | `allenai/ai2_arc`, config `ARC-Challenge` | Multiple-choice | Letter A-D |
| **TruthfulQA** | `truthfulqa/truthful_qa`, config `multiple_choice` | Multiple-choice | Letter index |
| **GSM8K** | `openai/gsm8k` | Math word problem | Extract final number |
| **HumanEval** | `openai_humaneval` | Code generation (coder models only) | Execute + check |

**Step 2: Create `BenchmarkSuite` orchestrator**

```python
class BenchmarkSuite:
    def __init__(self, model: Model, settings: Settings):
        self.evaluators = []
        for name in settings.benchmark_suites:
            if name == "hellaswag":
                self.evaluators.append(HellaSwagEvaluator(model, settings.benchmark_samples))
            elif name == "arc_challenge":
                self.evaluators.append(ARCEvaluator(model, settings.benchmark_samples))
            ...

    def run_all(self) -> dict[str, float]:
        results = {}
        for evaluator in self.evaluators:
            results[evaluator.name] = evaluator.evaluate()
        return results
```

**Step 3: Wire into `ValidationMetrics`**

Add `extended_benchmarks: dict[str, float]` field. The model card generator (Feature 1) picks these up automatically.

---

## Feature 7: "Bruno Score" Composite Metric

**Priority:** 7 | **Effort:** 2 hrs | **Impact:** Medium

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/validation.py` | Add `compute_bruno_score()` to `ValidationReport` |
| `src/bruno/utils.py` | Display in model card |

### Formula

```python
def compute_bruno_score(self) -> float:
    """Compute Bruno quality score (0-100).

    Formula: capability_preservation * refusal_removal_rate * 100

    - capability_preservation: 1.0 minus penalties for KL divergence and MMLU degradation
    - refusal_removal_rate: fraction of refusals successfully removed
    """
    if self.post_abliteration is None or self.baseline is None:
        return 0.0

    # Capability preservation (0-1)
    kl_penalty = min(self.kl_divergence_increase / 5.0, 0.5)
    mmlu_penalty = max(-self.mmlu_change, 0.0)
    capability = max(1.0 - kl_penalty - mmlu_penalty, 0.0)

    # Refusal removal (0-1)
    if self.baseline.refusal_rate > 0:
        removal = self.refusal_reduction / self.baseline.refusal_rate
    else:
        removal = 1.0

    return round(capability * removal * 100, 1)
```

### Display in Model Card

```markdown
## Bruno Score: 96.6 / 100

> 99.6% capability preserved | 97.0% refusals removed
```

---

## Feature 8: Competitor Comparison Tables

**Priority:** 8 | **Effort:** 1 day | **Impact:** High

### Approach

**Strategy B (Database-driven) recommended first**, then upgrade to Strategy A (automated) later.

### New Files

| File | Purpose |
|------|---------|
| `src/bruno/data/competitor_benchmarks.json` | Database of known competitor benchmark results |
| `src/bruno/competitor.py` | Comparison table generator |

### Implementation Steps

**Step 1: Create benchmark database**

```json
{
  "Qwen/Qwen2.5-Coder-7B-Instruct": {
    "original": {"mmlu": 48.9, "hellaswag": 72.1},
    "huihui-ai": {"mmlu": 46.2, "hellaswag": 70.5, "repo": "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated"},
    "failspy": {"mmlu": 45.8, "repo": "failspy/Qwen2.5-Coder-7B-Instruct-abliterated"}
  },
  "moonshotai/Moonlight-16B-A3B-Instruct": {
    "original": {"mmlu": 48.9},
    "huihui-ai": {"mmlu": 46.0, "repo": "huihui-ai/Moonlight-16B-A3B-Instruct-abliterated"}
  }
}
```

**Step 2: Lookup and generate comparison table**

```python
def get_competitor_comparison(base_model: str, our_scores: dict) -> str | None:
    """Generate markdown comparison table if competitor data exists."""
    # Load database, look up base_model, generate table
    ...
```

**Step 3: Include in model card** (conditional — only if competitor data exists for this base model)

```markdown
## Comparison with Other Abliterated Versions

| Metric | Bruno (this model) | huihui-ai | Original |
| :----- | :----------------: | :-------: | :------: |
| **MMLU** | 48.7% (-0.4%) | 46.2% (-5.5%) | 48.9% |
| **Refusals removed** | 97% | 95% | 0% |
```

### New Config Settings

```python
include_competitor_comparison: bool = Field(
    default=True,
    description="Include competitor comparison table in model card if data available.",
)
```

---

## Feature 9: imatrix Quantization

**Priority:** 9 | **Effort:** 1 day | **Impact:** High

Extension of Feature 4. Uses the `gguf_imatrix` config flag.

### Implementation Steps

1. **Calibration data** — Default to downloading `wikitext-2-raw-v1` test split as calibration text
2. **imatrix generation**:
   ```python
   def generate_imatrix(f16_gguf: str, calibration_file: str, output: str) -> str:
       imatrix_bin = find_imatrix_binary()  # Similar to find_quantize_binary()
       cmd = [str(imatrix_bin), "-m", f16_gguf, "-f", calibration_file,
              "--chunk", "512", "-o", output, "-ngl", "99"]
       subprocess.run(cmd, check=True)
       return output
   ```
3. **Pass imatrix to quantize**: `--imatrix imatrix.dat` flag
4. **Naming**: `Model-Q4_K_M-imat.gguf`
5. **Requires**: llama.cpp binaries built (llama-imatrix + llama-quantize)

---

## Feature 10: AWQ/GPTQ/EXL2 Formats

**Priority:** 10 | **Effort:** 2 days | **Impact:** Medium

### New Files

| File | Purpose |
|------|---------|
| `src/bruno/phases/gpu_quantization.py` | GPU-accelerated quantization (AWQ, GPTQ, EXL2) |

### New pip Dependencies (optional)

- `autoawq` — for AWQ quantization
- `auto-gptq` — for GPTQ quantization
- `exllamav2` — for EXL2 quantization

### Implementation Steps

**AWQ:**
```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained(model_path)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized(output_path)
```

**GPTQ:**
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
model.quantize(calibration_data)
model.save_quantized(output_path)
```

Upload each format to separate repos: `{repo_id}-AWQ`, `{repo_id}-GPTQ`.

### New Config Settings

```python
enable_awq: bool = Field(default=False)
enable_gptq: bool = Field(default=False)
enable_exl2: bool = Field(default=False)
awq_bits: int = Field(default=4)
gptq_bits: int = Field(default=4)
```

**Note:** GPU-intensive. Should be optional with clear VRAM requirements documented.

---

## Feature 11: Post-Abliteration DPO

**Priority:** 11 | **Effort:** 1 week | **Impact:** Medium

### The Concept

After abliteration, fine-tune with DPO (Direct Preference Optimization) to recover lost capability while maintaining uncensored behavior. This is what mlabonne does to differentiate their models.

### New Files

| File | Purpose |
|------|---------|
| `src/bruno/phases/dpo_recovery.py` | DPO training loop using TRL |

### New pip Dependencies

- `trl` — Transformer Reinforcement Learning library

### Implementation Steps

1. **Generate DPO dataset** from abliterated model:
   - Preferred: helpful, complete responses
   - Rejected: refusing, evasive, or incomplete responses
2. **DPO training** using TRL's `DPOTrainer`:
   ```python
   from trl import DPOConfig, DPOTrainer
   training_args = DPOConfig(
       output_dir="./dpo_output",
       num_train_epochs=1,
       per_device_train_batch_size=1,
       learning_rate=5e-7,
       beta=0.1,
   )
   trainer = DPOTrainer(model=model, args=training_args, ...)
   trainer.train()
   ```
3. **Integration**: After abliteration, before save, optionally run DPO

### New Config Settings

```python
enable_dpo_recovery: bool = Field(default=False)
dpo_learning_rate: float = Field(default=5e-7)
dpo_epochs: int = Field(default=1)
dpo_beta: float = Field(default=0.1)
dpo_dataset: str = Field(default="")
```

**Note:** Most complex feature. Requires significant VRAM. Recommended as last implementation phase.

---

## Feature 12: One-Command Full Pipeline

**Priority:** 12 | **Effort:** 1 day | **Impact:** Medium

### Files to Modify

| File | Change |
|------|--------|
| `src/bruno/config.py` | Add pipeline preset flags |
| `src/bruno/main.py` | Wire expanded flags |

### New Config Settings

```python
full_pipeline: bool = Field(
    default=False,
    description="Run full pipeline: abliterate -> benchmark -> GGUF -> upload all.",
)
gguf: bool = Field(
    default=False,
    description="Shorthand for --enable-gguf-conversion --gguf-upload.",
)
benchmark_full: bool = Field(
    default=False,
    description="Shorthand for --run-extended-benchmarks.",
)
```

### Implementation

Add a Pydantic model validator to expand shorthand flags:

```python
@model_validator(mode="after")
def expand_pipeline_flags(self) -> "Settings":
    if self.full_pipeline:
        self.auto_select = True
        self.enable_gguf_conversion = True
        self.gguf_upload = True
        self.run_extended_benchmarks = True
    if self.gguf:
        self.enable_gguf_conversion = True
        self.gguf_upload = True
    if self.benchmark_full:
        self.run_extended_benchmarks = True
    return self
```

### Usage

```bash
# Full pipeline: abliterate + benchmark + GGUF + upload
bruno Qwen/Qwen2.5-7B-Instruct --full-pipeline --hf-upload rawcell/Qwen2.5-7B-Instruct-bruno

# Just add GGUF conversion
bruno Qwen/Qwen2.5-7B-Instruct --gguf --hf-upload rawcell/Qwen2.5-7B-Instruct-bruno

# Just extended benchmarks
bruno Qwen/Qwen2.5-7B-Instruct --benchmark-full
```

---

## Appendix: Key Insight — What Makes Top Models Popular

Based on research of the top abliterated models on HuggingFace:

| Factor | Impact on Likes/Downloads | Bruno Status |
|--------|--------------------------|--------------|
| **GGUF availability** | 10-100x more downloads | ❌ Not integrated |
| **Rich model card with benchmarks** | 3-5x more likes | ❌ Minimal card |
| **Speed to market** (new models) | First-mover gets 80% of likes | ⚠️ Could be faster |
| **Before/after benchmark tables** | Proves quality, converts browsers | ❌ Data exists but not shown |
| **Multiple quantization formats** | Serves all hardware tiers | ❌ Only safetensors |
| **Usage examples** | Reduces friction to adoption | ❌ Not included |
| **Architecture-specific tags** | Better discoverability | ❌ Generic tags only |
| **Competitor comparison** | Direct quality proof | ❌ Not included |
| **Composite quality score** | Quick decision metric | ❌ Not computed |

Bruno's technical superiority is **invisible** to users browsing HuggingFace. This roadmap makes it visible.
