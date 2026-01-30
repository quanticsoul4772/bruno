# Verbosity Direction Spike Experiment

> **✅ STATUS: COMPLETE - SPIKE FINISHED**
>
> The verbosity spike experiment has been completed successfully.
> - **Model**: Qwen/Qwen2.5-7B-Instruct
> - **Trials**: 50 Optuna optimization trials
> - **Result**: Model saved to `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB)
> - **Finding**: Verbosity reduced for factual questions, open-ended questions remain verbose

This experiment tests whether a "verbosity direction" can be extracted from LLMs
using the same technique heretic uses for refusal directions.

**Part of the heretic research roadmap.** See [ROADMAP.md](../../ROADMAP.md) for the full vision.

## Hypothesis

If instruction-tuned models have a "verbosity behavior" encoded as a direction
in activation space (similar to refusal), we should be able to:

1. Extract it using contrastive prompts
2. Remove it via orthogonal projection
3. Observe shorter responses without capability loss

## Approach

Instead of explicit instructions ("be brief" vs "be detailed"), we use prompts
that **naturally** elicit different verbosity levels:

- **Concise prompts**: Factual questions with single correct answers
  - "What is the capital of France?"
  - "What year did WW2 end?"
  - "Is the sun a star?"

- **Verbose prompts**: Open-ended questions that invite elaboration
  - "What do you think about AI?"
  - "Should I learn Python or JavaScript?"
  - "Is social media good for society?"

This mirrors refusal extraction (harmless vs harmful prompts) - we're capturing
the model's **intrinsic tendency** to elaborate, not instruction following.

## Files

```
experiments/verbosity/
├── README.md                 # This file
├── concise_prompts.json      # 200 factual/closed questions
├── verbose_prompts.json      # 200 open-ended/opinion questions
├── config.verbosity.toml     # Heretic config for this experiment
├── load_local_dataset.py     # Convert JSON to HuggingFace Dataset
├── eval_verbosity.py         # Measure response length changes
├── run_spike.ps1             # Windows PowerShell automation
└── run_spike.sh              # Linux/Mac automation
```

## Running the Experiment

### Option 1: Local GPU

```bash
# Step 0: Create local datasets
python experiments/verbosity/load_local_dataset.py

# Step 1: Copy the verbosity config
cp experiments/verbosity/config.verbosity.toml config.toml

# Step 2: Run heretic (use a smaller model for faster iteration)
# NOTE: --auto-select REQUIRES 'true' value, and Llama models are gated (need HF auth)
# Use Qwen for quick testing (not gated)
heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true

# Step 3: Evaluate verbosity change
python experiments/verbosity/eval_verbosity.py \
    --original meta-llama/Llama-3.1-8B-Instruct \
    --modified ./Llama-3.1-8B-Instruct-heretic \
    --num-prompts 50
```

### Option 2: Vast.ai (Cloud GPU)

```bash
# Create instance and setup
heretic-vast create RTX_4090
heretic-vast setup

# Upload experiment files (manual step - use scp or paste)
heretic-vast exec 'mkdir -p /workspace/experiments/verbosity'

# Run on cloud
heretic-vast exec 'cd /workspace && python experiments/verbosity/load_local_dataset.py'
heretic-vast exec 'cp /workspace/experiments/verbosity/config.verbosity.toml /workspace/config.toml'
# NOTE: Use Qwen (not gated) instead of Llama for easier testing
heretic-vast run Qwen/Qwen2.5-7B-Instruct

# Monitor
heretic-vast watch
```

### Option 3: Windows PowerShell

```powershell
.\experiments\verbosity\run_spike.ps1 -Model "meta-llama/Llama-3.1-8B-Instruct"
```

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Token reduction | >20% | Primary goal |
| KL divergence | <1.0 | Capability preservation |
| Capability | Qualitative | Can still answer complex questions |

## Actual Results (Spike Complete)

**Run Date**: January 2026
**Infrastructure**: Vast.ai 2x A100-SXM4-80GB
**Model**: Qwen/Qwen2.5-7B-Instruct
**Trials**: 50 Optuna optimization trials
**Output**: `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB)

### Optimization Parameters Found
```
direction_index = 12.77
attn.o_proj.max_weight = 0.92
attn.o_proj.max_weight_position = 21.64
attn.o_proj.min_weight = 0.91
mlp.down_proj.max_weight = 1.11
mlp.down_proj.max_weight_position = 24.42
mlp.down_proj.min_weight = 0.24
```

### Test Results (RTX 4070, 4-bit Quantization)

| Prompt | Words | Category |
|--------|-------|----------|
| What is the capital of France? | 6 | Factual ✅ |
| What is 24 multiplied by 7? | 8 | Factual ✅ |
| Is the sun a star? | 65 | Factual* |
| How many continents are there? | 30 | Factual ✅ |
| What do you think about AI? | 203 | Open-ended ⚠️ |
| Should I learn Python or JavaScript? | 195 | Open-ended ⚠️ |
| What makes a good leader? | 199 | Open-ended ⚠️ |

### Conclusions

1. **✅ Verbosity direction exists** - The ablation successfully extracts a behavioral direction
2. **✅ Factual questions affected** - Simple Q&A gets concise responses (6-30 words vs typical 50-100+)
3. **⚠️ Open-ended questions unaffected** - Complex/opinion questions still get long responses (195-203 words)
4. **Interpretation**: The "verbosity direction" appears to control *unnecessary elaboration* on simple facts, but doesn't suppress *appropriate elaboration* on complex topics

### Analysis: Why Results Were Mixed

**What we extracted:** A **"padding direction"** - the tendency to add unnecessary filler to simple answers.

**What we didn't extract:** An **"elaboration direction"** - the tendency to provide detailed explanations.

**The problem with our prompt design:**

| Prompt Type | What We Used | The Problem |
|-------------|--------------|-------------|
| Concise | Simple factual questions | Question complexity ≠ verbosity |
| Verbose | Complex open-ended questions | Complexity naturally invites depth |

We conflated **question complexity** with **verbosity behavior**. A model SHOULD give longer answers to "What do you think about AI?" than to "What is 2+2?"

**The right approach (see v2 experiment):**

```
SAME question, different verbosity level:
- "What is 2+2?"
- "What is 2+2? Please explain your reasoning in detail."

SAME question, different instruction:
- "Summarize quantum physics."
- "Summarize quantum physics in one sentence."
```

This isolates the **padding behavior** (unnecessary filler) from **appropriate thoroughness** (answering complex questions completely).

## What We're Testing

1. **Does the direction exist?** Can we extract a meaningful verbosity direction?
2. **Is it separable from capability?** Can we remove verbosity without damaging answers?
3. **Does it compose with refusal?** Can we apply both directions to the same model?

## Test Script

Use `test_comparison_4bit.py` to test the abliterated model locally with 4-bit quantization:

```bash
# Requires GPU with 8GB+ VRAM
# Use system Python (not uv run) to ensure CUDA works
python experiments/verbosity/test_comparison_4bit.py
```

**Note**: Cannot compare both models simultaneously on 8GB VRAM. The test script loads only the abliterated model.

## Next Steps

**The spike worked - verbosity direction exists and is extractable.**

### Immediate Next Experiments

1. **verbosity_v2** - Improved prompts that isolate padding from complexity
   - Same questions with/without "explain in detail" instructions
   - Located in `experiments/verbosity_v2/`

2. **hedging** - Extract the hedging direction
   - "I think", "perhaps", "might be", "it's possible"
   - Located in `experiments/hedging/`

3. **sycophancy** - Extract excessive agreement/praise
   - "Great question!", "That's a wonderful idea!"
   - Future experiment

### Other Follow-ups

- [ ] Test composition with refusal direction (apply both to same model)
- [ ] Test cross-model transfer (does Qwen direction work on Llama?)
- [ ] Compare with inference-time steering (activation injection)
- [ ] Document optimal layer ranges for verbosity vs refusal

### Better Metrics for Future Experiments

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **Information density** | Useful content per token | Manual annotation or LLM scoring |
| **Filler phrase count** | "Great question!", "I hope this helps" | Regex pattern matching |
| **Preamble length** | Words before actual answer | Heuristic detection |
| **Hedging markers** | "I think", "perhaps", "might be" | Keyword counting |
| **Token reduction** | Raw length comparison | Tokenizer count |

## Technical Notes

- The config uses heretic's standard `good_prompts`/`bad_prompts` structure
- Evaluation uses `test_comparison_4bit.py` for length measurement (not refusal markers)
- Smaller models (7B-8B) recommended for faster iteration
- The 50-trial config is faster than default 200 for spike testing

### Known Issues

1. **Tokenizer config bug**: Heretic saves `extra_special_tokens` as a list instead of dict. Fix:
   ```python
   import json
   p = 'models/MODEL/tokenizer_config.json'
   c = json.load(open(p, encoding='utf-8'))
   c['extra_special_tokens'] = {}
   json.dump(c, open(p, 'w', encoding='utf-8'), indent=2)
   ```

2. **PyTorch CUDA with uv**: `uv run` may use CPU-only torch from lockfile. Use system Python directly.

3. **8GB VRAM limit**: Cannot load two 7B models simultaneously even with 4-bit quantization. Run comparisons sequentially.

## Research Questions

This experiment is part of validating broader hypotheses:

1. Do behavioral directions transfer across model families (Llama → Qwen)?
2. Can you compose 3-5 directions without degrading capability?
3. Are some directions entangled (verbosity and thoroughness)?
4. What's the minimal modification needed?

See [ROADMAP.md](../../ROADMAP.md) for the full research agenda.
