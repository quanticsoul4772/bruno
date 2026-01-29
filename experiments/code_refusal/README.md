# Code Refusal Abliteration Experiment

This experiment removes the model's tendency to refuse coding requests for:
- Web scraping tools
- Security/penetration testing tools
- Automation scripts
- Network analysis tools
- Other "grey area" code that models typically refuse

## Why This Works

Unlike verbosity (which is a continuous spectrum), **refusal is binary** - the model either writes the code or refuses. This makes it ideal for abliteration, which works by:

1. Computing the "refusal direction" = difference between model activations on refused vs complied prompts
2. Projecting that direction out of the model weights
3. Result: model no longer activates the "refuse" behavior

## Files

- `config.code_refusal.toml` - Experiment configuration
- `good_code_prompts.json` - Harmless coding requests (model complies)
- `bad_code_prompts.json` - Coding requests that trigger refusals
- `test_setup.py` - Verify setup before running

## Quick Start

### 1. Create the Datasets

First, convert the JSON prompts to HuggingFace dataset format:

```bash
python experiments/code_refusal/create_datasets.py
```

This creates:
- `experiments/code_refusal/good_code_dataset/` (train/test splits)
- `experiments/code_refusal/bad_code_dataset/` (train/test splits)

### 2. Test the Setup (Optional)

```bash
# Test without loading model (just check files)
python experiments/code_refusal/test_setup.py --skip-model

# Test with model (checks baseline refusal rate)
python experiments/code_refusal/test_setup.py --model-path ./models/Qwen2.5-7B-Instruct
```

### 3. Run the Abliteration

```bash
python -m heretic ./models/Qwen2.5-7B-Instruct \
  -c experiments/code_refusal/config.code_refusal.toml \
  -o experiments/code_refusal/
```

### 3. Test the Result

After abliteration, compare the original and abliterated models:

```bash
# The abliterated model will be saved to:
# experiments/code_refusal/Qwen2.5-7B-Instruct-heretic/
```

## Expected Results

- **Before**: Model refuses 70-90% of "bad" code prompts
- **After**: Model writes the requested code without refusing
- **KL Divergence**: Should be < 1.0 (model capabilities preserved)

## Refusal Markers

The experiment detects refusals using these patterns:

| Category | Examples |
|----------|----------|
| Direct refusals | "I can't help", "I cannot assist", "I won't" |
| AI self-reference | "as an AI", "as a language model" |
| Safety concerns | "security concerns", "potentially harmful", "malicious" |
| Moralizing | "I must emphasize", "responsible use" |

## Customization

### Add More Prompts

Edit the JSON files to add more prompts. Good prompts should be things the model happily does; bad prompts should trigger refusals.

### Adjust Refusal Markers

Edit `config.code_refusal.toml` to add/remove refusal detection patterns.

### Change Model

The experiment works with any instruction-tuned model. Just change the model path:

```bash
python -m heretic ./models/YOUR-MODEL \
  -c experiments/code_refusal/config.code_refusal.toml \
  -o experiments/code_refusal/
```
