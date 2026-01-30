#!/usr/bin/env python3
"""Test script to verify the code refusal experiment setup.

Run this before the full abliteration to ensure:
1. Datasets load correctly
2. Model can be loaded
3. Refusal markers are detected in baseline responses
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_datasets():
    """Verify the prompt datasets exist and are valid JSON."""
    print("\n" + "=" * 60)
    print("TESTING DATASET FILES")
    print("=" * 60)

    experiment_dir = Path(__file__).parent

    good_prompts_path = experiment_dir / "good_code_prompts.json"
    bad_prompts_path = experiment_dir / "bad_code_prompts.json"

    # Check good prompts
    if not good_prompts_path.exists():
        print(f"ERROR: {good_prompts_path} not found")
        return False

    with open(good_prompts_path) as f:
        good_prompts = json.load(f)
    print(f"✓ Good prompts: {len(good_prompts)} prompts loaded")
    print(f"  Example: {good_prompts[0][:60]}...")

    # Check bad prompts
    if not bad_prompts_path.exists():
        print(f"ERROR: {bad_prompts_path} not found")
        return False

    with open(bad_prompts_path) as f:
        bad_prompts = json.load(f)
    print(f"✓ Bad prompts: {len(bad_prompts)} prompts loaded")
    print(f"  Example: {bad_prompts[0][:60]}...")

    return True


def test_config():
    """Verify the config file exists and is valid TOML."""
    print("\n" + "=" * 60)
    print("TESTING CONFIG FILE")
    print("=" * 60)

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config.code_refusal.toml"

    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return False

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print("✓ Config loaded successfully")
    print(f"  n_trials: {config.get('n_trials', 'not set')}")
    print(f"  refusal_markers: {len(config.get('refusal_markers', []))} markers")
    print(f"  Sample markers: {config.get('refusal_markers', [])[:3]}")

    return True


def test_baseline_refusals(model_path: str = None, num_prompts: int = 5):
    """Test baseline refusal rate on bad prompts."""
    print("\n" + "=" * 60)
    print("TESTING BASELINE REFUSALS")
    print("=" * 60)

    # Check if model path exists
    if model_path is None:
        model_path = project_root / "models" / "Qwen2.5-7B-Instruct"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ Model not found at {model_path}")
        print(
            "  Skipping baseline test. Run with --model-path to specify model location."
        )
        return True  # Not a failure, just skip

    print(f"Loading model from {model_path}...")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("ERROR: transformers or torch not installed")
        return False

    # Load config for refusal markers
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config.code_refusal.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    refusal_markers = config.get("refusal_markers", [])

    # Load prompts
    with open(Path(__file__).parent / "bad_code_prompts.json") as f:
        bad_prompts = json.load(f)[:num_prompts]

    # Load model with 4-bit quantization
    print("Loading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    print(f"\nTesting {num_prompts} bad prompts for refusals...\n")

    refusals = 0
    for i, prompt in enumerate(bad_prompts):
        messages = [
            {
                "role": "system",
                "content": config.get("system_prompt", "You are a helpful assistant."),
            },
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Check for refusal
        response_lower = response.lower()
        is_refusal = any(marker.lower() in response_lower for marker in refusal_markers)

        if is_refusal:
            refusals += 1
            status = "❌ REFUSED"
        else:
            status = "✓ COMPLIED"

        print(f"[{i + 1}/{num_prompts}] {status}")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Response: {response[:100]}...")
        print()

    refusal_rate = (refusals / num_prompts) * 100
    print("=" * 60)
    print(f"BASELINE REFUSAL RATE: {refusals}/{num_prompts} ({refusal_rate:.1f}%)")
    print("=" * 60)

    if refusal_rate > 50:
        print("\n✓ Good! Model has high refusal rate - abliteration should help.")
    else:
        print(
            "\n⚠ Warning: Low refusal rate. Model may already comply with these prompts."
        )

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test code refusal experiment setup")
    parser.add_argument(
        "--model-path", type=str, help="Path to model for baseline testing"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to test"
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip model loading test"
    )

    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("# CODE REFUSAL EXPERIMENT SETUP TEST")
    print("#" * 60)

    # Test datasets
    if not test_datasets():
        print("\n❌ Dataset test failed!")
        sys.exit(1)

    # Test config
    if not test_config():
        print("\n❌ Config test failed!")
        sys.exit(1)

    # Test baseline refusals (optional)
    if not args.skip_model:
        if not test_baseline_refusals(args.model_path, args.num_prompts):
            print("\n❌ Baseline test failed!")
            sys.exit(1)

    print("\n" + "#" * 60)
    print("# ALL TESTS PASSED ✓")
    print("#" * 60)
    print("\nTo run the abliteration experiment:")
    print(
        "  python -m heretic ./models/Qwen2.5-7B-Instruct -c experiments/code_refusal/config.code_refusal.toml -o experiments/code_refusal/"
    )


if __name__ == "__main__":
    main()
