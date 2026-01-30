#!/usr/bin/env python3
"""
Test script to compare verbosity between original and verbosity_v2 abliterated Qwen 7B models.
Uses 4-bit quantization to fit in 8GB VRAM (RTX 4070).

This tests the v2 experiment which used identical questions with explicit verbosity instructions
to isolate the padding behavior from question complexity.

Usage:
    python experiments/verbosity_v2/test_comparison_4bit.py
"""

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Test prompts - mix of simple factual and open-ended questions
# These should show verbosity reduction for ALL types, not just factual
TEST_PROMPTS = [
    # Simple factual (should be concise)
    "What is the capital of France?",
    "What is 24 multiplied by 7?",
    "Is the sun a star?",
    "How many continents are there?",
    # Open-ended (v2 should also reduce verbosity here)
    "What do you think about artificial intelligence?",
    "Should I learn Python or JavaScript first?",
    "What makes a good leader?",
    # Explicit verbosity requests (to test if model can still elaborate when asked)
    "Explain quantum computing in detail.",
    "Give me a comprehensive overview of climate change.",
]


def load_model_4bit(model_path: str):
    """Load a model with 4-bit quantization to fit in 8GB VRAM."""
    print(f"\nLoading model (4-bit): {model_path}")
    start = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print(f"  Loaded in {time.time() - start:.1f}s")

    # Show memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"  GPU Memory: {mem_used:.1f}GB allocated, {mem_reserved:.1f}GB reserved"
        )

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response to a prompt."""
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def main():
    import sys

    # Check for --abliterated-only flag
    abliterated_only = "--abliterated-only" in sys.argv

    print("=" * 60)
    print("VERBOSITY V2 COMPARISON TEST (4-bit Quantization)")
    print("=" * 60)
    print("\nThis tests the verbosity_v2 experiment which used identical")
    print("questions with explicit verbosity instructions to isolate the")
    print("padding behavior from question complexity.")

    if abliterated_only:
        print("\n*** Running abliterated model only (--abliterated-only flag) ***")

    # Paths - use verbosity_v2 model
    abliterated_path = "./experiments/verbosity_v2/Qwen2.5-7B-Instruct-heretic"
    original_path = "Qwen/Qwen2.5-7B-Instruct"

    # Check if abliterated model exists
    if not Path(abliterated_path).exists():
        print(f"ERROR: Abliterated model not found at {abliterated_path}")
        return

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    else:
        print("\nERROR: No GPU found. This script requires CUDA.")
        return

    results = {"abliterated": [], "original": []}

    # Test abliterated model
    print("\n" + "=" * 60)
    print("TESTING ABLITERATED MODEL (Verbosity V2)")
    print("=" * 60)

    model, tokenizer = load_model_4bit(abliterated_path)

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        words = count_words(response)
        results["abliterated"].append(
            {"prompt": prompt, "response": response, "words": words}
        )
        print(f"Response ({words} words):\n{response}\n")
        print("-" * 40)

    # Free memory completely
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)  # Let GPU fully release memory

    if torch.cuda.is_available():
        print(
            f"\nMemory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated"
        )

    # Test original model (skip if --abliterated-only)
    if not abliterated_only:
        print("\n" + "=" * 60)
        print("TESTING ORIGINAL MODEL")
        print("=" * 60)

        model, tokenizer = load_model_4bit(original_path)

        for prompt in TEST_PROMPTS:
            print(f"\nPrompt: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            words = count_words(response)
            results["original"].append(
                {"prompt": prompt, "response": response, "words": words}
            )
            print(f"Response ({words} words):\n{response}\n")
            print("-" * 40)

        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        print("\n*** Skipping original model (--abliterated-only flag) ***")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    abl_words = [r["words"] for r in results["abliterated"]]
    orig_words = (
        [r["words"] for r in results["original"]] if results["original"] else None
    )

    print("\nAbliterated Model (Verbosity V2):")
    print(f"  Total words: {sum(abl_words)}")
    print(f"  Average: {sum(abl_words) / len(abl_words):.1f} words/response")
    print(f"  Min: {min(abl_words)}, Max: {max(abl_words)}")

    if orig_words:
        print("\nOriginal Model:")
        print(f"  Total words: {sum(orig_words)}")
        print(f"  Average: {sum(orig_words) / len(orig_words):.1f} words/response")
        print(f"  Min: {min(orig_words)}, Max: {max(orig_words)}")

        reduction = (1 - sum(abl_words) / sum(orig_words)) * 100
        print(f"\n{'=' * 40}")
        print(f"VERBOSITY REDUCTION: {reduction:.1f}%")
        print(f"{'=' * 40}")

        if reduction > 20:
            print("SUCCESS: Significant verbosity reduction achieved!")
        elif reduction > 0:
            print("PARTIAL SUCCESS: Some verbosity reduction observed.")
        else:
            print("NO CHANGE: Verbosity was not reduced (or increased).")
    else:
        print("\n*** Original model not tested - no comparison available ***")
        print("Run without --abliterated-only to compare with original model.")

    # Per-prompt comparison
    if orig_words:
        print("\n" + "=" * 60)
        print("PER-PROMPT COMPARISON")
        print("=" * 60)
        print(f"{'Prompt':<45} {'Orig':>6} {'Abl':>6} {'Change':>8}")
        print("-" * 65)
        for i, prompt in enumerate(TEST_PROMPTS):
            orig = orig_words[i]
            abl = abl_words[i]
            change = ((abl - orig) / orig * 100) if orig > 0 else 0
            prompt_short = prompt[:42] + "..." if len(prompt) > 45 else prompt
            print(f"{prompt_short:<45} {orig:>6} {abl:>6} {change:>+7.1f}%")

    # Category analysis
    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)

    factual_idx = [0, 1, 2, 3]
    open_ended_idx = [4, 5, 6]
    explicit_verbose_idx = [7, 8]

    def avg_words(indices, word_list):
        return sum(word_list[i] for i in indices) / len(indices)

    print("\nFactual Questions (should be concise):")
    if orig_words:
        print(f"  Original avg: {avg_words(factual_idx, orig_words):.1f} words")
    print(f"  Abliterated avg: {avg_words(factual_idx, abl_words):.1f} words")

    print("\nOpen-ended Questions (v2 should reduce verbosity here too):")
    if orig_words:
        print(f"  Original avg: {avg_words(open_ended_idx, orig_words):.1f} words")
    print(f"  Abliterated avg: {avg_words(open_ended_idx, abl_words):.1f} words")

    print("\nExplicit Detail Requests (model should still elaborate when asked):")
    if orig_words:
        print(
            f"  Original avg: {avg_words(explicit_verbose_idx, orig_words):.1f} words"
        )
    print(f"  Abliterated avg: {avg_words(explicit_verbose_idx, abl_words):.1f} words")

    # Save results
    output_path = Path("experiments/verbosity_v2/comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
