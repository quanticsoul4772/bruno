#!/usr/bin/env python3
"""
Test script to compare verbosity between original and abliterated Qwen 7B models.

Usage:
    python experiments/verbosity/test_comparison.py

This will:
1. Load the abliterated model from ./models/Qwen2.5-7B-Instruct-heretic
2. Load the original model from HuggingFace (Qwen/Qwen2.5-7B-Instruct)
3. Run test prompts through both
4. Compare response lengths
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test prompts - mix of simple factual and open-ended questions
TEST_PROMPTS = [
    # Simple factual (should be concise)
    "What is the capital of France?",
    "What is 24 multiplied by 7?",
    "Is the sun a star?",
    "How many continents are there?",
    # Open-ended (typically verbose)
    "What do you think about artificial intelligence?",
    "Should I learn Python or JavaScript first?",
    "What makes a good leader?",
]

def load_model(model_path: str, device_map: str = "auto"):
    """Load a model and tokenizer."""
    print(f"\nLoading model: {model_path}")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    
    print(f"  Loaded in {time.time() - start:.1f}s")
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
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def main():
    print("=" * 60)
    print("VERBOSITY COMPARISON TEST")
    print("=" * 60)
    
    # Paths
    abliterated_path = "./models/Qwen2.5-7B-Instruct-heretic"
    original_path = "Qwen/Qwen2.5-7B-Instruct"
    
    # Check if abliterated model exists
    if not Path(abliterated_path).exists():
        print(f"ERROR: Abliterated model not found at {abliterated_path}")
        return
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        
        if gpu_mem < 16:
            print("\nWARNING: Less than 16GB VRAM. May not fit both models.")
            print("Testing abliterated model only.\n")
            test_both = False
        else:
            test_both = True
    else:
        print("\nWARNING: No GPU found. Testing will be slow.")
        test_both = False
    
    results = {"abliterated": [], "original": []}
    
    # Test abliterated model
    print("\n" + "=" * 60)
    print("TESTING ABLITERATED MODEL")
    print("=" * 60)
    
    model, tokenizer = load_model(abliterated_path)
    
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        words = count_words(response)
        results["abliterated"].append({"prompt": prompt, "response": response, "words": words})
        print(f"Response ({words} words):\n{response}\n")
        print("-" * 40)
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    # Test original model if we have enough memory
    if test_both:
        print("\n" + "=" * 60)
        print("TESTING ORIGINAL MODEL")
        print("=" * 60)
        
        model, tokenizer = load_model(original_path)
        
        for prompt in TEST_PROMPTS:
            print(f"\nPrompt: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            words = count_words(response)
            results["original"].append({"prompt": prompt, "response": response, "words": words})
            print(f"Response ({words} words):\n{response}\n")
            print("-" * 40)
        
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    abl_words = [r["words"] for r in results["abliterated"]]
    print(f"\nAbliterated Model:")
    print(f"  Total words: {sum(abl_words)}")
    print(f"  Average: {sum(abl_words) / len(abl_words):.1f} words/response")
    print(f"  Min: {min(abl_words)}, Max: {max(abl_words)}")
    
    if results["original"]:
        orig_words = [r["words"] for r in results["original"]]
        print(f"\nOriginal Model:")
        print(f"  Total words: {sum(orig_words)}")
        print(f"  Average: {sum(orig_words) / len(orig_words):.1f} words/response")
        print(f"  Min: {min(orig_words)}, Max: {max(orig_words)}")
        
        reduction = (1 - sum(abl_words) / sum(orig_words)) * 100
        print(f"\nVerbosity Reduction: {reduction:.1f}%")
    
    # Save results
    output_path = Path("experiments/verbosity/comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
