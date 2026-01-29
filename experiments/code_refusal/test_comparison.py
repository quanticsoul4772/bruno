#!/usr/bin/env python3
"""Compare original vs abliterated model on code refusal prompts.

Run this after abliteration to see the difference in refusal rates.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_model_4bit(model_path: str):
    """Load a model with 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"":0},
        torch_dtype=torch.bfloat16,
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, system_prompt: str) -> str:
    """Generate a response from the model."""
    import torch
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def check_refusal(response: str, markers: list) -> bool:
    """Check if response contains refusal markers."""
    response_lower = response.lower()
    return any(marker.lower() in response_lower for marker in markers)


def test_model(model, tokenizer, prompts: list, markers: list, system_prompt: str, model_name: str):
    """Test a model on prompts and return refusal stats."""
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    refusals = 0
    results = []
    
    for i, prompt in enumerate(prompts):
        response = generate_response(model, tokenizer, prompt, system_prompt)
        is_refusal = check_refusal(response, markers)
        
        if is_refusal:
            refusals += 1
            status = "❌ REFUSED"
        else:
            status = "✓ COMPLIED"
        
        results.append({
            "prompt": prompt,
            "response": response,
            "refused": is_refusal
        })
        
        print(f"\n[{i+1}/{len(prompts)}] {status}")
        print(f"Prompt: {prompt[:60]}...")
        print(f"Response: {response[:150]}...")
    
    refusal_rate = (refusals / len(prompts)) * 100
    print(f"\n{'='*60}")
    print(f"REFUSAL RATE: {refusals}/{len(prompts)} ({refusal_rate:.1f}%)")
    print(f"{'='*60}")
    
    return results, refusal_rate


def main():
    import argparse
    import gc
    import torch
    
    parser = argparse.ArgumentParser(description="Compare original vs abliterated model")
    parser.add_argument("--original", type=str, default="./models/Qwen2.5-7B-Instruct",
                        help="Path to original model")
    parser.add_argument("--abliterated", type=str, default="./experiments/code_refusal/Qwen2.5-7B-Instruct-heretic",
                        help="Path to abliterated model")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts to test")
    parser.add_argument("--output", type=str, default="./experiments/code_refusal/comparison_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Load config
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    
    config_path = Path(__file__).parent / "config.code_refusal.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    markers = config.get("refusal_markers", [])
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    # Load prompts
    with open(Path(__file__).parent / "bad_code_prompts.json") as f:
        prompts = json.load(f)[:args.num_prompts]
    
    print("\n" + "#" * 60)
    print("# CODE REFUSAL COMPARISON TEST")
    print("#" * 60)
    print(f"\nTesting {len(prompts)} prompts on both models...")
    
    # Test original model
    print(f"\nLoading original model: {args.original}")
    model, tokenizer = load_model_4bit(args.original)
    original_results, original_rate = test_model(
        model, tokenizer, prompts, markers, system_prompt, "ORIGINAL MODEL"
    )
    
    # Free memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test abliterated model
    print(f"\nLoading abliterated model: {args.abliterated}")
    model, tokenizer = load_model_4bit(args.abliterated)
    abliterated_results, abliterated_rate = test_model(
        model, tokenizer, prompts, markers, system_prompt, "ABLITERATED MODEL"
    )
    
    # Summary
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    print(f"\nOriginal Model Refusal Rate:    {original_rate:.1f}%")
    print(f"Abliterated Model Refusal Rate: {abliterated_rate:.1f}%")
    print(f"\nREFUSAL REDUCTION: {original_rate - abliterated_rate:.1f} percentage points")
    
    if abliterated_rate < original_rate:
        print("\n✓ SUCCESS: Abliteration reduced refusals!")
    else:
        print("\n⚠ WARNING: Abliteration did not reduce refusals")
    
    # Save results
    results = {
        "original": {
            "model_path": args.original,
            "refusal_rate": original_rate,
            "results": original_results
        },
        "abliterated": {
            "model_path": args.abliterated,
            "refusal_rate": abliterated_rate,
            "results": abliterated_results
        },
        "prompts_tested": len(prompts),
        "refusal_reduction": original_rate - abliterated_rate
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
