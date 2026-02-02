#!/usr/bin/env python3
"""
Custom benchmark script for Moonlight-16B abliterated model.
Works around lm-evaluation-harness compatibility issues with DeepSeek-V3 architecture.
"""

import json
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "/workspace/moonlight"
OUTPUT_PATH = "/workspace/benchmark_results.json"
MMLU_CATEGORIES = [
    "abstract_algebra",
    "high_school_physics",
    "high_school_chemistry",
    "computer_security",
    "machine_learning",
]
MMLU_SAMPLES_PER_CAT = 30
HELLASWAG_SAMPLES = 200
GSM8K_SAMPLES = 100


def load_model():
    """Load the Moonlight model."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded! GPU memory: {gpu_mem:.1f}GB")

    return model, tokenizer


def evaluate_mmlu(model, tokenizer, categories, samples_per_cat):
    """Evaluate on MMLU benchmark."""
    print(f"\n{'=' * 50}")
    print("MMLU Benchmark")
    print(f"{'=' * 50}")

    results = {}
    total_correct = 0
    total_samples = 0

    for category in categories:
        print(f"\nEvaluating: {category}")
        try:
            dataset = load_dataset(
                "cais/mmlu", category, split="test", trust_remote_code=True
            )
            samples = list(dataset)[:samples_per_cat]
        except Exception as e:
            print(f"  Failed to load {category}: {e}")
            continue

        correct = 0
        for item in tqdm(samples, desc=f"  {category}"):
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]

            # Format as multiple choice
            prompt = f"Question: {question}\n"
            prompt += (
                f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            )
            prompt += "Answer with just the letter (A, B, C, or D): "

            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # Extract answer letter
            predicted = response[0].upper() if response else ""
            expected = chr(65 + answer_idx)  # 0->A, 1->B, etc.

            if predicted == expected:
                correct += 1

        accuracy = correct / len(samples) * 100
        results[category] = {
            "correct": correct,
            "total": len(samples),
            "accuracy": accuracy,
        }
        total_correct += correct
        total_samples += len(samples)

        print(f"  {category}: {correct}/{len(samples)} = {accuracy:.1f}%")

    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    results["overall"] = {
        "correct": total_correct,
        "total": total_samples,
        "accuracy": overall_accuracy,
    }
    print(f"\nMMU Overall: {total_correct}/{total_samples} = {overall_accuracy:.1f}%")

    return results


def evaluate_hellaswag(model, tokenizer, n_samples):
    """Evaluate on HellaSwag benchmark (commonsense reasoning)."""
    print(f"\n{'=' * 50}")
    print("HellaSwag Benchmark")
    print(f"{'=' * 50}")

    print("Loading HellaSwag dataset...")
    dataset = load_dataset(
        "Rowan/hellaswag", split="validation", trust_remote_code=True
    )
    samples = list(dataset)[:n_samples]

    correct = 0
    for item in tqdm(samples, desc="HellaSwag"):
        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        # Format as multiple choice
        prompt = "Complete the following sentence in the most logical way.\n\n"
        prompt += f"Context: {context}\n\n"
        prompt += "Options:\n"
        prompt += (
            f"A. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}\n"
        )
        prompt += "\nWhich option (A, B, C, or D) best completes the context? Answer with just the letter: "

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        predicted = response[0].upper() if response else ""
        expected = chr(65 + label)

        if predicted == expected:
            correct += 1

    accuracy = correct / len(samples) * 100
    print(f"\nHellaSwag: {correct}/{len(samples)} = {accuracy:.1f}%")

    return {"correct": correct, "total": len(samples), "accuracy": accuracy}


def evaluate_gsm8k(model, tokenizer, n_samples):
    """Evaluate on GSM8K benchmark (math reasoning)."""
    print(f"\n{'=' * 50}")
    print("GSM8K Benchmark (Math Reasoning)")
    print(f"{'=' * 50}")

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    samples = list(dataset)[:n_samples]

    correct = 0
    for item in tqdm(samples, desc="GSM8K"):
        question = item["question"]
        answer_text = item["answer"]

        # Extract the final numeric answer (after ####)
        if "####" in answer_text:
            expected_answer = answer_text.split("####")[-1].strip()
            expected_answer = expected_answer.replace(",", "").strip()
        else:
            continue

        prompt = f"Solve this math problem step by step. End your answer with the final number on its own line.\n\nProblem: {question}\n\nSolution:"

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Try to extract final number from response
        import re

        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response.replace(",", ""))
        predicted = numbers[-1] if numbers else ""

        # Compare (handle decimals and integers)
        try:
            if float(predicted) == float(expected_answer):
                correct += 1
        except (ValueError, TypeError):
            pass

    accuracy = correct / len(samples) * 100
    print(f"\nGSM8K: {correct}/{len(samples)} = {accuracy:.1f}%")

    return {"correct": correct, "total": len(samples), "accuracy": accuracy}


def main():
    print("=" * 60)
    print("Moonlight-16B Abliterated Model Benchmark")
    print("=" * 60)

    start_time = time.time()

    # Load model
    model, tokenizer = load_model()

    results = {
        "model": MODEL_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    # Run benchmarks
    results["benchmarks"]["mmlu"] = evaluate_mmlu(
        model, tokenizer, MMLU_CATEGORIES, MMLU_SAMPLES_PER_CAT
    )

    results["benchmarks"]["hellaswag"] = evaluate_hellaswag(
        model, tokenizer, HELLASWAG_SAMPLES
    )

    results["benchmarks"]["gsm8k"] = evaluate_gsm8k(model, tokenizer, GSM8K_SAMPLES)

    # Calculate overall time
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print()
    print("Results:")
    print(
        f"  MMLU Overall:  {results['benchmarks']['mmlu']['overall']['accuracy']:.1f}%"
    )
    print(f"  HellaSwag:     {results['benchmarks']['hellaswag']['accuracy']:.1f}%")
    print(f"  GSM8K:         {results['benchmarks']['gsm8k']['accuracy']:.1f}%")
    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
