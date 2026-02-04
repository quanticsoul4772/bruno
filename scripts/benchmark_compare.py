#!/usr/bin/env python3
"""
Benchmark Comparison Script for Abliterated Models

Compares two models (original vs abliterated) on MMLU, HellaSwag, and GSM8K benchmarks.
Outputs a comparison table and saves detailed results to JSON.

Usage:
    # Compare HuggingFace model with local abliterated model
    python scripts/benchmark_compare.py \
        --model-a moonshotai/Moonlight-16B-A3B-Instruct \
        --model-b /workspace/moonlight-bruno \
        --output results.json

    # Quick test with fewer samples
    python scripts/benchmark_compare.py \
        --model-a Qwen/Qwen2.5-7B-Instruct \
        --model-b ./models/Qwen2.5-7B-Instruct-heretic \
        --mmlu-samples 10 --hellaswag-samples 50 --gsm8k-samples 25

    # With 4-bit quantization for limited VRAM
    python scripts/benchmark_compare.py \
        --model-a Qwen/Qwen2.5-7B-Instruct \
        --model-b ./models/Qwen2.5-7B-Instruct-heretic \
        --quantize-4bit
"""

import argparse
import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default benchmark configurations
DEFAULT_MMLU_CATEGORIES = [
    "abstract_algebra",
    "high_school_physics",
    "high_school_chemistry",
    "computer_security",
    "machine_learning",
]
DEFAULT_MMLU_SAMPLES = 30
DEFAULT_HELLASWAG_SAMPLES = 200
DEFAULT_GSM8K_SAMPLES = 100


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    correct: int
    total: int
    accuracy: float
    details: dict = field(default_factory=dict)


@dataclass
class ModelResults:
    """All benchmark results for a model."""

    model_path: str
    mmlu: BenchmarkResult = None
    hellaswag: BenchmarkResult = None
    gsm8k: BenchmarkResult = None
    load_time: float = 0.0
    benchmark_time: float = 0.0


def load_model(model_path: str, quantize_4bit: bool = False):
    """Load a model and tokenizer."""
    print(f"\nLoading model: {model_path}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if quantize_4bit:
        print("  Using 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    load_time = time.time() - start

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Loaded in {load_time:.1f}s, GPU memory: {gpu_mem:.1f}GB")
    else:
        print(f"  Loaded in {load_time:.1f}s")

    return model, tokenizer, load_time


def unload_model(model, tokenizer):
    """Unload model and free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 5) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    return response


def evaluate_mmlu(
    model, tokenizer, categories: list, samples_per_cat: int
) -> BenchmarkResult:
    """Evaluate on MMLU benchmark."""
    print(f"\n{'=' * 50}")
    print("MMLU Benchmark")
    print(f"{'=' * 50}")

    category_results = {}
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

            prompt = f"Question: {question}\n"
            prompt += (
                f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            )
            prompt += "Answer with just the letter (A, B, C, or D): "

            response = generate_response(model, tokenizer, prompt, max_new_tokens=5)
            predicted = response[0].upper() if response else ""
            expected = chr(65 + answer_idx)

            if predicted == expected:
                correct += 1

        accuracy = correct / len(samples) * 100 if samples else 0
        category_results[category] = {
            "correct": correct,
            "total": len(samples),
            "accuracy": accuracy,
        }
        total_correct += correct
        total_samples += len(samples)

        print(f"  {category}: {correct}/{len(samples)} = {accuracy:.1f}%")

    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\nMMU Overall: {total_correct}/{total_samples} = {overall_accuracy:.1f}%")

    return BenchmarkResult(
        correct=total_correct,
        total=total_samples,
        accuracy=overall_accuracy,
        details={"categories": category_results},
    )


def evaluate_hellaswag(model, tokenizer, n_samples: int) -> BenchmarkResult:
    """Evaluate on HellaSwag benchmark."""
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

        prompt = "Complete the following sentence in the most logical way.\n\n"
        prompt += f"Context: {context}\n\n"
        prompt += "Options:\n"
        prompt += (
            f"A. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}\n"
        )
        prompt += "\nWhich option (A, B, C, or D) best completes the context? Answer with just the letter: "

        response = generate_response(model, tokenizer, prompt, max_new_tokens=5)
        predicted = response[0].upper() if response else ""
        expected = chr(65 + label)

        if predicted == expected:
            correct += 1

    accuracy = correct / len(samples) * 100 if samples else 0
    print(f"\nHellaSwag: {correct}/{len(samples)} = {accuracy:.1f}%")

    return BenchmarkResult(correct=correct, total=len(samples), accuracy=accuracy)


def evaluate_gsm8k(model, tokenizer, n_samples: int) -> BenchmarkResult:
    """Evaluate on GSM8K benchmark."""
    print(f"\n{'=' * 50}")
    print("GSM8K Benchmark (Math Reasoning)")
    print(f"{'=' * 50}")

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    samples = list(dataset)[:n_samples]

    correct = 0
    evaluated = 0

    for item in tqdm(samples, desc="GSM8K"):
        question = item["question"]
        answer_text = item["answer"]

        if "####" not in answer_text:
            continue

        expected_answer = answer_text.split("####")[-1].strip().replace(",", "")

        prompt = f"Solve this math problem step by step. End your answer with the final number on its own line.\n\nProblem: {question}\n\nSolution:"

        response = generate_response(model, tokenizer, prompt, max_new_tokens=256)

        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response.replace(",", ""))
        predicted = numbers[-1] if numbers else ""

        try:
            if float(predicted) == float(expected_answer):
                correct += 1
        except (ValueError, TypeError):
            pass

        evaluated += 1

    accuracy = correct / evaluated * 100 if evaluated > 0 else 0
    print(f"\nGSM8K: {correct}/{evaluated} = {accuracy:.1f}%")

    return BenchmarkResult(correct=correct, total=evaluated, accuracy=accuracy)


def run_benchmarks(
    model,
    tokenizer,
    model_path: str,
    mmlu_categories: list,
    mmlu_samples: int,
    hellaswag_samples: int,
    gsm8k_samples: int,
    load_time: float,
) -> ModelResults:
    """Run all benchmarks on a model."""
    start = time.time()

    results = ModelResults(model_path=model_path, load_time=load_time)

    results.mmlu = evaluate_mmlu(model, tokenizer, mmlu_categories, mmlu_samples)
    results.hellaswag = evaluate_hellaswag(model, tokenizer, hellaswag_samples)
    results.gsm8k = evaluate_gsm8k(model, tokenizer, gsm8k_samples)

    results.benchmark_time = time.time() - start
    return results


def print_comparison(
    results_a: ModelResults, results_b: ModelResults, name_a: str, name_b: str
):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    def calc_change(a: float, b: float) -> str:
        diff = b - a
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1f}%"

    def format_row(benchmark: str, a: BenchmarkResult, b: BenchmarkResult) -> str:
        change = calc_change(a.accuracy, b.accuracy)
        indicator = "[OK]" if b.accuracy >= a.accuracy else "[--]"
        return f"| {benchmark:<15} | {a.accuracy:>6.1f}% ({a.correct:>3}/{a.total:<3}) | {b.accuracy:>6.1f}% ({b.correct:>3}/{b.total:<3}) | {change:>8} {indicator} |"

    # Header
    print(f"\n| {'Benchmark':<15} | {name_a:<20} | {name_b:<20} | {'Change':<12} |")
    print(f"|{'-' * 17}|{'-' * 22}|{'-' * 22}|{'-' * 14}|")

    # Results
    print(format_row("MMLU Overall", results_a.mmlu, results_b.mmlu))
    print(format_row("HellaSwag", results_a.hellaswag, results_b.hellaswag))
    print(format_row("GSM8K", results_a.gsm8k, results_b.gsm8k))

    # MMLU breakdown
    if results_a.mmlu.details.get("categories") and results_b.mmlu.details.get(
        "categories"
    ):
        print(f"\n{'MMLU Breakdown:':<15}")
        print(f"|{'-' * 17}|{'-' * 22}|{'-' * 22}|{'-' * 14}|")

        for category in results_a.mmlu.details["categories"]:
            if category in results_b.mmlu.details["categories"]:
                a_cat = results_a.mmlu.details["categories"][category]
                b_cat = results_b.mmlu.details["categories"][category]

                a_result = BenchmarkResult(
                    correct=a_cat["correct"],
                    total=a_cat["total"],
                    accuracy=a_cat["accuracy"],
                )
                b_result = BenchmarkResult(
                    correct=b_cat["correct"],
                    total=b_cat["total"],
                    accuracy=b_cat["accuracy"],
                )

                short_name = category[:15]
                print(format_row(short_name, a_result, b_result))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_change = (
        (results_b.mmlu.accuracy - results_a.mmlu.accuracy)
        + (results_b.hellaswag.accuracy - results_a.hellaswag.accuracy)
        + (results_b.gsm8k.accuracy - results_a.gsm8k.accuracy)
    ) / 3

    print(f"\nAverage change: {'+' if total_change >= 0 else ''}{total_change:.2f}%")

    if total_change >= 0:
        print(
            "\n[OK] Model B shows improved or maintained performance across benchmarks."
        )
    else:
        print(f"\n[WARN] Model B shows {abs(total_change):.2f}% average degradation.")

    print(f"\nModel A benchmark time: {results_a.benchmark_time / 60:.1f} minutes")
    print(f"Model B benchmark time: {results_b.benchmark_time / 60:.1f} minutes")


def save_results(
    results_a: ModelResults,
    results_b: ModelResults,
    output_path: str,
    name_a: str,
    name_b: str,
):
    """Save comparison results to JSON."""

    def result_to_dict(result: BenchmarkResult) -> dict:
        return {
            "correct": result.correct,
            "total": result.total,
            "accuracy": result.accuracy,
            "details": result.details,
        }

    def model_results_to_dict(results: ModelResults) -> dict:
        return {
            "model_path": results.model_path,
            "mmlu": result_to_dict(results.mmlu),
            "hellaswag": result_to_dict(results.hellaswag),
            "gsm8k": result_to_dict(results.gsm8k),
            "load_time_seconds": results.load_time,
            "benchmark_time_seconds": results.benchmark_time,
        }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_a": {
            "name": name_a,
            **model_results_to_dict(results_a),
        },
        "model_b": {
            "name": name_b,
            **model_results_to_dict(results_b),
        },
        "comparison": {
            "mmlu_change": results_b.mmlu.accuracy - results_a.mmlu.accuracy,
            "hellaswag_change": results_b.hellaswag.accuracy
            - results_a.hellaswag.accuracy,
            "gsm8k_change": results_b.gsm8k.accuracy - results_a.gsm8k.accuracy,
            "average_change": (
                (results_b.mmlu.accuracy - results_a.mmlu.accuracy)
                + (results_b.hellaswag.accuracy - results_a.hellaswag.accuracy)
                + (results_b.gsm8k.accuracy - results_a.gsm8k.accuracy)
            )
            / 3,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two models on MMLU, HellaSwag, and GSM8K benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare original with abliterated model
  python scripts/benchmark_compare.py \\
      --model-a Qwen/Qwen2.5-7B-Instruct \\
      --model-b ./models/Qwen2.5-7B-Instruct-heretic

  # Quick test with fewer samples
  python scripts/benchmark_compare.py \\
      --model-a model-original --model-b model-abliterated \\
      --mmlu-samples 10 --hellaswag-samples 50 --gsm8k-samples 25

  # With 4-bit quantization for limited VRAM
  python scripts/benchmark_compare.py \\
      --model-a model-original --model-b model-abliterated \\
      --quantize-4bit
""",
    )

    parser.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="Path or HuggingFace ID of first model (typically original/baseline)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Path or HuggingFace ID of second model (typically abliterated/modified)",
    )
    parser.add_argument(
        "--name-a",
        type=str,
        default="Model A",
        help="Display name for model A (default: 'Model A')",
    )
    parser.add_argument(
        "--name-b",
        type=str,
        default="Model B",
        help="Display name for model B (default: 'Model B')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_comparison.json",
        help="Output JSON file path (default: benchmark_comparison.json)",
    )
    parser.add_argument(
        "--mmlu-samples",
        type=int,
        default=DEFAULT_MMLU_SAMPLES,
        help=f"Samples per MMLU category (default: {DEFAULT_MMLU_SAMPLES})",
    )
    parser.add_argument(
        "--mmlu-categories",
        type=str,
        nargs="+",
        default=DEFAULT_MMLU_CATEGORIES,
        help="MMLU categories to evaluate",
    )
    parser.add_argument(
        "--hellaswag-samples",
        type=int,
        default=DEFAULT_HELLASWAG_SAMPLES,
        help=f"Number of HellaSwag samples (default: {DEFAULT_HELLASWAG_SAMPLES})",
    )
    parser.add_argument(
        "--gsm8k-samples",
        type=int,
        default=DEFAULT_GSM8K_SAMPLES,
        help=f"Number of GSM8K samples (default: {DEFAULT_GSM8K_SAMPLES})",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Use 4-bit quantization (for limited VRAM)",
    )

    args = parser.parse_args()

    # Auto-generate names from paths if not provided
    if args.name_a == "Model A":
        args.name_a = Path(args.model_a).name
    if args.name_b == "Model B":
        args.name_b = Path(args.model_b).name

    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"\nModel A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print("\nBenchmark settings:")
    print(
        f"  MMLU: {args.mmlu_samples} samples/category, {len(args.mmlu_categories)} categories"
    )
    print(f"  HellaSwag: {args.hellaswag_samples} samples")
    print(f"  GSM8K: {args.gsm8k_samples} samples")
    print(f"  4-bit quantization: {args.quantize_4bit}")

    total_start = time.time()

    # Benchmark Model A
    print("\n" + "#" * 80)
    print(f"# BENCHMARKING MODEL A: {args.name_a}")
    print("#" * 80)

    model_a, tokenizer_a, load_time_a = load_model(args.model_a, args.quantize_4bit)
    results_a = run_benchmarks(
        model_a,
        tokenizer_a,
        args.model_a,
        args.mmlu_categories,
        args.mmlu_samples,
        args.hellaswag_samples,
        args.gsm8k_samples,
        load_time_a,
    )

    # Free memory before loading Model B
    unload_model(model_a, tokenizer_a)

    # Benchmark Model B
    print("\n" + "#" * 80)
    print(f"# BENCHMARKING MODEL B: {args.name_b}")
    print("#" * 80)

    model_b, tokenizer_b, load_time_b = load_model(args.model_b, args.quantize_4bit)
    results_b = run_benchmarks(
        model_b,
        tokenizer_b,
        args.model_b,
        args.mmlu_categories,
        args.mmlu_samples,
        args.hellaswag_samples,
        args.gsm8k_samples,
        load_time_b,
    )

    unload_model(model_b, tokenizer_b)

    # Print comparison
    print_comparison(results_a, results_b, args.name_a, args.name_b)

    # Save results
    save_results(results_a, results_b, args.output, args.name_a, args.name_b)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
