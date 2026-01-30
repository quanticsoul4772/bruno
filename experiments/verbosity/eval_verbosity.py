#!/usr/bin/env python3
"""Evaluate verbosity changes in a model.

This script measures the average response length (in tokens and words)
for a model before and after verbosity direction extraction.

Usage:
    # Compare original vs modified model
    python eval_verbosity.py --original meta-llama/Llama-3.1-8B-Instruct \
                             --modified ./Llama-3.1-8B-Instruct-concise

    # Just measure a single model
    python eval_verbosity.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import statistics
import sys

import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


def load_prompts(path: str) -> list[str]:
    """Load prompts from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_response(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response


def measure_verbosity(
    model, tokenizer, prompts: list[str], desc: str = "Measuring"
) -> dict:
    """Measure verbosity metrics for a set of prompts."""
    token_counts = []
    word_counts = []
    char_counts = []
    responses = []

    for prompt in track(prompts, description=desc):
        response = get_response(model, tokenizer, prompt)
        responses.append(response)

        # Token count
        tokens = tokenizer.encode(response)
        token_counts.append(len(tokens))

        # Word count
        words = response.split()
        word_counts.append(len(words))

        # Character count
        char_counts.append(len(response))

    return {
        "token_counts": token_counts,
        "word_counts": word_counts,
        "char_counts": char_counts,
        "responses": responses,
        "avg_tokens": statistics.mean(token_counts),
        "avg_words": statistics.mean(word_counts),
        "avg_chars": statistics.mean(char_counts),
        "median_tokens": statistics.median(token_counts),
        "median_words": statistics.median(word_counts),
        "std_tokens": statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
    }


def load_model(model_path: str):
    """Load a model and tokenizer."""
    console.print(f"Loading model: [bold]{model_path}[/]")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    return model, tokenizer


def print_comparison(original_metrics: dict, modified_metrics: dict):
    """Print a comparison table of verbosity metrics."""
    table = Table(title="Verbosity Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Original", justify="right")
    table.add_column("Modified", justify="right")
    table.add_column("Change", justify="right")

    for metric in [
        "avg_tokens",
        "avg_words",
        "avg_chars",
        "median_tokens",
        "median_words",
    ]:
        orig = original_metrics[metric]
        mod = modified_metrics[metric]
        change = ((mod - orig) / orig) * 100 if orig > 0 else 0

        change_str = f"{change:+.1f}%"
        if change < 0:
            change_str = f"[green]{change_str}[/]"
        elif change > 0:
            change_str = f"[red]{change_str}[/]"

        table.add_row(
            metric.replace("_", " ").title(),
            f"{orig:.1f}",
            f"{mod:.1f}",
            change_str,
        )

    console.print(table)


def print_metrics(metrics: dict, name: str):
    """Print metrics for a single model."""
    table = Table(title=f"Verbosity Metrics: {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Avg Tokens", f"{metrics['avg_tokens']:.1f}")
    table.add_row("Avg Words", f"{metrics['avg_words']:.1f}")
    table.add_row("Avg Chars", f"{metrics['avg_chars']:.1f}")
    table.add_row("Median Tokens", f"{metrics['median_tokens']:.1f}")
    table.add_row("Median Words", f"{metrics['median_words']:.1f}")
    table.add_row("Std Tokens", f"{metrics['std_tokens']:.1f}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model verbosity")
    parser.add_argument("--model", help="Single model to evaluate")
    parser.add_argument("--original", help="Original model for comparison")
    parser.add_argument("--modified", help="Modified model for comparison")
    parser.add_argument(
        "--prompts",
        default="experiments/verbosity/verbose_prompts.json",
        help="Path to prompts JSON file",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to evaluate",
    )
    parser.add_argument(
        "--output",
        help="Save detailed results to JSON file",
    )

    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompts)[: args.num_prompts]
    console.print(f"Loaded [bold]{len(prompts)}[/] prompts")

    if args.model:
        # Single model evaluation
        model, tokenizer = load_model(args.model)
        metrics = measure_verbosity(model, tokenizer, prompts, "Measuring verbosity")
        print_metrics(metrics, args.model)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "model": args.model,
                        "metrics": {
                            k: v for k, v in metrics.items() if k != "responses"
                        },
                        "responses": metrics["responses"],
                    },
                    f,
                    indent=2,
                )
            console.print(f"Results saved to [bold]{args.output}[/]")

    elif args.original and args.modified:
        # Comparison mode
        orig_model, orig_tokenizer = load_model(args.original)
        orig_metrics = measure_verbosity(
            orig_model, orig_tokenizer, prompts, "Measuring original"
        )

        # Free memory
        del orig_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mod_model, mod_tokenizer = load_model(args.modified)
        mod_metrics = measure_verbosity(
            mod_model, mod_tokenizer, prompts, "Measuring modified"
        )

        print_comparison(orig_metrics, mod_metrics)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "original": {
                            "model": args.original,
                            "metrics": {
                                k: v
                                for k, v in orig_metrics.items()
                                if k != "responses"
                            },
                            "responses": orig_metrics["responses"],
                        },
                        "modified": {
                            "model": args.modified,
                            "metrics": {
                                k: v for k, v in mod_metrics.items() if k != "responses"
                            },
                            "responses": mod_metrics["responses"],
                        },
                    },
                    f,
                    indent=2,
                )
            console.print(f"Results saved to [bold]{args.output}[/]")

    else:
        console.print(
            "[red]Error: Specify either --model or both --original and --modified[/]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
