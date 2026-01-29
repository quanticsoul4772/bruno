#!/usr/bin/env python3
"""Convert local JSON prompts to HuggingFace Dataset format.

This script creates local HuggingFace datasets from JSON prompt files
so they can be used with heretic's standard dataset loading.

Usage:
    python create_datasets.py

This will create datasets in:
    - experiments/code_refusal/good_code_dataset/
    - experiments/code_refusal/bad_code_dataset/
"""

import json
from pathlib import Path

from datasets import Dataset


def create_dataset(json_path: str, output_dir: str):
    """Create a HuggingFace dataset from a JSON file of prompts."""
    with open(json_path, encoding="utf-8") as f:
        prompts = json.load(f)
    
    # Create dataset with 'text' column (matching heretic's expected format)
    dataset = Dataset.from_dict({"text": prompts})
    
    # Split into train/test (70/30 split)
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    
    # Save to disk
    split_dataset.save_to_disk(output_dir)
    
    print(f"Created dataset at {output_dir}")
    print(f"  Train: {len(split_dataset['train'])} prompts")
    print(f"  Test: {len(split_dataset['test'])} prompts")
    
    return split_dataset


def main():
    base_dir = Path(__file__).parent
    
    print("=" * 60)
    print("CODE REFUSAL DATASET CREATION")
    print("=" * 60)
    
    # Create good prompts dataset (harmless coding requests)
    print("\nCreating good code prompts dataset (harmless requests)...")
    create_dataset(
        base_dir / "good_code_prompts.json",
        base_dir / "good_code_dataset",
    )
    
    # Create bad prompts dataset (requests that trigger refusals)
    print("\nCreating bad code prompts dataset (refusal-triggering requests)...")
    create_dataset(
        base_dir / "bad_code_prompts.json",
        base_dir / "bad_code_dataset",
    )
    
    print("\n" + "=" * 60)
    print("Datasets created successfully!")
    print("=" * 60)
    
    print("\nSample prompts:")
    
    with open(base_dir / "good_code_prompts.json", encoding="utf-8") as f:
        good = json.load(f)
    with open(base_dir / "bad_code_prompts.json", encoding="utf-8") as f:
        bad = json.load(f)
    
    print("\n  Good (model complies):")
    for p in good[:2]:
        print(f"    - {p[:60]}...")
    
    print("\n  Bad (model refuses):")
    for p in bad[:2]:
        print(f"    - {p[:60]}...")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Run abliteration:")
    print("     python -m heretic ./models/Qwen2.5-7B-Instruct \\")
    print("       -c experiments/code_refusal/config.code_refusal.toml \\")
    print("       -o experiments/code_refusal/")
    print("=" * 60)


if __name__ == "__main__":
    main()
