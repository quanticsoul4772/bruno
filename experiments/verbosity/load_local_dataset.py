#!/usr/bin/env python3
"""Convert local JSON prompts to HuggingFace Dataset format.

This script creates a local HuggingFace dataset from JSON prompt files
so they can be used with heretic's standard dataset loading.

Usage:
    python load_local_dataset.py

This will create datasets in:
    - experiments/verbosity/concise_dataset/
    - experiments/verbosity/verbose_dataset/
"""

import json
from pathlib import Path

from datasets import Dataset


def create_dataset(json_path: str, output_dir: str):
    """Create a HuggingFace dataset from a JSON file of prompts."""
    with open(json_path) as f:
        prompts = json.load(f)
    
    # Create dataset with 'text' column (matching heretic's expected format)
    dataset = Dataset.from_dict({"text": prompts})
    
    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    
    # Save to disk
    split_dataset.save_to_disk(output_dir)
    
    print(f"Created dataset at {output_dir}")
    print(f"  Train: {len(split_dataset['train'])} prompts")
    print(f"  Test: {len(split_dataset['test'])} prompts")
    
    return split_dataset


def main():
    base_dir = Path(__file__).parent
    
    # Create concise prompts dataset
    create_dataset(
        base_dir / "concise_prompts.json",
        base_dir / "concise_dataset",
    )
    
    # Create verbose prompts dataset
    create_dataset(
        base_dir / "verbose_prompts.json",
        base_dir / "verbose_dataset",
    )
    
    print("\nDatasets created! Update config.verbosity.toml to use:")
    print("")
    print("[good_prompts]")
    print('dataset = "experiments/verbosity/concise_dataset"')
    print('split = "train"')
    print('column = "text"')
    print("")
    print("[bad_prompts]")
    print('dataset = "experiments/verbosity/verbose_dataset"')
    print('split = "train"')
    print('column = "text"')


if __name__ == "__main__":
    main()
