#!/usr/bin/env python3
"""Load hedging prompts and save as HuggingFace datasets.

This creates local datasets that heretic can load via load_from_disk().
"""

import json
from pathlib import Path

from datasets import Dataset


def main():
    script_dir = Path(__file__).parent

    # Load confident prompts (factual questions)
    confident_path = script_dir / "confident_prompts.json"
    with open(confident_path, encoding="utf-8") as f:
        confident_prompts = json.load(f)

    # Load hedged prompts (opinion questions)
    hedged_path = script_dir / "hedged_prompts.json"
    with open(hedged_path, encoding="utf-8") as f:
        hedged_prompts = json.load(f)

    print(f"Loaded {len(confident_prompts)} confident prompts")
    print(f"Loaded {len(hedged_prompts)} hedged prompts")

    # Show comparison of first few
    print("\n=== Prompt Comparison (first 3) ===")
    for i in range(3):
        print(f"\nConfident: {confident_prompts[i]}")
        print(f"Hedged:    {hedged_prompts[i]}")

    # Create HuggingFace datasets
    confident_ds = Dataset.from_dict({"text": confident_prompts})
    hedged_ds = Dataset.from_dict({"text": hedged_prompts})

    # Save to disk (heretic can load these)
    confident_save_path = (
        Path.home() / ".cache" / "huggingface" / "datasets" / "hedging_confident"
    )
    hedged_save_path = (
        Path.home() / ".cache" / "huggingface" / "datasets" / "hedging_hedged"
    )

    confident_ds.save_to_disk(str(confident_save_path))
    hedged_ds.save_to_disk(str(hedged_save_path))

    print("\n=== Datasets Saved ===")
    print(f"Confident: {confident_save_path}")
    print(f"Hedged: {hedged_save_path}")
    print("\nNote: These use load_from_disk() which requires the heretic fix.")
    print("If heretic fails to load, push to HuggingFace Hub instead.")


if __name__ == "__main__":
    main()
