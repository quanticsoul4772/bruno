#!/usr/bin/env python3
"""Create and manage HuggingFace collections for Bruno models.

This script creates organized collections following sasa2000's approach:
- Main "Bruno Abliterated" collection with all models
- Family-specific collections (Qwen, Moonlight, Llama)
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()


def get_hf_token() -> str:
    """Get HuggingFace token from environment or .env file."""
    token = os.getenv("HF_TOKEN")

    if not token:
        try:
            with open(".env", "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
        except FileNotFoundError:
            pass

    if not token:
        raise ValueError("HF_TOKEN not found in environment or .env file")

    return token


def create_collections():
    """Create HuggingFace collections for Bruno models."""
    token = get_hf_token()
    api = HfApi(token=token)

    # Define collections and their models
    collections = {
        "Bruno Abliterated Models": {
            "description": "Optuna-optimized abliterated models using Bruno framework. Features MPOA, sacred directions, concept cones, and neural refusal detection.",
            "models": [
                "rawcell/Moonlight-16B-A3B-Instruct-bruno",
                "rawcell/Qwen2.5-Coder-7B-Instruct-bruno",
                "rawcell/bruno",
                "rawcell/Moonlight-16B-A3B-Instruct-abliterated",
                "rawcell/Llama-3.2-3B-Instruct-heretic",
                "rawcell/Qwen2.5-3B-Instruct-heretic",
            ],
        },
        "Bruno - Qwen Models": {
            "description": "Abliterated Qwen models optimized for code generation and instruction following.",
            "models": [
                "rawcell/Qwen2.5-Coder-7B-Instruct-bruno",
                "rawcell/Qwen2.5-3B-Instruct-heretic",
            ],
        },
        "Bruno - Moonlight Models": {
            "description": "Abliterated Moonlight MoE models with DeepSeek-V3 architecture support.",
            "models": [
                "rawcell/Moonlight-16B-A3B-Instruct-bruno",
                "rawcell/Moonlight-16B-A3B-Instruct-abliterated",
            ],
        },
        "Bruno - Llama Models": {
            "description": "Abliterated Llama models for general instruction following.",
            "models": [
                "rawcell/Llama-3.2-3B-Instruct-heretic",
            ],
        },
    }

    created = []
    failed = []

    for name, config in collections.items():
        try:
            print(f"Creating collection: {name}")

            # Create collection
            collection = api.create_collection(
                title=name,
                description=config["description"],
                namespace="rawcell",
                private=False,
            )

            print(f"  Collection created: {collection.slug}")

            # Add models to collection
            for model_id in config["models"]:
                try:
                    api.add_collection_item(
                        collection_slug=collection.slug,
                        item_id=model_id,
                        item_type="model",
                    )
                    print(f"  Added: {model_id}")
                except Exception as e:
                    print(f"  Failed to add {model_id}: {e}")

            created.append(name)
            print()

        except Exception as e:
            print(f"  ERROR: Failed to create collection: {e}")
            failed.append(name)
            print()

    # Summary
    print("=" * 80)
    print(f"SUCCESS: Created {len(created)} collections")
    for name in created:
        print(f"  - {name}")

    if failed:
        print(f"\nFAILED: {len(failed)} collections")
        for name in failed:
            print(f"  - {name}")

    print("=" * 80)


if __name__ == "__main__":
    create_collections()
