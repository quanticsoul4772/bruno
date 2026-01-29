#!/usr/bin/env python3
"""Generate checksums for model weight files.

This script creates a checksums.json file that can be used to verify
model integrity before loading. Useful for detecting corrupted downloads
or tampered models.

Usage:
    python scripts/generate_checksums.py /path/to/model
    python scripts/generate_checksums.py /path/to/model --output checksums.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Files to checksum (weight files and critical config)
CHECKSUM_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def generate_checksums(model_dir: Path) -> dict:
    """Generate checksums for all model files."""
    checksums = {}
    
    for pattern in CHECKSUM_PATTERNS:
        for filepath in model_dir.glob(pattern):
            if filepath.is_file():
                relative_path = filepath.relative_to(model_dir)
                print(f"Hashing {relative_path}...", end=" ", flush=True)
                checksums[str(relative_path)] = compute_sha256(filepath)
                print("done")
    
    return checksums


def main():
    parser = argparse.ArgumentParser(
        description="Generate checksums for model weight files"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: <model_dir>/checksums.json)",
    )
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output or (args.model_dir / "checksums.json")
    
    print(f"Generating checksums for: {args.model_dir}")
    checksums = generate_checksums(args.model_dir)
    
    if not checksums:
        print("Warning: No files found to checksum", file=sys.stderr)
        sys.exit(1)
    
    # Write checksums file
    with open(output_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "algorithm": "sha256",
                "files": checksums,
            },
            f,
            indent=2,
        )
    
    print(f"\nChecksums written to: {output_path}")
    print(f"Total files: {len(checksums)}")


if __name__ == "__main__":
    main()
