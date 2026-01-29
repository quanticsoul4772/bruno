#!/usr/bin/env python3
"""Verify model weight checksums.

This script verifies that model files match their expected checksums,
detecting corrupted downloads or tampered models.

Usage:
    python scripts/verify_checksums.py /path/to/model
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_checksums(model_dir: Path) -> tuple[bool, list[str]]:
    """Verify checksums for a model directory.
    
    Returns:
        Tuple of (all_valid, list of error messages)
    """
    checksums_file = model_dir / "checksums.json"
    
    if not checksums_file.exists():
        return True, ["No checksums.json found - skipping verification"]
    
    with open(checksums_file) as f:
        data = json.load(f)
    
    if data.get("algorithm") != "sha256":
        return False, [f"Unsupported algorithm: {data.get('algorithm')}"]
    
    errors = []
    files_checked = 0
    
    for filename, expected_hash in data.get("files", {}).items():
        filepath = model_dir / filename
        
        if not filepath.exists():
            errors.append(f"Missing file: {filename}")
            continue
        
        print(f"Verifying {filename}...", end=" ", flush=True)
        actual_hash = compute_sha256(filepath)
        
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch: {filename}\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
            print("FAILED")
        else:
            print("OK")
            files_checked += 1
    
    if not errors:
        print(f"\nAll {files_checked} files verified successfully!")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Verify model weight checksums"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory",
    )
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Verifying checksums for: {args.model_dir}\n")
    valid, errors = verify_checksums(args.model_dir)
    
    if not valid:
        print("\nVerification FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
