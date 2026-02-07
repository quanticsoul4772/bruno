# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""GGUF conversion and quantization phase.

This module handles converting abliterated models to GGUF format
and creating multiple quantization variants.
"""

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

from ..config import Settings
from ..utils import print

# Paths to llama.cpp tools
LLAMA_CPP_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


def find_quantize_binary() -> Path | None:
    """Find llama-quantize binary in known locations.

    Returns:
        Path to llama-quantize binary, or None if not found
    """
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",  # Windows
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",  # Windows alt
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",  # Linux
        LLAMA_CPP_DIR / "llama-quantize",  # Root
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_imatrix_binary() -> Path | None:
    """Find llama-imatrix binary in known locations.

    Returns:
        Path to llama-imatrix binary, or None if not found
    """
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-imatrix.exe",  # Windows
        LLAMA_CPP_DIR / "build" / "bin" / "llama-imatrix.exe",  # Windows alt
        LLAMA_CPP_DIR / "build" / "bin" / "llama-imatrix",  # Linux
        LLAMA_CPP_DIR / "llama-imatrix",  # Root
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def generate_imatrix(
    f16_gguf: str, calibration_file: str, output: str, n_gpu_layers: int = 99
) -> str:
    """Generate importance matrix for higher quality quantization.

    Args:
        f16_gguf: Path to F16 GGUF file
        calibration_file: Path to calibration text file
        output: Output path for imatrix file
        n_gpu_layers: Number of layers to offload to GPU (default: 99 = all)

    Returns:
        Path to generated imatrix file

    Raises:
        FileNotFoundError: If llama-imatrix binary not found
        subprocess.CalledProcessError: If generation fails
    """
    imatrix_bin = find_imatrix_binary()
    if not imatrix_bin:
        raise FileNotFoundError(
            "llama-imatrix binary not found. Build llama.cpp first:\n"
            "  cd tools/llama.cpp\n"
            "  cmake -B build\n"
            "  cmake --build build --config Release"
        )

    print(f"Generating importance matrix: [bold]{Path(output).name}[/]...")

    cmd = [
        str(imatrix_bin),
        "-m",
        f16_gguf,
        "-f",
        calibration_file,
        "--chunk",
        "512",
        "-o",
        output,
        "-ngl",
        str(n_gpu_layers),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"  Importance matrix generated: {Path(output).stat().st_size / 1024:.1f} KB")

    return output


def download_calibration_data(output_path: str) -> str:
    """Download wikitext-2 calibration data for imatrix generation.

    Args:
        output_path: Where to save the calibration text file

    Returns:
        Path to the downloaded calibration file
    """
    from datasets import load_dataset

    print("Downloading calibration data (wikitext-2)...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Join all text entries
    calibration_text = "\n\n".join(dataset["text"])

    # Save to file
    Path(output_path).write_text(calibration_text, encoding="utf-8")

    print(f"  Calibration data saved: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path


def convert_to_gguf_f16(model_path: str, output_path: str) -> str:
    """Convert HuggingFace model to F16 GGUF.

    Args:
        model_path: Path to the HuggingFace model directory
        output_path: Output path for the F16 GGUF file

    Returns:
        Path to the created F16 GGUF file

    Raises:
        FileNotFoundError: If convert script not found
        subprocess.CalledProcessError: If conversion fails
    """
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}. "
            "Ensure llama.cpp submodule is initialized."
        )

    print(f"Converting to F16 GGUF: [bold]{output_path}[/]...")

    cmd = [
        "python",
        str(CONVERT_SCRIPT),
        model_path,
        "--outfile",
        output_path,
        "--outtype",
        "f16",
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"  F16 GGUF created: {Path(output_path).stat().st_size / (1024**3):.1f} GB")

    return output_path


def quantize_gguf(
    input_gguf: str, output_gguf: str, quant_type: str, imatrix: str | None = None
) -> str:
    """Quantize F16 GGUF to specified quantization type.

    Args:
        input_gguf: Path to F16 GGUF file
        output_gguf: Output path for quantized GGUF
        quant_type: Quantization type (Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.)
        imatrix: Optional path to importance matrix file

    Returns:
        Path to quantized GGUF file

    Raises:
        FileNotFoundError: If llama-quantize binary not found
        subprocess.CalledProcessError: If quantization fails
    """
    quantize_bin = find_quantize_binary()
    if not quantize_bin:
        raise FileNotFoundError(
            "llama-quantize binary not found. Build llama.cpp first:\n"
            "  cd tools/llama.cpp\n"
            "  cmake -B build\n"
            "  cmake --build build --config Release"
        )

    print(f"Quantizing to {quant_type}: [bold]{Path(output_gguf).name}[/]...")

    cmd = [str(quantize_bin)]
    if imatrix:
        cmd.extend(["--imatrix", imatrix])
    cmd.extend([input_gguf, output_gguf, quant_type])

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    size_gb = Path(output_gguf).stat().st_size / (1024**3)
    print(f"  {quant_type} created: {size_gb:.1f} GB")

    return output_gguf


def convert_and_quantize(model_path: str, settings: Settings) -> list[Path]:
    """Full pipeline: HF model -> F16 GGUF -> multiple quantized GGUFs.

    Args:
        model_path: Path to the HuggingFace model directory
        settings: Configuration settings with GGUF options

    Returns:
        List of paths to created GGUF files (including F16 if not cleaned up)
    """
    model_name = Path(model_path).name
    output_dir = Path(settings.gguf_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to F16 GGUF
    f16_path = output_dir / f"{model_name}-F16.gguf"
    convert_to_gguf_f16(model_path, str(f16_path))

    # Step 2: Generate importance matrix if requested (Feature 9)
    imatrix_path = None
    if settings.gguf_imatrix:
        try:
            # Download/prepare calibration data
            calib_path = output_dir / "calibration.txt"
            if settings.gguf_imatrix_dataset == "wikitext":
                download_calibration_data(str(calib_path))
            else:
                # Assume it's a file path
                calib_path = Path(settings.gguf_imatrix_dataset)
                if not calib_path.exists():
                    raise FileNotFoundError(f"Calibration file not found: {calib_path}")

            # Generate imatrix
            imatrix_path = str(output_dir / f"{model_name}-imatrix.dat")
            generate_imatrix(str(f16_path), str(calib_path), imatrix_path)

        except Exception as e:
            print(f"[yellow]Imatrix generation failed: {e}[/]")
            print("[yellow]Continuing with standard quantization[/]")
            imatrix_path = None

    # Step 3: Quantize to multiple formats (parallel)
    gguf_files = []
    quant_jobs = {}

    for quant_type in settings.gguf_quantizations:
        quant_path = output_dir / f"{model_name}-{quant_type}.gguf"
        quant_jobs[quant_type] = quant_path

    print(f"\nQuantizing to {len(quant_jobs)} formats in parallel...")

    # Add imatrix suffix to filenames if used
    if imatrix_path:
        quant_jobs = {
            qt: output_dir / f"{model_name}-{qt}-imat.gguf"
            for qt in settings.gguf_quantizations
        }

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                quantize_gguf, str(f16_path), str(out_path), qt, imatrix_path
            ): qt
            for qt, out_path in quant_jobs.items()
        }

        for future in as_completed(futures):
            quant_type = futures[future]
            try:
                result_path = future.result()
                gguf_files.append(Path(result_path))
            except Exception as e:
                print(f"[red]Failed to quantize {quant_type}: {e}[/]")

    # Step 3: Cleanup F16 if requested
    if settings.gguf_cleanup_f16 and f16_path.exists():
        f16_path.unlink()
        print(f"Cleaned up F16 intermediate: {f16_path.name}")
    elif f16_path.exists():
        gguf_files.insert(0, f16_path)

    print(f"\n[bold green]Created {len(gguf_files)} GGUF files[/]")
    return gguf_files


def generate_gguf_readme(
    base_repo_id: str, model_name: str, gguf_files: list[Path]
) -> str:
    """Generate README for GGUF repository.

    Args:
        base_repo_id: Original safetensors repo ID
        model_name: Model name
        gguf_files: List of GGUF file paths

    Returns:
        README markdown content
    """
    # Quantization quality descriptions
    quant_info = {
        "Q4_K_M": ("4.83", "Good balance, recommended for most users"),
        "Q5_K_M": ("5.69", "Higher quality, moderate VRAM"),
        "Q6_K": ("6.56", "High quality"),
        "Q8_0": ("8.50", "Near-lossless"),
        "F16": ("16.00", "Full precision (no quantization)"),
    }

    sections = []

    sections.append(f"""# {model_name} - GGUF

GGUF quantized versions of [{base_repo_id}](https://huggingface.co/{base_repo_id}).

## Available Quantizations

| Quantization | File | Size | Bits/Weight | Use Case |
| :----------- | :--- | :--: | :---------: | :------- |
""")

    for gguf_file in sorted(gguf_files):
        # Extract quant type from filename
        quant_type = gguf_file.stem.split("-")[-1]
        size_gb = gguf_file.stat().st_size / (1024**3)

        bits, description = quant_info.get(quant_type, ("?", "Unknown"))

        sections.append(
            f"| {quant_type} | {gguf_file.name} | {size_gb:.1f} GB | {bits} | {description} |"
        )

    sections.append(
        """
## Usage

### With llama.cpp

```bash
# Download a quantization
wget https://huggingface.co/"""
        + f"{base_repo_id}-GGUF/resolve/main/"
        + """{FILENAME}

# Run with llama-cli
./llama-cli -m {FILENAME} -p "Hello, how are you?" -n 512
```

### With Ollama

```bash
# Create Modelfile
cat > Modelfile <<EOF
FROM ./{FILENAME}
EOF

# Load model
ollama create my-model -f Modelfile

# Run
ollama run my-model
```

### With llama-server (OpenAI-compatible API)

```bash
./llama-server -m {FILENAME} --port 8080

# Use with OpenAI client
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## About Bruno

Bruno is an advanced abliteration framework featuring:
- Optuna-optimized hyperparameter search
- MPOA (Norm-Preserving Biprojected Abliteration)
- Sacred direction preservation
- Neural refusal detection
- MoE-aware abliteration

See [quanticsoul4772/bruno](https://github.com/quanticsoul4772/bruno) for more information.

## Original Model

This is a quantized version of the abliterated model. See the [original safetensors version](https://huggingface.co/"""
        + base_repo_id
        + """) for full details, benchmarks, and model card.
"""
    )

    return "\n".join(sections)


def upload_gguf_files(
    gguf_files: list[Path], base_repo_id: str, token: str, model_card_text: str
) -> bool:
    """Upload GGUF files to HuggingFace.

    Args:
        gguf_files: List of GGUF file paths to upload
        base_repo_id: Base repository ID (will create {base_repo_id}-GGUF)
        token: HuggingFace token
        model_card_text: README content for the GGUF repo

    Returns:
        True if upload succeeded, False otherwise
    """
    api = HfApi(token=token)
    gguf_repo_id = f"{base_repo_id}-GGUF"

    try:
        print(f"\nCreating GGUF repository: [bold]{gguf_repo_id}[/]...")
        api.create_repo(repo_id=gguf_repo_id, repo_type="model", exist_ok=True)

        # Upload README
        print("Uploading model card...")
        api.upload_file(
            path_or_fileobj=model_card_text.encode(),
            path_in_repo="README.md",
            repo_id=gguf_repo_id,
            token=token,
        )

        # Upload GGUF files
        print(f"Uploading {len(gguf_files)} GGUF files...")
        for gguf_file in gguf_files:
            print(f"  Uploading {gguf_file.name}...")
            api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=gguf_file.name,
                repo_id=gguf_repo_id,
                token=token,
            )

        print(f"[bold green]GGUF files uploaded to {gguf_repo_id}[/]")
        return True

    except Exception as e:
        print(f"[red]Failed to upload GGUF files: {e}[/]")
        return False


if __name__ == "__main__":
    # Example usage
    print("Use convert_and_quantize() from your abliteration pipeline")
    print("This module is meant to be imported, not run directly")
