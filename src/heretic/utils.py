# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset, load_from_disk
from optuna import Trial
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


class BatchSizeError(Exception):
    """Raised when batch size is too large for available GPU memory."""

    pass


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage.

    Returns:
        dict with keys: total_gb, used_gb, free_gb, utilization_pct
        Returns zeros if no GPU is available.
    """
    if not torch.cuda.is_available():
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "utilization_pct": 0}

    try:
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total - reserved

        return {
            "total_gb": total / 1e9,
            "used_gb": allocated / 1e9,
            "free_gb": free / 1e9,
            "utilization_pct": (allocated / total) * 100 if total > 0 else 0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "utilization_pct": 0}


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _parse_split_count(split: str) -> int | None:
    """Parse sample count from split string like 'train[:200]'.

    Args:
        split: Split specification (e.g., "train[:200]", "train[400:600]")

    Returns:
        Sample count if parseable, None if no count specified (e.g., "train")

    Examples:
        "train[:200]" -> 200
        "train[400:600]" -> 200 (600-400)
        "train[100:150]" -> 50
        "train" -> None
    """
    import re

    # Match patterns like [M:N] or [:N]
    match = re.search(r"\[(\d*):(\d+)\]", split)
    if match:
        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str)
        return end - start

    # No slice notation - no count specified
    return None


def load_prompts(specification: DatasetSpecification) -> list[str]:
    """Load prompts from dataset, using streaming for large datasets like C4.

    For C4 dataset, uses streaming to avoid downloading 30-50GB of shards
    for small splits like train[:200]. Other datasets use standard download.

    Args:
        specification: Dataset configuration

    Returns:
        List of prompt strings

    Raises:
        ValueError: If C4 dataset is missing required config or sample count
        RuntimeError: If streaming fails due to network or other issues
    """
    # Check if dataset is a local path
    dataset_path = Path(specification.dataset)
    if dataset_path.exists() and dataset_path.is_dir():
        # Local dataset saved with save_to_disk()
        dataset_dict = load_from_disk(specification.dataset)
        dataset = dataset_dict[specification.split]
        return list(dataset[specification.column])

    # HuggingFace Hub dataset
    # Detect C4 dataset (massive, benefits from streaming)
    is_c4 = "c4" in specification.dataset.lower()

    if is_c4:
        # C4 is ~800GB - use streaming to avoid downloading shards
        # Parse split to extract sample count (e.g., "train[:200]" -> 200)
        sample_count = _parse_split_count(specification.split)

        # FAIL LOUDLY: C4 requires explicit sample count
        if sample_count is None:
            raise ValueError(
                f"C4 dataset requires explicit sample count in split. "
                f"Got: '{specification.split}'. "
                f"Expected format: 'train[:N]' or 'train[M:N]'"
            )

        # FAIL LOUDLY: C4 requires config parameter
        if not specification.config:
            raise ValueError(
                "C4 dataset requires config parameter (e.g., 'en'). "
                "Use --unhelpfulness-prompts.config en"
            )

        # Extract base split name (e.g., "train[:200]" -> "train")
        # Streaming datasets don't support slice notation in split parameter
        import re

        base_split = re.sub(r"\[.*\]", "", specification.split)

        # Load as streaming dataset
        try:
            dataset = load_dataset(
                specification.dataset,
                specification.config,
                split=base_split,  # Use base split without slice notation
                streaming=True,  # KEY: Stream instead of download
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to stream C4 dataset. Network required. Original error: {e}"
            ) from e

        # Take N samples from stream and materialize to list
        # This downloads only the data needed, not full shards
        prompts = []
        try:
            for i, example in enumerate(dataset):
                if i >= sample_count:
                    break
                prompts.append(example[specification.column])
        except Exception as e:
            raise RuntimeError(
                f"Failed to stream C4 examples (got {len(prompts)}/{sample_count}). "
                f"Check network connectivity. Original error: {e}"
            ) from e

        # FAIL LOUDLY: Verify we got expected count
        if len(prompts) < sample_count:
            raise ValueError(
                f"C4 stream exhausted early: got {len(prompts)}, expected {sample_count}"
            )

        return prompts
    else:
        # Other datasets: use existing download behavior
        if specification.config:
            dataset = load_dataset(
                specification.dataset, specification.config, split=specification.split
            )
        else:
            dataset = load_dataset(specification.dataset, split=specification.split)

        return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        # parameters is already a dict (serialized for SQLite storage)
        for name, value in parameters.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
