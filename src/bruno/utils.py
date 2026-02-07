# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
import time
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .validation import ValidationReport

import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset, load_from_disk
from datasets.exceptions import DatasetNotFoundError
from optuna import Trial
from requests import HTTPError
from requests.exceptions import ConnectionError, Timeout
from rich.console import Console

from .config import DatasetSpecification, Settings
from .error_tracker import record_suppressed_error
from .exceptions import (
    BatchSizeError,
    DatasetConfigError,
    DatasetError,
    NetworkTimeoutError,
)
from .logging import get_logger

logger = get_logger(__name__)

print = Console(highlight=False).print


# Re-export BatchSizeError for backwards compatibility
__all__ = [
    "BatchSizeError",
    "print",
    "empty_cache",
    "load_prompts",
    "batchify",
    "get_gpu_memory_info",
    "retry_with_backoff",
]


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (ConnectionError, Timeout),
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3)
        def download_file(url):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s",
                            error=str(e),
                            func=func.__name__,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted",
                            error=str(e),
                            func=func.__name__,
                        )
            raise last_exception

        return wrapper

    return decorator


def _load_streaming_dataset(
    specification: "DatasetSpecification",
    sample_count: int,
    base_split: str,
) -> list[str]:
    """Load dataset using streaming with retry logic.

    Internal helper that handles the actual streaming with retries.
    """
    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries + 1):
        try:
            dataset = load_dataset(
                specification.dataset,
                specification.config,
                split=base_split,
                streaming=True,
            )

            prompts = []
            for i, example in enumerate(dataset):
                if i >= sample_count:
                    break
                prompts.append(example[specification.column])

            if len(prompts) < sample_count:
                raise ValueError(
                    f"Stream exhausted early: got {len(prompts)}, expected {sample_count}"
                )

            return prompts

        except (ConnectionError, Timeout) as e:
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), 60.0)
                logger.warning(
                    f"Network error, retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
                )
                time.sleep(delay)
            else:
                raise NetworkTimeoutError(
                    f"Network error after {max_retries} retries streaming '{specification.dataset}'. "
                    f"Check internet connection."
                ) from e
        except KeyError as e:
            raise DatasetConfigError(
                f"Column '{specification.column}' not found in dataset '{specification.dataset}'. "
                f"Check column name or list available columns."
            ) from e

    # Should never reach here, but satisfy type checker
    raise NetworkTimeoutError(
        f"Failed to load streaming dataset after {max_retries} retries"
    )


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
    except Exception as e:
        record_suppressed_error(
            error=e,
            context="get_gpu_memory_info",
            module="utils",
            severity="debug",
            details={"fallback": "returning zeros"},
        )
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
    logger.debug(
        "Loading prompts",
        dataset=specification.dataset,
        config=specification.config,
        split=specification.split,
        column=specification.column,
    )

    # Check if dataset is a local path
    dataset_path = Path(specification.dataset)
    if dataset_path.exists() and dataset_path.is_dir():
        # Local dataset saved with save_to_disk()
        logger.debug("Loading from local disk", path=str(dataset_path))
        dataset_dict = load_from_disk(specification.dataset)
        dataset = dataset_dict[specification.split]
        return list(dataset[specification.column])

    # HuggingFace Hub dataset
    # Detect C4 dataset (massive, benefits from streaming)
    # Use exact match to avoid false positives like "mc4", "c4de", etc.
    dataset_lower = specification.dataset.lower()
    is_c4 = dataset_lower in ("allenai/c4", "c4") or dataset_lower.endswith("/c4")

    if is_c4:
        # C4 is ~800GB - use streaming to avoid downloading shards
        # Parse split to extract sample count (e.g., "train[:200]" -> 200)
        sample_count = _parse_split_count(specification.split)

        # FAIL LOUDLY: C4 requires explicit sample count
        if sample_count is None:
            raise DatasetConfigError(
                f"C4 dataset requires explicit sample count in split. "
                f"Got: '{specification.split}'. "
                f"Expected format: 'train[:N]' or 'train[M:N]'"
            )

        # FAIL LOUDLY: C4 requires config parameter
        if not specification.config:
            raise DatasetConfigError(
                "C4 dataset requires config parameter (e.g., 'en'). "
                "Use --unhelpfulness-prompts.config en"
            )

        # Extract base split name (e.g., "train[:200]" -> "train")
        # Streaming datasets don't support slice notation in split parameter
        import re

        base_split = re.sub(r"\[.*\]", "", specification.split)

        # Load with retry logic
        try:
            return _load_streaming_dataset(specification, sample_count, base_split)
        except DatasetNotFoundError as e:
            print(f"[red]Dataset Not Found: {specification.dataset}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check dataset name spelling")
            print("  2. Search HuggingFace Hub: https://huggingface.co/datasets")
            raise DatasetError(
                f"Dataset '{specification.dataset}' not found. "
                f"Search: https://huggingface.co/datasets"
            ) from e
        except ValueError as e:
            # Config parameter error
            error_msg = str(e).lower()
            if "config" in error_msg or "variant" in error_msg:
                print(f"[red]Dataset Config Error: {specification.dataset}[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Add config parameter: --unhelpfulness-prompts.config en")
                print(
                    "  2. List available configs: from datasets import get_dataset_config_names"
                )
                print(f"     get_dataset_config_names('{specification.dataset}')")
                raise DatasetConfigError(
                    f"Dataset '{specification.dataset}' requires a config parameter. "
                    f"Try: --unhelpfulness-prompts.config en"
                ) from e
            raise
        except KeyboardInterrupt:
            raise
    else:
        # Other datasets: use existing download behavior
        try:
            if specification.config:
                dataset = load_dataset(
                    specification.dataset,
                    specification.config,
                    split=specification.split,
                )
            else:
                dataset = load_dataset(specification.dataset, split=specification.split)
        except (ConnectionError, Timeout) as e:
            print("[red]Network Error: Cannot download dataset[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check your internet connection")
            print("  2. Try again in a few minutes")
            print("  3. Check HuggingFace Hub status: https://status.huggingface.co")
            raise NetworkTimeoutError(
                f"Network timeout while downloading dataset '{specification.dataset}'. "
                f"Check internet connection and try again."
            ) from e
        except HTTPError as e:
            # Check for authentication or not found
            if e.response.status_code == 401:
                print(f"[red]Authentication Required: {specification.dataset}[/]")
                print("[yellow]Run: huggingface-cli login[/]")
                raise DatasetError(
                    f"Dataset '{specification.dataset}' requires authentication. "
                    f"Run 'huggingface-cli login' first."
                ) from e
            elif e.response.status_code == 404:
                print(f"[red]Dataset Not Found: {specification.dataset}[/]")
                print(
                    "[yellow]Search HuggingFace Hub: https://huggingface.co/datasets[/]"
                )
                raise DatasetError(
                    f"Dataset '{specification.dataset}' not found. "
                    f"Check name or search: https://huggingface.co/datasets"
                ) from e
            raise
        except DatasetNotFoundError as e:
            print(f"[red]Dataset Config Not Found: {specification.config}[/]")
            print(f"[yellow]Dataset: {specification.dataset}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check config name spelling")
            print(
                "  2. List available configs: from datasets import get_dataset_config_names"
            )
            print(f"     get_dataset_config_names('{specification.dataset}')")
            raise DatasetConfigError(
                f"Config '{specification.config}' not found for dataset '{specification.dataset}'. "
                f"Use get_dataset_config_names() to list available configs."
            ) from e
        except ValueError as e:
            # Check if split format error
            error_msg = str(e).lower()
            if "split" in error_msg:
                print(f"[red]Invalid Split Format: {specification.split}[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Use valid split format: 'train', 'test', 'validation'")
                print("  2. For slices: 'train[:100]' or 'train[10:20]'")
                print("  3. For percentages: 'train[:10%]'")
                raise DatasetConfigError(
                    f"Invalid split format: '{specification.split}'. "
                    f"Use format like 'train', 'train[:100]', or 'train[:10%]'"
                ) from e
            elif "config" in error_msg:
                print(f"[red]Dataset Config Required: {specification.dataset}[/]")
                print(
                    "[yellow]Add config parameter (e.g., --unhelpfulness-prompts.config en)[/]"
                )
                raise DatasetConfigError(
                    f"Dataset '{specification.dataset}' requires a config parameter. "
                    f"Try adding: --unhelpfulness-prompts.config <config_name>"
                ) from e
            raise
        except OSError as e:
            # Check if disk space error
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg or "storage" in error_msg:
                print("[red]Insufficient Disk Space[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Free up disk space")
                print(
                    "  2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/datasets"
                )
                print(
                    "  3. Use smaller dataset split (e.g., train[:100] instead of train[:1000])"
                )
                raise DatasetError(
                    f"Insufficient disk space to download dataset '{specification.dataset}'. "
                    f"Free up space or use smaller split."
                ) from e
            raise
        except KeyboardInterrupt:
            raise

        try:
            return list(dataset[specification.column])
        except KeyError as e:
            print(f"[red]Column Not Found: {specification.column}[/]")
            print(f"[yellow]Dataset: {specification.dataset}[/]")
            print("[yellow]Available columns:[/]")
            print(f"  {list(dataset.column_names)}")
            raise DatasetConfigError(
                f"Column '{specification.column}' not found in dataset '{specification.dataset}'. "
                f"Available columns: {list(dataset.column_names)}"
            ) from e


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
    validation_report: "ValidationReport | None" = None,
    features_active: list[str] | None = None,
    repo_id: str | None = None,
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"
    bruno_version = version("bruno-ai")

    # Build technique description from active features
    technique_parts = ["Optuna-optimized abliteration"]
    if features_active:
        feature_map = {
            "mpoa": "MPOA (Norm-Preserving Biprojected Abliteration)",
            "sacred_directions": "sacred direction preservation",
            "neural_refusal_detection": "neural refusal detection",
            "activation_calibration": "activation calibration",
            "concept_cones": "concept cone extraction",
            "caa": "Contrastive Activation Addition",
            "ensemble_probe_pca": "ensemble probe+PCA extraction",
        }
        for feature in features_active:
            if feature in feature_map:
                technique_parts.append(feature_map[feature])

    technique_desc = (
        ", ".join(technique_parts) if len(technique_parts) > 1 else technique_parts[0]
    )

    # Start building the README
    sections = []

    # Title and badge
    sections.append(f"""# {settings.model.split("/")[-1]}-bruno

> Made with [Bruno](https://github.com/quanticsoul4772/bruno) v{bruno_version} ({technique_desc})

This is an abliterated version of {model_link} with reduced refusals while preserving capabilities.
""")

    # Bruno Score (Feature 7)
    if validation_report and isinstance(validation_report, ValidationReport):
        if validation_report.baseline and validation_report.post_abliteration:
            bruno_score = validation_report.compute_bruno_score()
            if bruno_score > 0:
                # Compute component scores
                cap_pct = (
                    1.0
                    - min(validation_report.kl_divergence_increase / 5.0, 0.5)
                    - max(-validation_report.mmlu_change, 0.0)
                ) * 100
                removal_pct = validation_report.refusal_reduction_pct

                sections.append(f"""## Bruno Score: {bruno_score:.1f} / 100

> {cap_pct:.1f}% capability preserved | {removal_pct:.1f}% refusals removed

""")

    # Benchmark table (if validation report available)
    if validation_report and isinstance(validation_report, ValidationReport):
        if validation_report.baseline and validation_report.post_abliteration:
            base_mmlu = validation_report.baseline.mmlu_scores
            post_mmlu = validation_report.post_abliteration.mmlu_scores

            if base_mmlu and post_mmlu:
                sections.append("## Benchmarks\n")
                sections.append("| Benchmark | This Model | Original Model | Delta |")
                sections.append("| :-------- | :--------: | :------------: | :---: |")

                # MMLU average first
                base_avg = validation_report.baseline.mmlu_average
                post_avg = validation_report.post_abliteration.mmlu_average
                delta_avg = post_avg - base_avg
                delta_str = f"{delta_avg:+.1f}%" if delta_avg != 0 else "0.0%"
                sections.append(
                    f"| **MMLU (average)** | {post_avg:.1f}% | {base_avg:.1f}% | {delta_str} |"
                )

                # Individual categories
                for category in sorted(base_mmlu.keys()):
                    if category in post_mmlu:
                        base_score = base_mmlu[category] * 100
                        post_score = post_mmlu[category] * 100
                        delta = post_score - base_score
                        delta_str = f"{delta:+.1f}%" if delta != 0 else "0.0%"
                        cat_name = category.replace("_", " ").title()
                        sections.append(
                            f"| {cat_name} | {post_score:.1f}% | {base_score:.1f}% | {delta_str} |"
                        )

                # Extended benchmarks (Feature 6)
                base_ext = validation_report.baseline.extended_benchmarks
                post_ext = validation_report.post_abliteration.extended_benchmarks

                if base_ext and post_ext:
                    for bench_name in sorted(base_ext.keys()):
                        if bench_name in post_ext:
                            base_score = base_ext[bench_name] * 100
                            post_score = post_ext[bench_name] * 100
                            delta = post_score - base_score
                            delta_str = f"{delta:+.1f}%" if delta != 0 else "0.0%"
                            sections.append(
                                f"| **{bench_name}** | {post_score:.1f}% | {base_score:.1f}% | {delta_str} |"
                            )

                sections.append("")

    # Performance summary
    sections.append("## Performance\n")

    kl_div = trial.user_attrs.get("kl_divergence", 0)
    refusals = trial.user_attrs.get("refusals", 0)

    # Compute capability preservation note
    capability_note = "good" if kl_div < 1.0 else "moderate"

    sections.append("| Metric | This Model | Original Model |")
    sections.append("| :----- | :--------: | :------------: |")
    sections.append(
        f"| **KL Divergence** | {kl_div:.2f} *({capability_note})* | 0.00 *(baseline)* |"
    )
    sections.append(
        f"| **Refusals** | {refusals}/{len(bad_prompts)} | {base_refusals}/{len(bad_prompts)} |"
    )

    # Refusal reduction percentage
    if base_refusals > 0:
        reduction_pct = ((base_refusals - refusals) / base_refusals) * 100
        sections.append(
            f"| **Refusal Reduction** | {reduction_pct:.1f}% | 0% *(baseline)* |"
        )

    sections.append("")

    # Usage examples (Feature 3)
    if repo_id:
        is_chat = any(kw in settings.model.lower() for kw in ["instruct", "chat"])

        sections.append("## Usage\n")

        if is_chat:
            sections.append(
                """### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = \""""
                + repo_id
                + """\"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
            )
        else:
            sections.append(
                """### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = \""""
                + repo_id
                + """\"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
            )

        sections.append(
            """### Using vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model=\""""
            + repo_id
            + """\")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```

### Using Ollama

```bash
# Pull the model (if GGUF available)
ollama pull """
            + repo_id
            + """

# Or create from local GGUF
cat > Modelfile <<EOF
FROM ./"""
            + repo_id.split("/")[-1]
            + """.gguf
EOF
ollama create my-model -f Modelfile

# Run
ollama run my-model
```

## Recommended Settings

This model has been abliterated (refusals removed). For best results:

**Temperature:**
- Creative writing: 0.9-1.2
- Coding: 0.6-0.8
- General use: 0.7-0.9

**Sampling:**
- top_p: 0.9-0.95
- top_k: 40-50
- repetition_penalty: 1.05-1.15

**Context:**
- Minimum: 2048 tokens
- Recommended: 4096-8192 tokens

**Abliteration Notes:**
- Model responds more directly without moralizing
- May need explicit direction for sensitive topics
- Respects standard chat formatting and system prompts

"""
        )

    # Abliteration parameters
    sections.append("## Abliteration Parameters\n")
    sections.append("| Parameter | Value |")
    sections.append("| :-------- | :---: |")
    for name, value in get_trial_parameters(trial).items():
        sections.append(f"| **{name}** | {value} |")
    sections.append("")

    # Disclaimer
    sections.append("""## Disclaimer

This model has been modified to reduce refusals. It may generate content that would normally be filtered. Use responsibly and in compliance with applicable laws and regulations.

-----

""")

    return "\n".join(sections)
