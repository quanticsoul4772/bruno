# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Model saving phase for heretic.

This module handles saving and uploading abliterated models:
- Local disk saving
- HuggingFace Hub upload
- Model card generation
"""

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import ModelCard, ModelCardData, get_token

if TYPE_CHECKING:
    from ..validation import ValidationReport
from huggingface_hub.errors import HfHubHTTPError
from optuna import Trial

from ..config import Settings
from ..evaluator import Evaluator
from ..model import Model
from ..utils import get_readme_intro, print


def save_model_local(
    model: Model,
    save_directory: str,
) -> bool:
    """Save the abliterated model to a local directory.

    Args:
        model: The model to save
        save_directory: Path to save the model

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        print(f"Saving model to [bold]{save_directory}[/]...")
        model.model.save_pretrained(save_directory)
        model.tokenizer.save_pretrained(save_directory)
        print(f"[bold green]Model saved to {save_directory}[/]")
        return True

    except PermissionError:
        print(f"[red]Permission Denied: {save_directory}[/]")
        print("[yellow]Check directory permissions or choose different path[/]")
        return False

    except OSError as e:
        error_msg = str(e).lower()
        if "disk" in error_msg or "space" in error_msg:
            print("[red]Insufficient Disk Space[/]")
            print("[yellow]Free up space or choose different location[/]")
        else:
            print(f"[red]Failed to save model: {e}[/]")
        return False


def upload_model_huggingface(
    model: Model,
    settings: Settings,
    trial: Trial,
    evaluator: Evaluator,
    repo_id: str | None = None,
    private: bool | None = None,
    token: str | None = None,
    validation_report: "ValidationReport | None" = None,
    features_active: list[str] | None = None,
) -> bool:
    """Upload the abliterated model to HuggingFace Hub.

    Args:
        model: The model to upload
        settings: Configuration settings
        trial: The selected Optuna trial
        evaluator: Evaluator instance for model card generation
        repo_id: HuggingFace repository ID (defaults to settings.hf_upload)
        private: Whether to make the repo private (defaults to settings.hf_private)
        token: HuggingFace token (auto-detected if None)
        validation_report: Optional validation report with benchmarks
        features_active: Optional list of active feature names

    Returns:
        True if upload succeeded, False otherwise
    """
    repo_id = repo_id or settings.hf_upload
    private = private if private is not None else settings.hf_private
    token = token or get_token()

    if not repo_id:
        print("[red]No repository ID specified for upload[/]")
        return False

    if not token:
        print(
            "[red]No HuggingFace token found. Set HF_TOKEN env var or run 'huggingface-cli login'[/]"
        )
        return False

    print()
    print(f"Uploading model to HuggingFace: [bold]{repo_id}[/]...")

    try:
        model.model.push_to_hub(
            repo_id,
            private=private,
            token=token,
        )
        model.tokenizer.push_to_hub(
            repo_id,
            private=private,
            token=token,
        )

        # Upload model card if source model is from HuggingFace
        if not Path(settings.model).exists():
            _upload_model_card(
                settings=settings,
                trial=trial,
                evaluator=evaluator,
                repo_id=repo_id,
                token=token,
                validation_report=validation_report,
                features_active=features_active,
            )

        print(f"[bold green]Model uploaded to {repo_id}[/]")
        return True

    except HfHubHTTPError as error:
        if error.response.status_code == 401:
            print("[red]Authentication Failed[/]")
            print("[yellow]Run: huggingface-cli login[/]")
        elif error.response.status_code == 403:
            print(f"[red]Permission Denied: {repo_id}[/]")
            print("[yellow]Check repository permissions or create repository first[/]")
        elif error.response.status_code == 404:
            print(f"[red]Repository Not Found: {repo_id}[/]")
            print("[yellow]Create repository first on HuggingFace Hub[/]")
        else:
            print(f"[red]HuggingFace Hub Error: {error}[/]")
        return False

    except OSError as error:
        error_msg = str(error).lower()
        if "disk" in error_msg or "space" in error_msg:
            print("[red]Insufficient Disk Space[/]")
            print("[yellow]Free up space for model upload[/]")
        else:
            print(f"[red]File System Error: {error}[/]")
        return False

    except Exception as error:
        print(f"[red]Failed to upload to HuggingFace: {error}[/]")
        print("[yellow]Check network connection and repository permissions[/]")
        return False


def _get_model_tags(model_name: str, is_moe: bool = False) -> list[str]:
    """Generate dynamic tags based on model architecture and capabilities.

    Args:
        model_name: Name of the base model
        is_moe: Whether the model is a Mixture-of-Experts architecture

    Returns:
        List of relevant tags for HuggingFace discoverability
    """
    tags = ["heretic", "uncensored", "decensored", "abliterated", "text-generation"]
    name_lower = model_name.lower()

    # Architecture tags
    if "qwen" in name_lower:
        tags.append("qwen2")
    if "llama" in name_lower:
        tags.append("llama")
    if "deepseek" in name_lower:
        tags.append("deepseek")
    if "gemma" in name_lower:
        tags.append("gemma")
    if "phi" in name_lower:
        tags.append("phi")
    if "mistral" in name_lower:
        tags.append("mistral")
    if "moonlight" in name_lower:
        tags.append("moonlight")

    # Capability tags
    if "coder" in name_lower or "code" in name_lower:
        tags.append("code")
    if "instruct" in name_lower or "chat" in name_lower:
        tags.append("conversational")
    if is_moe or "moe" in name_lower:
        tags.append("moe")

    # Bruno-specific
    tags.append("optuna-optimized")
    tags.append("bruno")

    return tags


def _upload_model_card(
    settings: Settings,
    trial: Trial,
    evaluator: Evaluator,
    repo_id: str,
    token: str,
    validation_report: "ValidationReport | None" = None,
    features_active: list[str] | None = None,
) -> None:
    """Upload a model card to HuggingFace Hub.

    Args:
        settings: Configuration settings
        trial: The selected Optuna trial
        evaluator: Evaluator instance for model card generation
        repo_id: HuggingFace repository ID
        token: HuggingFace token
        validation_report: Optional validation report with benchmarks
        features_active: Optional list of active feature names
    """
    card = ModelCard.load(settings.model)

    if card.data is None:
        card.data = ModelCardData()
    if card.data.tags is None:
        card.data.tags = []

    # Dynamic tags based on model (Feature 2)
    is_moe = "moe" in settings.model.lower() or "moonlight" in settings.model.lower()
    new_tags = _get_model_tags(settings.model, is_moe)
    # Deduplicate tags
    card.data.tags = list(set(card.data.tags + new_tags))

    # YAML frontmatter metadata (Feature 2)
    card.data.base_model = settings.model
    card.data.pipeline_tag = "text-generation"
    card.data.language = ["en"]
    card.data.library_name = "transformers"

    # Rich README with benchmarks and usage examples (Features 1, 3, 7)
    card.text = (
        get_readme_intro(
            settings,
            trial,
            evaluator.base_refusals,
            evaluator.bad_prompts,
            validation_report=validation_report,
            features_active=features_active,
            repo_id=repo_id,
        )
        + card.text
    )

    card.push_to_hub(repo_id, token=token)
