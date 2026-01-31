# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Model saving phase for heretic.

This module handles saving and uploading abliterated models:
- Local disk saving
- HuggingFace Hub upload
- Model card generation
"""

from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData, get_token
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


def _upload_model_card(
    settings: Settings,
    trial: Trial,
    evaluator: Evaluator,
    repo_id: str,
    token: str,
) -> None:
    """Upload a model card to HuggingFace Hub.

    Args:
        settings: Configuration settings
        trial: The selected Optuna trial
        evaluator: Evaluator instance for model card generation
        repo_id: HuggingFace repository ID
        token: HuggingFace token
    """
    card = ModelCard.load(settings.model)

    if card.data is None:
        card.data = ModelCardData()
    if card.data.tags is None:
        card.data.tags = []

    card.data.tags.extend(["heretic", "uncensored", "decensored", "abliterated"])

    card.text = (
        get_readme_intro(
            settings,
            trial,
            evaluator.base_refusals,
            evaluator.bad_prompts,
        )
        + card.text
    )

    card.push_to_hub(repo_id, token=token)
