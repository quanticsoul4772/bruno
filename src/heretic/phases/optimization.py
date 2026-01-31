# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Optimization phase for heretic.

This module handles Optuna-based optimization:
- Study creation and resumption
- Warm-start parameter injection
- Trial execution and scoring
"""

from typing import Callable

import optuna
from optuna import Study
from optuna.samplers import TPESampler
from optuna.study import StudyDirection

from ..config import Settings
from ..exceptions import ConfigurationError
from ..model import Model
from ..transfer import (
    detect_model_family,
    get_model_profile,
    get_warm_start_params,
    get_warm_start_variations,
)
from ..utils import print


def create_study(settings: Settings) -> Study:
    """Create or load an Optuna study with persistent storage.

    Args:
        settings: Configuration settings

    Returns:
        Optuna Study object, ready for optimization

    Raises:
        ConfigurationError: If study creation fails
    """
    try:
        study = optuna.create_study(
            study_name=settings.study_name,
            storage=settings.storage,
            load_if_exists=True,
            sampler=TPESampler(
                n_startup_trials=settings.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
            ),
            directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        )
        return study

    except OSError as e:
        error_str = str(e).lower()
        if "database is locked" in error_str or "locked" in error_str:
            print("[red]Database is locked. Another process may be using it.[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Wait for other process to finish")
            print("  2. Kill other heretic processes")
            print(
                f"  3. If stuck, delete: {settings.storage.replace('sqlite:///', '')}"
            )
            raise ConfigurationError(
                "Optuna database is locked. Another process may be using it."
            ) from e
        elif "disk" in error_str or "space" in error_str:
            print("[red]Insufficient Disk Space[/]")
            print("[yellow]Free up disk space and try again[/]")
            raise ConfigurationError(
                f"Insufficient disk space for Optuna database: {settings.storage}"
            ) from e
        elif "corrupt" in error_str:
            print(f"[red]Database may be corrupted: {e}[/]")
            print("[yellow]Backup and delete the .db file to start fresh.[/]")
            raise ConfigurationError(
                f"Optuna database corrupted: {settings.storage}"
            ) from e
        raise

    except Exception as e:
        print(f"[red]Failed to create Optuna study: {e}[/]")
        print("[yellow]Check storage path and permissions[/]")
        raise ConfigurationError(f"Failed to create Optuna study: {e}") from e


def enqueue_warm_start_trials(
    study: Study,
    settings: Settings,
    model: Model,
) -> bool:
    """Enqueue warm-start trials if enabled and no trials completed.

    Args:
        study: The Optuna study
        settings: Configuration settings
        model: The loaded model instance

    Returns:
        True if warm-start trials were enqueued, False otherwise
    """
    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    if not settings.use_warm_start_params or completed_trials > 0:
        return False

    components = model.get_abliterable_components()
    warm_start_params = get_warm_start_params(
        model_name=settings.model,
        n_layers=len(model.get_layers()),
        components=components,
        override_family=settings.model_family,
    )

    if warm_start_params is None:
        return False

    family = settings.model_family or detect_model_family(settings.model)
    profile = get_model_profile(settings.model, settings.model_family)

    print()
    print(f"[cyan]Warm-start enabled for model family: [bold]{family}[/][/]")
    if profile:
        print(f"  * Profile: {profile.description}")
        print(
            f"  * Recommended max_weight_position: {profile.recommended_max_weight_position:.0%} of layers"
        )
        print(f"  * Typical KL divergence: {profile.typical_kl_divergence:.2f}")

    variations = get_warm_start_variations(
        warm_start_params,
        n_variations=settings.warm_start_n_trials,
        n_layers=len(model.get_layers()),
    )

    print(f"  * Enqueuing {len(variations)} warm-start trials...")
    first_component = components[0] if components else None

    for i, params in enumerate(variations):
        study.enqueue_trial(params)
        if first_component:
            weight_key = f"{first_component}.max_weight"
            weight_val = params.get(weight_key, "N/A")
            weight_str = (
                f"{weight_val:.2f}"
                if isinstance(weight_val, (int, float))
                else str(weight_val)
            )
            if i == 0:
                print(f"    * Base trial: max_weight={weight_str}")
            else:
                print(f"    * Variation {i}: max_weight={weight_str}")

    return True


def run_optimization(
    study: Study,
    objective: Callable,
    n_trials: int,
) -> None:
    """Run the Optuna optimization loop.

    Args:
        study: The Optuna study
        objective: The objective function to optimize
        n_trials: Maximum number of trials to run
    """
    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining_trials = max(0, n_trials - completed_trials)

    if completed_trials > 0:
        print()
        print(
            f"[cyan]Resuming from trial {completed_trials + 1} ({completed_trials} completed trials found)[/]"
        )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print(f"[green]All {n_trials} trials already completed![/]")
