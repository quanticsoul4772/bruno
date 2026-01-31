# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Cross-Model Hyperparameter Transfer for faster Optuna convergence.

This module provides model family profiles with known good hyperparameters
that can be used as warm-start points for Optuna optimization.

IMPORTANT: This transfers HYPERPARAMETERS only, NOT directions.
Directions cannot be transferred between models with different hidden dimensions.
"""

from dataclasses import dataclass

from .error_tracker import record_suppressed_error
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelAbliterationProfile:
    """Transferable hyperparameters (NOT directions) for a model family.

    These are learned from successful abliterations across multiple models
    in the same family. They provide good starting points for Optuna.

    Attributes:
        model_family: Name of the model family (e.g., 'llama', 'qwen')
        recommended_max_weight: Typical max ablation weight
        recommended_max_weight_position: Layer position for max weight (as fraction 0.0-1.0)
        recommended_min_weight_frac: Min weight as fraction of max weight
        recommended_min_weight_distance: Distance from max position (as fraction 0.0-1.0)
        attn_vs_mlp_ratio: Relative ablation strength for attention vs MLP
        typical_kl_divergence: Expected KL divergence at good ablation
        description: Human-readable description of the profile
    """

    model_family: str
    recommended_max_weight: float
    recommended_max_weight_position: float  # As fraction of layers (0.0-1.0)
    recommended_min_weight_frac: float  # min_weight = max_weight * this
    recommended_min_weight_distance: float  # As fraction of layers (0.0-1.0)
    attn_vs_mlp_ratio: float  # MLP weight = attn weight * this
    typical_kl_divergence: float
    description: str = ""


# Profiles learned from successful abliterations across various models
# These represent good starting points for Optuna optimization
MODEL_PROFILES: dict[str, ModelAbliterationProfile] = {
    "llama": ModelAbliterationProfile(
        model_family="llama",
        recommended_max_weight=1.1,
        recommended_max_weight_position=0.65,
        recommended_min_weight_frac=0.3,
        recommended_min_weight_distance=0.35,
        attn_vs_mlp_ratio=1.0,
        typical_kl_divergence=0.3,
        description="Llama family (Llama 2, Llama 3, Llama 3.1, Llama 3.2)",
    ),
    "qwen": ModelAbliterationProfile(
        model_family="qwen",
        recommended_max_weight=1.15,
        recommended_max_weight_position=0.60,
        recommended_min_weight_frac=0.25,
        recommended_min_weight_distance=0.30,
        attn_vs_mlp_ratio=1.1,
        typical_kl_divergence=0.35,
        description="Qwen family (Qwen, Qwen2, Qwen2.5)",
    ),
    "mistral": ModelAbliterationProfile(
        model_family="mistral",
        recommended_max_weight=1.05,
        recommended_max_weight_position=0.70,
        recommended_min_weight_frac=0.35,
        recommended_min_weight_distance=0.40,
        attn_vs_mlp_ratio=0.9,
        typical_kl_divergence=0.25,
        description="Mistral family (Mistral, Mixtral)",
    ),
    "gemma": ModelAbliterationProfile(
        model_family="gemma",
        recommended_max_weight=1.0,
        recommended_max_weight_position=0.62,
        recommended_min_weight_frac=0.4,
        recommended_min_weight_distance=0.32,
        attn_vs_mlp_ratio=1.05,
        typical_kl_divergence=0.28,
        description="Gemma family (Gemma, Gemma 2)",
    ),
    "phi": ModelAbliterationProfile(
        model_family="phi",
        recommended_max_weight=0.95,
        recommended_max_weight_position=0.68,
        recommended_min_weight_frac=0.35,
        recommended_min_weight_distance=0.38,
        attn_vs_mlp_ratio=0.95,
        typical_kl_divergence=0.32,
        description="Phi family (Phi-2, Phi-3)",
    ),
}


def detect_model_family(model_name: str) -> str | None:
    """Detect the model family from a model name or path.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        Model family name if detected, None otherwise
    """
    model_lower = model_name.lower()

    # Check each family (order matters - more specific patterns first)
    family_patterns = {
        "llama": ["llama", "meta-llama"],
        "qwen": ["qwen"],
        "mistral": ["mistral", "mixtral"],
        "gemma": ["gemma"],
        "phi": ["phi-", "phi2", "phi3", "phi-2", "phi-3"],
    }

    for family, patterns in family_patterns.items():
        for pattern in patterns:
            if pattern in model_lower:
                logger.debug(
                    "Detected model family",
                    model=model_name,
                    family=family,
                    matched_pattern=pattern,
                )
                return family

    record_suppressed_error(
        error=None,
        context="detect_model_family",
        module="transfer",
        severity="debug",
        details={
            "model": model_name,
            "checked_families": list(family_patterns.keys()),
            "reason": "No matching family pattern found",
        },
    )
    return None


def get_model_profile(
    model_name: str, override_family: str = ""
) -> ModelAbliterationProfile | None:
    """Get the abliteration profile for a model.

    Args:
        model_name: HuggingFace model ID or local path
        override_family: If set, use this family instead of auto-detecting

    Returns:
        ModelAbliterationProfile if found, None otherwise
    """
    if override_family:
        family = override_family.lower()
    else:
        family = detect_model_family(model_name)

    if family is None:
        logger.debug(
            "No model profile available - family not detected",
            model=model_name,
        )
        return None

    profile = MODEL_PROFILES.get(family)
    if profile is None:
        record_suppressed_error(
            error=None,
            context="get_model_profile",
            module="transfer",
            severity="debug",
            details={
                "model": model_name,
                "family": family,
                "available_profiles": list(MODEL_PROFILES.keys()),
                "reason": "Family not in profiles",
            },
        )
    return profile


def get_warm_start_params(
    model_name: str,
    n_layers: int,
    components: list[str],
    override_family: str = "",
) -> dict | None:
    """Get warm-start parameters for Optuna based on model family.

    This does NOT transfer directions, just hyperparameters.
    The parameters are converted to the actual layer indices based on n_layers.

    Args:
        model_name: HuggingFace model ID or local path
        n_layers: Number of layers in the model
        components: List of abliterable component names (e.g., ['self_attn', 'mlp'])
        override_family: If set, use this family instead of auto-detecting

    Returns:
        Dictionary of warm-start parameters for Optuna trial, or None if no profile found
    """
    profile = get_model_profile(model_name, override_family)

    if profile is None:
        record_suppressed_error(
            error=None,
            context="get_warm_start_params",
            module="transfer",
            severity="debug",
            details={
                "model": model_name,
                "override_family": override_family or "(auto-detect)",
                "reason": "No profile found for model",
            },
        )
        return None

    # Convert fractional positions to actual layer indices
    max_weight_position = profile.recommended_max_weight_position * (n_layers - 1)
    min_weight_distance = profile.recommended_min_weight_distance * (n_layers - 1)

    params: dict[str, float | str] = {
        "direction_scope": "per layer",  # Usually best for warm-start
        "direction_index": 0.65 * (n_layers - 1),  # Will be ignored for "per layer"
    }

    # Set parameters for each component
    for component in components:
        # Adjust MLP weight if different ratio
        weight_multiplier = 1.0
        if "mlp" in component.lower():
            weight_multiplier = profile.attn_vs_mlp_ratio

        params[f"{component}.max_weight"] = (
            profile.recommended_max_weight * weight_multiplier
        )
        params[f"{component}.max_weight_position"] = max_weight_position
        params[f"{component}.min_weight"] = profile.recommended_min_weight_frac
        params[f"{component}.min_weight_distance"] = min_weight_distance

    return params


def get_warm_start_variations(
    base_params: dict,
    n_variations: int,
    n_layers: int,
) -> list[dict]:
    """Generate variations of warm-start parameters for exploration.

    Creates slight perturbations around the base parameters to explore
    the neighborhood during warm-start trials.

    Args:
        base_params: Base warm-start parameters from get_warm_start_params
        n_variations: Number of variations to generate (including the base)
        n_layers: Number of layers in the model

    Returns:
        List of parameter dictionaries including the base and variations
    """
    if n_variations <= 1:
        return [base_params]

    variations = [base_params]  # First one is the base

    # Perturbation factors for variations
    perturbations = [
        {"weight": 0.95, "position": 0.98},  # Slightly more conservative
        {"weight": 1.05, "position": 1.02},  # Slightly more aggressive
        {"weight": 1.0, "position": 0.95},  # Earlier peak
        {"weight": 1.0, "position": 1.05},  # Later peak
    ]

    for perturb in perturbations:
        if len(variations) >= n_variations:
            break

        variant = base_params.copy()

        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                if "max_weight" in key and "position" not in key:
                    # Perturb weight
                    new_val = value * perturb["weight"]
                    # Clamp to valid range
                    variant[key] = max(0.8, min(1.5, new_val))
                elif "position" in key:
                    # Perturb position
                    new_val = value * perturb["position"]
                    # Clamp to valid range
                    variant[key] = max(0.6 * (n_layers - 1), min(n_layers - 1, new_val))

        variations.append(variant)

    return variations
