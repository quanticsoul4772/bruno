# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Direction extraction phase for heretic.

This module handles extracting refusal directions from model activations:
- PCA-based extraction (contrastive PCA)
- Supervised probing
- Ensemble (probe + PCA)
- Sacred direction extraction and orthogonalization
"""

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from ..config import Settings
from ..constants import LOW_MAGNITUDE_WARNING
from ..error_tracker import record_suppressed_error
from ..exceptions import SacredDirectionError
from ..model import Model, SacredDirectionResult
from ..utils import empty_cache, print


@dataclass
class DirectionExtractionResult:
    """Result from the direction extraction phase.

    Attributes:
        refusal_directions: Primary refusal directions (n_layers, hidden_dim)
        use_multi_direction: Whether multi-direction ablation should be used
        refusal_directions_multi: Optional multi-component directions
            (n_layers, n_components, hidden_dim)
        direction_weights: Weights for each direction component
        helpfulness_direction: Optional helpfulness direction for reference
        sacred_directions_result: Optional sacred direction extraction result
    """

    refusal_directions: Tensor
    use_multi_direction: bool
    refusal_directions_multi: Tensor | None = None
    direction_weights: list[float] | None = None
    helpfulness_direction: Tensor | None = None
    sacred_directions_result: SacredDirectionResult | None = None


def extract_refusal_directions(
    model: Model,
    good_prompts: list[str],
    bad_prompts: list[str],
    settings: Settings,
) -> tuple[Tensor, Tensor, DirectionExtractionResult]:
    """Extract refusal directions from good and bad prompts.

    This function handles the full direction extraction pipeline including:
    - Computing residuals for good and bad prompts
    - Extracting directions via PCA or mean difference
    - Computing eigenvalue-based weights if enabled

    Args:
        model: The loaded model instance
        good_prompts: List of harmless prompts
        bad_prompts: List of harmful prompts
        settings: Configuration settings

    Returns:
        Tuple of (good_residuals, bad_residuals, extraction_result)
        Residuals are returned for reuse by advanced extraction methods
    """
    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    direction_weights = settings.direction_weights
    refusal_directions_multi = None
    use_multi_direction = False

    if settings.use_pca_extraction and settings.n_refusal_directions > 1:
        print(
            f"* Extracting {settings.n_refusal_directions} refusal directions using contrastive PCA..."
        )
        pca_result = model.get_refusal_directions_pca(
            good_residuals,
            bad_residuals,
            n_components=settings.n_refusal_directions,
            alpha=settings.pca_alpha,
        )
        refusal_directions_multi = pca_result.directions

        if settings.use_eigenvalue_weights:
            direction_weights = pca_result.get_eigenvalue_weights(
                method=settings.eigenvalue_weight_method,
                temperature=settings.eigenvalue_weight_temperature,
            )
            print(
                f"  * Using eigenvalue-based weights ({settings.eigenvalue_weight_method}): "
                f"{[f'{w:.3f}' for w in direction_weights]}"
            )
        else:
            print(f"  * Using fixed weights: {settings.direction_weights}")

        # For compatibility, also compute mean-difference direction
        refusal_directions = F.normalize(
            bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
            p=2,
            dim=1,
        )
        use_multi_direction = True
    else:
        # Original single-direction approach
        refusal_directions = F.normalize(
            bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
            p=2,
            dim=1,
        )

    result = DirectionExtractionResult(
        refusal_directions=refusal_directions,
        use_multi_direction=use_multi_direction,
        refusal_directions_multi=refusal_directions_multi,
        direction_weights=direction_weights,
    )

    return good_residuals, bad_residuals, result


def apply_helpfulness_orthogonalization(
    model: Model,
    result: DirectionExtractionResult,
    helpful_prompts: list[str],
    unhelpful_prompts: list[str],
) -> DirectionExtractionResult:
    """Orthogonalize refusal direction against helpfulness direction.

    This preserves model helpfulness by removing any overlap between
    refusal and helpfulness directions.

    Args:
        model: The loaded model instance
        result: Current direction extraction result
        helpful_prompts: Prompts representing helpful behavior
        unhelpful_prompts: Prompts representing unhelpful/random text

    Returns:
        Updated DirectionExtractionResult with orthogonalized directions
    """
    print("* Extracting helpfulness direction for orthogonalization...")
    print(f"  * Loaded {len(helpful_prompts)} helpful prompts")
    print(f"  * Loaded {len(unhelpful_prompts)} unhelpful prompts")

    helpful_residuals = model.get_residuals_batched(helpful_prompts)
    unhelpful_residuals = model.get_residuals_batched(unhelpful_prompts)

    helpfulness_direction = model.extract_helpfulness_direction(
        helpful_residuals,
        unhelpful_residuals,
    )

    print("  * Orthogonalizing refusal direction against helpfulness...")
    refusal_directions = model.orthogonalize_direction(
        result.refusal_directions,
        helpfulness_direction,
    )

    del helpful_residuals, unhelpful_residuals

    return DirectionExtractionResult(
        refusal_directions=refusal_directions,
        use_multi_direction=result.use_multi_direction,
        refusal_directions_multi=result.refusal_directions_multi,
        direction_weights=result.direction_weights,
        helpfulness_direction=helpfulness_direction,
        sacred_directions_result=result.sacred_directions_result,
    )


def extract_sacred_directions(
    model: Model,
    result: DirectionExtractionResult,
    sacred_prompts: list[str],
    sacred_baseline_prompts: list[str],
    settings: Settings,
) -> DirectionExtractionResult:
    """Extract sacred directions and orthogonalize refusal against them.

    Sacred directions encode capabilities (e.g., MMLU reasoning) that should
    be preserved during ablation.

    Args:
        model: The loaded model instance
        result: Current direction extraction result
        sacred_prompts: Prompts encoding capabilities to preserve
        sacred_baseline_prompts: Baseline prompts for contrastive extraction
        settings: Configuration settings

    Returns:
        Updated DirectionExtractionResult with sacred direction orthogonalization
    """
    print()
    print("Extracting sacred (capability) directions...")

    try:
        print(f"* Loaded {len(sacred_prompts)} capability prompts")
        print(f"* Loaded {len(sacred_baseline_prompts)} baseline prompts")

        sacred_residuals = model.get_residuals_batched(sacred_prompts)
        baseline_residuals = model.get_residuals_batched(sacred_baseline_prompts)

        sacred_directions_result = model.extract_sacred_directions(
            sacred_residuals,
            baseline_residuals,
            n_directions=settings.n_sacred_directions,
        )

        print(f"* Extracted {settings.n_sacred_directions} sacred directions per layer")

        # Compute overlap before orthogonalization
        if result.refusal_directions.dim() == 3:
            overlap_dir = result.refusal_directions[:, 0, :]
        else:
            overlap_dir = result.refusal_directions

        overlap_stats = model.compute_sacred_overlap(
            overlap_dir, sacred_directions_result.directions
        )

        print(f"* Mean refusal-sacred overlap: {overlap_stats['mean_overlap']:.3f}")
        print(f"* Max refusal-sacred overlap: {overlap_stats['max_overlap']:.3f}")

        if overlap_stats["mean_overlap"] > settings.sacred_overlap_threshold:
            print(
                f"[yellow]Warning: High overlap ({overlap_stats['mean_overlap']:.3f}) between refusal and sacred directions. "
                f"Orthogonalization will remove significant refusal signal.[/yellow]"
            )

        # Orthogonalize refusal direction against sacred directions
        print("* Orthogonalizing refusal direction against sacred directions...")

        refusal_directions_multi = result.refusal_directions_multi
        if result.use_multi_direction and refusal_directions_multi is not None:
            refusal_directions_multi = model.orthogonalize_against_sacred(
                refusal_directions_multi,
                sacred_directions_result.directions,
            )

        refusal_directions = model.orthogonalize_against_sacred(
            result.refusal_directions,
            sacred_directions_result.directions,
        )

        # Check magnitude after orthogonalization
        if refusal_directions.dim() == 3:
            orthog_magnitude = refusal_directions.norm(dim=2).mean().item()
        else:
            orthog_magnitude = refusal_directions.norm(dim=1).mean().item()

        if orthog_magnitude < LOW_MAGNITUDE_WARNING:
            print(
                f"[yellow]Warning: Low direction magnitude ({orthog_magnitude:.3f}) after orthogonalization. "
                f"Ablation may be less effective. Consider reducing n_sacred_directions.[/yellow]"
            )
        else:
            print(
                f"* Direction magnitude after orthogonalization: {orthog_magnitude:.3f}"
            )

        del sacred_residuals, baseline_residuals
        empty_cache()

        return DirectionExtractionResult(
            refusal_directions=refusal_directions,
            use_multi_direction=result.use_multi_direction,
            refusal_directions_multi=refusal_directions_multi,
            direction_weights=result.direction_weights,
            helpfulness_direction=result.helpfulness_direction,
            sacred_directions_result=sacred_directions_result,
        )

    except SacredDirectionError as e:
        print(f"[red]Sacred direction error: {e}[/red]")
        print("[yellow]Continuing without sacred direction preservation...[/yellow]")
        empty_cache()
        record_suppressed_error(
            error=e,
            context="sacred_direction_orthogonalization",
            module="direction_extraction",
            severity="warning",
            details={"n_sacred_directions": settings.n_sacred_directions},
        )
        return result

    except Exception as e:
        print(
            f"[yellow]Warning: Failed to load sacred prompts: {e}. "
            f"Skipping sacred direction preservation.[/yellow]"
        )
        record_suppressed_error(
            error=e,
            context="sacred_direction_extraction",
            module="direction_extraction",
            severity="warning",
            details={"n_sacred_directions": settings.n_sacred_directions},
        )
        return result
