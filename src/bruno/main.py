# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Main entry point for heretic abliteration.

This module contains the run() function which orchestrates the full abliteration
pipeline. The workflow is divided into logical phases that can be found in the
heretic.phases submodule:

1. **Dataset Loading** (phases.dataset_loading)
   - Load good/bad prompts for direction extraction
   - Load helpfulness prompts for orthogonalization
   - Load sacred prompts for capability preservation

2. **Direction Extraction** (phases.direction_extraction)
   - Extract refusal directions via PCA or supervised probing
   - Orthogonalize against helpfulness direction
   - Extract and orthogonalize sacred capability directions

3. **Optimization** (phases.optimization)
   - Create/resume Optuna study
   - Enqueue warm-start trials from model family profiles
   - Run multi-objective optimization loop

4. **Model Saving** (phases.model_saving)
   - Save abliterated model to disk
   - Upload to HuggingFace Hub with model card

For backwards compatibility and robustness, the core logic remains in this file,
but the phase modules provide cleaner interfaces for testing and extension.
"""

import math
import os
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import huggingface_hub
import optuna
import questionary
import torch
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData, get_token
from huggingface_hub.errors import HfHubHTTPError
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from pydantic import ValidationError
from questionary import Choice, Style
from rich.traceback import install

from .config import Settings
from .constants import (
    MAX_OOM_RETRIES,
    LayerPos,
    Memory,
)
from .error_tracker import error_tracker, record_suppressed_error
from .evaluator import Evaluator
from .exceptions import (
    AbliterationError,
    BatchSizeError,
    CAAExtractionError,
    ConceptConeError,
    ModelInferenceError,
    SupervisedProbeError,
    WarmStartError,
)
from .logging import get_logger
from .model import (
    AbliterationParameters,
    ConceptConeExtractionResult,
    LayerRangeProfile,
    Model,
    SupervisedExtractionResult,
)
from .phases import (
    apply_helpfulness_orthogonalization,
    create_study,
    enqueue_warm_start_trials,
    extract_refusal_directions,
    extract_sacred_directions,
    load_datasets,
    run_optimization,
    save_model_local,
    upload_model_huggingface,
)
from .transfer import detect_model_family
from .utils import (
    empty_cache,
    format_duration,
    get_gpu_memory_info,
    get_readme_intro,
    get_trial_parameters,
    print,
)
from .validation import AbliterationValidator

logger = get_logger(__name__)


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # An odd number of arguments have been passed (argv[0] is the program name),
        # so that after accounting for "--param VALUE" pairs, there is one left over.
        len(sys.argv) % 2 == 0
        # The leftover argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        settings = Settings()
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        print(f"GPU type: [bold]{torch.cuda.get_device_name()}[/]")
    elif is_xpu_available():
        print(f"XPU type: [bold]{torch.xpu.get_device_name()}[/]")
    elif is_mlu_available():
        print(f"MLU type: [bold]{torch.mlu.get_device_name()}[/]")
    elif is_sdaa_available():
        print(f"SDAA type: [bold]{torch.sdaa.get_device_name()}[/]")
    elif is_musa_available():
        print(f"MUSA type: [bold]{torch.musa.get_device_name()}[/]")
    elif is_npu_available():
        print(f"CANN version: [bold]{torch.version.cann}[/]")
    elif torch.backends.mps.is_available():
        print("GPU type: [bold]Apple Metal (MPS)[/]")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = Memory.DYNAMO_CACHE_SIZE_LIMIT

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    model = Model(settings)

    # Phase 1: Load all datasets using phase module
    datasets = load_datasets(settings)
    good_prompts = datasets.good_prompts
    bad_prompts = datasets.bad_prompts

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except torch.cuda.OutOfMemoryError as error:
                if batch_size == 1:
                    # Even a batch size of 1 causes OOM - cannot recover
                    print()
                    print("[red]GPU Out of Memory with batch size 1[/]")
                    print("[yellow]Solutions:[/]")
                    print("  1. Use smaller model or quantized version")
                    print("  2. Reduce max_response_length in config")
                    print("  3. Free up GPU memory (close other programs)")
                    raise ModelInferenceError(
                        "GPU out of memory even with batch_size=1. "
                        "Model may be too large for available GPU memory."
                    ) from error

                print("[yellow]GPU OOM[/] (batch too large)")
                break
            except RuntimeError as error:
                # Other runtime errors during inference
                error_msg = str(error).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    # Catch OOM that doesn't raise torch.cuda.OutOfMemoryError
                    if batch_size == 1:
                        print()
                        print("[red]Out of Memory with batch size 1[/]")
                        print(
                            "[yellow]Try smaller model or reduce max_response_length[/]"
                        )
                        raise ModelInferenceError(
                            f"Out of memory with batch_size=1: {error}"
                        ) from error
                    print("[yellow]OOM[/] (batch too large)")
                    break

                if batch_size == 1:
                    # Unrecoverable runtime error
                    raise
                print(f"[red]Failed[/] ({error})")
                break
            except KeyboardInterrupt:
                # CRITICAL: Always let user interrupt
                raise
            except Exception as error:
                # Unexpected error - reraise if batch_size=1, otherwise break
                if batch_size == 1:
                    raise
                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    evaluator = Evaluator(settings, model)

    # Initialize validation framework if enabled
    validator = None
    if settings.enable_validation:
        validator = AbliterationValidator(settings, model, evaluator)
        validator.establish_baseline()

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reload_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    # Phase 2: Extract refusal directions using phase module
    good_residuals, bad_residuals, direction_result = extract_refusal_directions(
        model=model,
        good_prompts=good_prompts,
        bad_prompts=bad_prompts,
        settings=settings,
    )

    # Unpack direction extraction results
    refusal_directions = direction_result.refusal_directions
    refusal_directions_multi = direction_result.refusal_directions_multi
    use_multi_direction = direction_result.use_multi_direction
    direction_weights = direction_result.direction_weights or settings.direction_weights

    # Apply helpfulness orthogonalization if enabled
    if (
        settings.orthogonalize_directions
        and datasets.helpful_prompts
        and datasets.unhelpful_prompts
    ):
        direction_result = apply_helpfulness_orthogonalization(
            model=model,
            result=direction_result,
            helpful_prompts=datasets.helpful_prompts,
            unhelpful_prompts=datasets.unhelpful_prompts,
        )
        refusal_directions = direction_result.refusal_directions

    # Sacred Direction Preservation using phase module
    if (
        settings.use_sacred_directions
        and datasets.sacred_prompts
        and datasets.sacred_baseline_prompts
    ):
        direction_result = extract_sacred_directions(
            model=model,
            result=direction_result,
            sacred_prompts=datasets.sacred_prompts,
            sacred_baseline_prompts=datasets.sacred_baseline_prompts,
            settings=settings,
        )
        refusal_directions = direction_result.refusal_directions
        refusal_directions_multi = direction_result.refusal_directions_multi

    # NOTE: We keep good_residuals and bad_residuals in memory for reuse by
    # advanced extraction methods (supervised probing, ensemble, concept cones, CAA).
    # This avoids redundant GPU computation - a 40-60% speedup for extraction phase.

    # Convert layer profile configs to LayerRangeProfile objects
    layer_profiles: list[LayerRangeProfile] | None = None
    if settings.enable_layer_profiles and settings.layer_range_profiles:
        layer_profiles = [
            LayerRangeProfile(
                range_start=p.range_start,
                range_end=p.range_end,
                weight_multiplier=p.weight_multiplier,
            )
            for p in settings.layer_range_profiles
        ]
        print()
        print("Layer-range profiles enabled:")
        for p in layer_profiles:
            print(
                f"  * Layers {p.range_start:.0%}-{p.range_end:.0%}: "
                f"weight multiplier = {p.weight_multiplier:.2f}"
            )

    # Store additional extraction results for advanced features
    supervised_directions = None
    concept_cone_result = None
    compliance_direction = None
    refusal_circuits = None

    # Phase 2: Supervised Probing
    if settings.use_supervised_probing and not settings.ensemble_probe_pca:
        print("* Extracting refusal directions using supervised probing...")
        # Reuse cached residuals instead of recomputing
        supervised_result = model.get_refusal_directions_supervised(
            good_residuals,
            bad_residuals,
            bad_prompts,
            evaluator,
            min_probe_accuracy=settings.min_probe_accuracy,
        )
        if isinstance(supervised_result, SupervisedExtractionResult):
            supervised_directions = supervised_result.directions
            print(
                f"  * Supervised probe succeeded (mean accuracy: {supervised_result.get_mean_accuracy():.2f})"
            )
        else:
            raise SupervisedProbeError(
                f"Supervised probing failed (accuracy below {settings.min_probe_accuracy} or class imbalance). "
                "This can happen if the model doesn't clearly distinguish refusal vs compliance patterns. "
                "Try lowering min_probe_accuracy, or disable use_supervised_probing to use PCA instead."
            )

    # Phase 2: Ensemble Probe + PCA
    if settings.ensemble_probe_pca:
        print("* Extracting refusal directions using ensemble (probe + PCA)...")
        # Reuse cached residuals instead of recomputing
        supervised_directions = model.get_refusal_directions_ensemble(
            good_residuals,
            bad_residuals,
            bad_prompts,
            evaluator,
            probe_weight=settings.ensemble_weight_probe,
            pca_weight=settings.ensemble_weight_pca,
            min_probe_accuracy=settings.min_probe_accuracy,
        )
        # Note: get_refusal_directions_ensemble raises RuntimeError if supervised probing fails
        print("  * Ensemble directions extracted")

    # Phase 4: Concept Cones
    if settings.use_concept_cones:
        print("* Extracting concept cones...")
        # Reuse cached residuals instead of recomputing
        cc_result = model.get_refusal_directions_concept_cones(
            good_residuals,
            bad_residuals,
            n_cones=settings.n_concept_cones,
            min_cone_size=settings.min_cone_size,
            directions_per_cone=settings.directions_per_cone,
            min_silhouette_score=settings.min_silhouette_score,
        )
        if isinstance(cc_result, ConceptConeExtractionResult):
            concept_cone_result = cc_result
            print(f"  * Extracted {len(concept_cone_result.cones)} concept cones")
        else:
            raise ConceptConeError(
                f"Concept cone extraction failed (silhouette score below {settings.min_silhouette_score}). "
                "This indicates poor clustering quality - harmful prompts may not have distinct categories. "
                "Try increasing min_cone_size or decreasing n_concept_cones, or disable use_concept_cones."
            )

    # Phase 5: CAA - Extract compliance direction
    if settings.use_caa:
        print("* Extracting compliance direction for CAA...")
        # Reuse cached bad_residuals instead of recomputing
        caa_result = model.get_compliance_directions_from_responses(
            bad_prompts,
            bad_residuals,
            evaluator,
        )
        if caa_result is not None:
            compliant_residuals, refusing_residuals = caa_result
            compliance_direction = model.extract_compliance_direction(
                compliant_residuals,
                refusing_residuals,
            )
            print("  * Compliance direction extracted")
        else:
            raise CAAExtractionError(
                "CAA extraction failed due to insufficient samples. Need at least 10 refusals "
                "and 10 compliant responses from bad_prompts. Either your model already complies "
                "with most harmful prompts (nothing to ablate), or it refuses everything "
                "(no compliance examples). Disable use_caa or use a different model."
            )

    # Now we can release the cached residuals - all extraction methods are done
    del good_residuals, bad_residuals
    empty_cache()

    # Phase 6: Circuit Discovery
    if settings.use_circuit_ablation:
        # Check for GQA models upfront
        is_gqa, n_heads, n_kv_heads = model.is_gqa_model()
        if is_gqa:
            raise AbliterationError(
                f"Circuit ablation is not supported for GQA (Grouped Query Attention) models. "
                f"Detected: n_heads={n_heads}, n_kv_heads={n_kv_heads}. "
                f"GQA models include Llama 3.x, Qwen2.5, and others. "
                f"Disable use_circuit_ablation to use standard layer-level ablation instead."
            )
        print("* Discovering refusal circuits...")
        refusal_circuits = model.get_or_discover_circuits(
            refusal_directions,
            n_circuits=settings.n_refusal_circuits,
            importance_threshold=settings.circuit_importance_threshold,
            cache_path=settings.circuit_cache_path,
        )
        if refusal_circuits:
            print(f"  * Discovered {len(refusal_circuits)} refusal circuits")
        else:
            raise AbliterationError(
                f"No refusal circuits found with importance threshold {settings.circuit_importance_threshold}. "
                f"Try lowering circuit_importance_threshold or disable use_circuit_ablation."
            )

    def restore_and_abliterate_trial(
        trial_params: dict[str, AbliterationParameters],
        direction_index: float | None,
        skip_reload: bool = False,
    ) -> None:
        """Restore model and apply abliteration based on trial parameters.

        This helper function centralizes the abliteration logic to avoid duplication
        across validation, auto-select, and interactive modes.
        """
        if not skip_reload:
            model.reload_model()

        # Get residuals for activation calibration if needed
        calibrated_params = trial_params
        if settings.use_activation_calibration:
            print("  * Computing activation statistics for calibration...")
            bad_residuals_cal = model.get_residuals_batched(bad_prompts)
            activation_stats = model.compute_refusal_activation_stats(
                refusal_directions,
                bad_residuals_cal,
                target_layer_frac=settings.activation_calibration_layer_frac,
            )
            calibrated_params = model.get_calibrated_parameters(
                trial_params,
                activation_stats,
                target_percentile=settings.activation_target_percentile,
                min_factor=settings.activation_calibration_min_factor,
                max_factor=settings.activation_calibration_max_factor,
            )
            del bad_residuals_cal
            empty_cache()

        # Phase 6: Circuit-level ablation (if enabled and has circuits)
        if settings.use_circuit_ablation and refusal_circuits:
            print("  * Applying circuit-level ablation...")
            circuit_result = model.abliterate_circuits(
                refusal_circuits,
                refusal_directions,
                ablation_strength=calibrated_params[
                    list(calibrated_params.keys())[0]
                ].max_weight,
            )
            if circuit_result.gqa_detected:
                raise AbliterationError(
                    "Circuit ablation failed: GQA model detected at runtime. "
                    "This should have been caught earlier. Disable use_circuit_ablation."
                )
            elif circuit_result.circuits_ablated > 0:
                print(f"  * Ablated {circuit_result.circuits_ablated} circuits")
            else:
                raise AbliterationError(
                    f"Circuit ablation failed: 0 circuits ablated, {circuit_result.circuits_skipped} skipped. "
                    "Check circuit_importance_threshold setting."
                )

        # Phase 4: Concept Cones ablation
        if settings.use_concept_cones and concept_cone_result is not None:
            print("  * Applying concept cone ablation...")
            model.abliterate_concept_cones(
                concept_cone_result,
                calibrated_params,
                direction_weights=direction_weights if use_multi_direction else None,
                layer_profiles=layer_profiles,
            )
        # Phase 5: CAA ablation
        elif settings.use_caa and compliance_direction is not None:
            print("  * Applying ablation with CAA...")
            # Use supervised directions if available, otherwise standard
            ablation_directions = (
                supervised_directions
                if supervised_directions is not None
                else refusal_directions
            )
            model.abliterate_with_caa(
                ablation_directions,
                compliance_direction,
                calibrated_params,
                removal_strength=1.0,
                addition_strength=settings.caa_addition_strength,
                max_overlap=settings.caa_max_overlap,
                layer_profiles=layer_profiles,
            )
        # Phase 4: Iterative refinement
        elif settings.iterative_rounds > 1:
            model.abliterate_iterative(
                good_prompts,
                bad_prompts,
                calibrated_params,
                max_rounds=settings.iterative_rounds,
                min_direction_magnitude=settings.min_direction_magnitude,
                max_kl_per_round=settings.max_kl_per_round,
                base_logprobs=evaluator.base_logprobs
                if settings.kl_divergence_tokens == 1
                else None,
                kl_check_prompts=evaluator.good_prompts
                if settings.kl_divergence_tokens == 1
                else None,
                layer_profiles=layer_profiles,
            )
        # Phase 2: Supervised directions (if extracted)
        elif supervised_directions is not None:
            print("  * Using supervised probe directions...")
            model.abliterate(
                supervised_directions,
                None,  # Per-layer mode for supervised directions
                calibrated_params,
                layer_profiles,
            )
        # Phase 1: Multi-direction ablation
        elif use_multi_direction and refusal_directions_multi is not None:
            model.abliterate_multi_direction(
                refusal_directions_multi,
                direction_weights,
                calibrated_params,
                layer_profiles,
            )
        # Original single-direction ablation
        else:
            model.abliterate(
                refusal_directions,
                direction_index,
                calibrated_params,
                layer_profiles,
            )

    trial_index = 0
    start_time = time.perf_counter()
    oom_count = 0

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            LayerPos.DIRECTION_INDEX_MIN_FRAC * (len(model.get_layers()) - 1),
            LayerPos.DIRECTION_INDEX_MAX_FRAC * (len(model.get_layers()) - 1),
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                LayerPos.MAX_WEIGHT_MIN_FRAC
                if hasattr(LayerPos, "MAX_WEIGHT_MIN_FRAC")
                else 0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                LayerPos.MAX_WEIGHT_POSITION_MIN_FRAC * (len(model.get_layers()) - 1),
                len(model.get_layers()) - 1,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                LayerPos.MIN_WEIGHT_DISTANCE_MIN_FRAC * (len(model.get_layers()) - 1),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        # Convert AbliterationParameters to dicts for JSON serialization (required for SQLite storage)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Reloading model...")
        print("* Abliterating...")
        restore_and_abliterate_trial(parameters, direction_index, skip_reload=False)

        print("* Evaluating...")
        try:
            score, kl_divergence, refusals = evaluator.get_score()
        except torch.cuda.OutOfMemoryError as e:
            nonlocal oom_count
            oom_count += 1

            # Clear GPU memory aggressively
            empty_cache()
            try:
                model.reload_model()  # Reload to reset model state
            except Exception as reload_error:
                # Track but don't fail - model state may be inconsistent
                record_suppressed_error(
                    error=reload_error,
                    context="oom_recovery_model_reload",
                    module="main",
                    severity="warning",
                    details={
                        "oom_count": oom_count,
                        "advice": "Model state may be inconsistent, subsequent trials may produce unexpected results",
                    },
                )
            empty_cache()

            mem_info = get_gpu_memory_info()
            print(f"[red]GPU OOM detected (attempt {oom_count}/{MAX_OOM_RETRIES})[/]")
            print(
                f"[yellow]GPU Memory: {mem_info['used_gb']:.1f}/{mem_info['total_gb']:.1f} GB used[/]"
            )

            if oom_count >= MAX_OOM_RETRIES:
                print(
                    f"[red]Repeated OOM errors ({oom_count}). Progress saved via Optuna storage.[/]"
                )
                print(
                    f"[yellow]Resume with: heretic {settings.model} --storage {settings.storage}[/]"
                )
                raise BatchSizeError(
                    f"Out of GPU memory after {oom_count} attempts. "
                    "Try reducing batch_size or max_batch_size, or use a GPU with more VRAM."
                ) from e

            # Reduce batch size and signal retry
            if settings.batch_size > 1:
                settings.batch_size = max(1, settings.batch_size // 2)
                print(
                    f"[yellow]Reducing batch_size to {settings.batch_size} and retrying...[/]"
                )

            raise  # Re-raise to let Optuna handle the failed trial

        # Note: trial.report() and trial.should_prune() are not supported for multi-objective
        # optimization in Optuna. Pruning is disabled for now.
        #
        # Design note: Switching to single-objective optimization with a weighted score
        # would enable pruning (~30% compute savings), but at the cost of losing the
        # Pareto front diversity. The current multi-objective approach is preferred
        # because it gives users more choices among different refusal/KL trade-offs.

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / trial_index) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    # Phase 3: Create or load study using phase module
    study = create_study(settings)

    # Update trial_index if resuming
    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if completed_trials > 0:
        trial_index = completed_trials

    # Phase 7: Enqueue warm-start trials using phase module
    warm_start_enqueued = enqueue_warm_start_trials(study, settings, model)
    if (
        settings.use_warm_start_params
        and not warm_start_enqueued
        and completed_trials == 0
    ):
        # Warm-start was requested but no profile found
        detected = detect_model_family(settings.model)
        raise WarmStartError(
            f"Warm-start enabled but no profile found for model: {settings.model}. "
            f"Detected family: {detected or 'unknown'}. "
            f"Set --model-family to one of: llama, qwen, mistral, gemma, phi. "
            f"Or disable use_warm_start_params to use random initialization."
        )

    # Run optimization loop using phase module
    run_optimization(study, objective, settings.n_trials)

    best_trials = sorted(
        study.best_trials,
        key=lambda trial: trial.user_attrs["refusals"],
    )

    choices = [
        Choice(
            title=(
                f"[Trial {trial.user_attrs['index']:>3}] "
                f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence: {trial.user_attrs['kl_divergence']:.2f}"
            ),
            value=trial,
        )
        for trial in best_trials
    ]

    choices.append(
        Choice(
            title="None (exit program)",
            value="",
        )
    )

    # Run post-abliteration validation if enabled
    if validator is not None:
        # We need to restore the best trial for validation
        if best_trials:
            best_trial = best_trials[0]
            parameters = {
                k: AbliterationParameters(**v)
                for k, v in best_trial.user_attrs["parameters"].items()
            }
            print("* Restoring best trial for validation...")
            restore_and_abliterate_trial(
                parameters,
                best_trial.user_attrs["direction_index"],
            )
            validator.measure_post_abliteration()
            validator.print_summary()

            # Save validation report if enabled
            if settings.save_validation_report:
                report = validator.get_report()
                if settings.validation_report_path:
                    report_path = settings.validation_report_path
                else:
                    model_name = Path(settings.model).name.replace("/", "_")
                    timestamp = int(time.time())
                    report_path = f"./validation_report_{model_name}_{timestamp}.json"
                report.save(report_path)
                print(f"Validation report saved to [bold]{report_path}[/]")

    print()
    print("[bold green]Optimization finished![/]")
    print()

    # Auto-select mode: automatically pick best trial and save without prompts
    if settings.auto_select:
        if not best_trials:
            print("[red]No Pareto-optimal trials found. Cannot auto-select.[/]")
            return

        # Select the best trial (first one = lowest refusals)
        trial = best_trials[0]
        print(
            f"[cyan]Auto-selecting trial {trial.user_attrs['index']} "
            f"(Refusals: {trial.user_attrs['refusals']}, "
            f"KL divergence: {trial.user_attrs['kl_divergence']:.2f})[/]"
        )

        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        # Convert parameters dicts back to AbliterationParameters
        parameters = {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        }
        print("* Reloading model...")
        print("* Abliterating...")
        restore_and_abliterate_trial(
            parameters,
            trial.user_attrs["direction_index"],
        )

        # Determine save path
        if settings.auto_select_path:
            save_directory = settings.auto_select_path
        else:
            # Default to model name with -heretic suffix
            auto_model_name = Path(settings.model).name
            save_directory = f"./{auto_model_name}-heretic"

        # Phase 4: Save model using phase module
        save_model_local(model, save_directory)

        # Auto-upload to HuggingFace if --hf-upload is specified
        if settings.hf_upload:
            upload_model_huggingface(
                model=model,
                settings=settings,
                trial=trial,
                evaluator=evaluator,
            )

        return

    # Interactive mode
    print(
        (
            "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
            "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
            "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
            "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
        )
    )

    while True:
        print()
        trial = questionary.select(
            "Which trial do you want to use?",
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()

        if trial is None or trial == "":
            break

        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        # Convert parameters dicts back to AbliterationParameters
        parameters = {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        }
        print("* Reloading model...")
        print("* Abliterating...")
        restore_and_abliterate_trial(
            parameters,
            trial.user_attrs["direction_index"],
        )

        while True:
            print()
            action = questionary.select(
                "What do you want to do with the decensored model?",
                choices=[
                    "Save the model to a local folder",
                    "Upload the model to Hugging Face",
                    "Chat with the model",
                    "Nothing (return to trial selection menu)",
                ],
                style=Style([("highlighted", "reverse")]),
            ).ask()

            if action is None or action == "Nothing (return to trial selection menu)":
                break

            # All actions are wrapped in a try/except block so that if an error occurs,
            # another action can be tried, instead of the program crashing and losing
            # the optimized model.
            try:
                match action:
                    case "Save the model to a local folder":
                        save_directory = questionary.path("Path to the folder:").ask()
                        if not save_directory:
                            continue

                        # Use phase module for saving
                        save_model_local(model, save_directory)

                    case "Upload the model to Hugging Face":
                        # We don't use huggingface_hub.login() because that stores the token on disk,
                        # and since this program will often be run on rented or shared GPU servers,
                        # it's better to not persist credentials.
                        token = get_token()
                        if not token:
                            token = questionary.password(
                                "Hugging Face access token:"
                            ).ask()
                        if not token:
                            continue

                        user = huggingface_hub.whoami(token)
                        fullname = user.get(
                            "fullname",
                            user.get("name", "unknown user"),
                        )
                        email = user.get("email", "no email found")
                        print(f"Logged in as [bold]{fullname} ({email})[/]")

                        repo_id = questionary.text(
                            "Name of repository:",
                            default=f"{user['name']}/{Path(settings.model).name}-heretic",
                        ).ask()

                        visibility = questionary.select(
                            "Should the repository be public or private?",
                            choices=[
                                "Public",
                                "Private",
                            ],
                            style=Style([("highlighted", "reverse")]),
                        ).ask()
                        private = visibility == "Private"

                        print("Uploading model...")

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

                        # If the model path doesn't exist locally, it can be assumed
                        # to be a model hosted on the Hugging Face Hub, in which case
                        # we can retrieve the model card.
                        if not Path(settings.model).exists():
                            card = ModelCard.load(settings.model)
                            if card.data is None:
                                card.data = ModelCardData()
                            if card.data.tags is None:
                                card.data.tags = []
                            card.data.tags.append("heretic")
                            card.data.tags.append("uncensored")
                            card.data.tags.append("decensored")
                            card.data.tags.append("abliterated")
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

                        print(f"Model uploaded to [bold]{repo_id}[/].")

                    case "Chat with the model":
                        print()
                        print(
                            "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                        )

                        chat = [
                            {"role": "system", "content": settings.system_prompt},
                        ]

                        while True:
                            try:
                                message = questionary.text(
                                    "User:",
                                    qmark=">",
                                ).unsafe_ask()
                                if not message:
                                    break
                                chat.append({"role": "user", "content": message})

                                print("[bold]Assistant:[/] ", end="")
                                response = model.stream_chat_response(chat)
                                chat.append({"role": "assistant", "content": response})
                            except (KeyboardInterrupt, EOFError):
                                # Ctrl+C/Ctrl+D
                                break
            except HfHubHTTPError as error:
                # HuggingFace Hub errors (upload, authentication, etc.)
                if error.response.status_code == 401:
                    print("[red]Authentication Failed[/]")
                    print("[yellow]Check HuggingFace token[/]")
                elif error.response.status_code == 403:
                    print("[red]Permission Denied[/]")
                    print("[yellow]Check repository access permissions[/]")
                else:
                    print(f"[red]HuggingFace Error: {error}[/]")
            except torch.cuda.OutOfMemoryError:
                print("[red]GPU Out of Memory during chat[/]")
                print("[yellow]Try shorter messages or restart chat[/]")
            except ModelInferenceError as error:
                print(f"[red]Model Inference Error: {error}[/]")
            except KeyboardInterrupt:
                # User interrupted - propagate
                raise
            except Exception as error:
                # Safety net for unexpected errors in interactive menu
                # Don't crash and lose the optimized model
                print(f"[red]Error: {error}[/]")
                print("[yellow]You can try again or select a different action[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
    finally:
        # Print suppressed error summary if any errors were tracked
        if error_tracker.count() > 0:
            print()
            print("[dim]" + "=" * 50 + "[/dim]")
            print(f"[dim]{error_tracker.get_summary()}[/dim]")
