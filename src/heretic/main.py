# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

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
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData, get_token
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from pydantic import ValidationError
from questionary import Choice, Style
from rich.traceback import install

from .config import Settings
from .evaluator import Evaluator
from .model import (
    AbliterationParameters,
    ConceptConeExtractionResult,
    LayerRangeProfile,
    Model,
    SupervisedExtractionResult,
)
from .transfer import (
    detect_model_family,
    get_model_profile,
    get_warm_start_params,
    get_warm_start_variations,
)
from .utils import (
    BatchSizeError,
    empty_cache,
    format_duration,
    get_gpu_memory_info,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
)
from .validation import AbliterationValidator


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
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    model = Model(settings)

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

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
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
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

    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    # Phase 1: Support multi-direction PCA extraction
    direction_weights = settings.direction_weights  # Default fixed weights
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

        # Compute eigenvalue-based weights if enabled
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
        refusal_directions_multi = None
        use_multi_direction = False

    # Phase 5: Direction orthogonalization
    helpfulness_direction = None
    if settings.orthogonalize_directions:
        print("* Extracting helpfulness direction for orthogonalization...")
        helpful_prompts = load_prompts(settings.helpfulness_prompts)
        unhelpful_prompts = load_prompts(settings.unhelpfulness_prompts)
        print(f"  * Loaded {len(helpful_prompts)} helpful prompts")
        print(f"  * Loaded {len(unhelpful_prompts)} unhelpful prompts")

        helpful_residuals = model.get_residuals_batched(helpful_prompts)
        unhelpful_residuals = model.get_residuals_batched(unhelpful_prompts)

        helpfulness_direction = model.extract_helpfulness_direction(
            helpful_residuals,
            unhelpful_residuals,
        )

        # Orthogonalize the refusal direction
        print("  * Orthogonalizing refusal direction against helpfulness...")
        refusal_directions = model.orthogonalize_direction(
            refusal_directions,
            helpfulness_direction,
        )

        del helpful_residuals, unhelpful_residuals

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals
    empty_cache()

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
        # Need to reload residuals for supervised probing
        good_residuals_sp = model.get_residuals_batched(good_prompts)
        bad_residuals_sp = model.get_residuals_batched(bad_prompts)
        supervised_result = model.get_refusal_directions_supervised(
            good_residuals_sp,
            bad_residuals_sp,
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
            raise RuntimeError(
                f"Supervised probing failed (accuracy below {settings.min_probe_accuracy} or class imbalance). "
                "This can happen if the model doesn't clearly distinguish refusal vs compliance patterns. "
                "Try lowering min_probe_accuracy, or disable use_supervised_probing to use PCA instead."
            )
        del good_residuals_sp, bad_residuals_sp
        empty_cache()

    # Phase 2: Ensemble Probe + PCA
    if settings.ensemble_probe_pca:
        print("* Extracting refusal directions using ensemble (probe + PCA)...")
        good_residuals_ens = model.get_residuals_batched(good_prompts)
        bad_residuals_ens = model.get_residuals_batched(bad_prompts)
        supervised_directions = model.get_refusal_directions_ensemble(
            good_residuals_ens,
            bad_residuals_ens,
            bad_prompts,
            evaluator,
            probe_weight=settings.ensemble_weight_probe,
            pca_weight=settings.ensemble_weight_pca,
            min_probe_accuracy=settings.min_probe_accuracy,
        )
        # Note: get_refusal_directions_ensemble raises RuntimeError if supervised probing fails
        print("  * Ensemble directions extracted")
        del good_residuals_ens, bad_residuals_ens
        empty_cache()

    # Phase 4: Concept Cones
    if settings.use_concept_cones:
        print("* Extracting concept cones...")
        good_residuals_cc = model.get_residuals_batched(good_prompts)
        bad_residuals_cc = model.get_residuals_batched(bad_prompts)
        cc_result = model.get_refusal_directions_concept_cones(
            good_residuals_cc,
            bad_residuals_cc,
            n_cones=settings.n_concept_cones,
            min_cone_size=settings.min_cone_size,
            directions_per_cone=settings.directions_per_cone,
            min_silhouette_score=settings.min_silhouette_score,
        )
        if isinstance(cc_result, ConceptConeExtractionResult):
            concept_cone_result = cc_result
            print(f"  * Extracted {len(concept_cone_result.cones)} concept cones")
        else:
            raise RuntimeError(
                f"Concept cone extraction failed (silhouette score below {settings.min_silhouette_score}). "
                "This indicates poor clustering quality - harmful prompts may not have distinct categories. "
                "Try increasing min_cone_size or decreasing n_concept_cones, or disable use_concept_cones."
            )
        del good_residuals_cc, bad_residuals_cc
        empty_cache()

    # Phase 5: CAA - Extract compliance direction
    if settings.use_caa:
        print("* Extracting compliance direction for CAA...")
        bad_residuals_caa = model.get_residuals_batched(bad_prompts)
        caa_result = model.get_compliance_directions_from_responses(
            bad_prompts,
            bad_residuals_caa,
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
            raise RuntimeError(
                "CAA extraction failed due to insufficient samples. Need at least 10 refusals "
                "and 10 compliant responses from bad_prompts. Either your model already complies "
                "with most harmful prompts (nothing to ablate), or it refuses everything "
                "(no compliance examples). Disable use_caa or use a different model."
            )
        del bad_residuals_caa
        empty_cache()

    # Phase 6: Circuit Discovery
    if settings.use_circuit_ablation:
        # Check for GQA models upfront
        is_gqa, n_heads, n_kv_heads = model.is_gqa_model()
        if is_gqa:
            raise RuntimeError(
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
            raise RuntimeError(
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
                raise RuntimeError(
                    "Circuit ablation failed: GQA model detected at runtime. "
                    "This should have been caught earlier. Disable use_circuit_ablation."
                )
            elif circuit_result.circuits_ablated > 0:
                print(f"  * Ablated {circuit_result.circuits_ablated} circuits")
            else:
                raise RuntimeError(
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
    MAX_OOM_RETRIES = 3

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
            0.4 * (len(model.get_layers()) - 1),
            0.9 * (len(model.get_layers()) - 1),
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
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * (len(model.get_layers()) - 1),
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
                0.6 * (len(model.get_layers()) - 1),
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

            # Clear GPU memory
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

    # Create or load study with persistent storage for resume support
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
    except Exception as e:
        error_str = str(e).lower()
        if "database is locked" in error_str:
            print("[red]Database is locked. Another process may be using it.[/]")
            print("[yellow]If not, delete the .db file and restart:[/]")
            print(f"[yellow]  rm {settings.storage.replace('sqlite:///', '')}[/]")
            return
        elif "disk" in error_str or "corrupt" in error_str:
            print(f"[red]Database may be corrupted: {e}[/]")
            print("[yellow]Backup and delete the .db file to start fresh.[/]")
            return
        raise

    # Calculate remaining trials if resuming
    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining_trials = max(0, settings.n_trials - completed_trials)

    if completed_trials > 0:
        print()
        print(
            f"[cyan]Resuming from trial {completed_trials + 1} ({completed_trials} completed trials found)[/]"
        )
        trial_index = completed_trials

    # Phase 7: Enqueue warm-start trials if enabled and no trials completed yet
    if settings.use_warm_start_params and completed_trials == 0:
        components = model.get_abliterable_components()
        warm_start_params = get_warm_start_params(
            model_name=settings.model,
            n_layers=len(model.get_layers()),
            components=components,
            override_family=settings.model_family,
        )

        if warm_start_params is not None:
            # Get detected or overridden family
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

            # Generate variations for exploration
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
        else:
            detected = detect_model_family(settings.model)
            raise RuntimeError(
                f"Warm-start enabled but no profile found for model: {settings.model}. "
                f"Detected family: {detected or 'unknown'}. "
                f"Set --model-family to one of: llama, qwen, mistral, gemma, phi. "
                f"Or disable use_warm_start_params to use random initialization."
            )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print(f"[green]All {settings.n_trials} trials already completed![/]")

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

        print(f"Saving model to [bold]{save_directory}[/]...")
        model.model.save_pretrained(save_directory)
        model.tokenizer.save_pretrained(save_directory)
        print(f"[bold green]Model saved to {save_directory}[/]")

        # Auto-upload to HuggingFace if --hf-upload is specified
        if settings.hf_upload:
            print()
            print(f"Uploading model to HuggingFace: [bold]{settings.hf_upload}[/]...")
            try:
                token = get_token()
                if not token:
                    print(
                        "[red]No HuggingFace token found. Set HF_TOKEN env var or run 'huggingface-cli login'[/]"
                    )
                else:
                    model.model.push_to_hub(
                        settings.hf_upload,
                        private=settings.hf_private,
                        token=token,
                    )
                    model.tokenizer.push_to_hub(
                        settings.hf_upload,
                        private=settings.hf_private,
                        token=token,
                    )

                    # Upload model card if source model is from HuggingFace
                    if not Path(settings.model).exists():
                        card = ModelCard.load(settings.model)
                        if card.data is None:
                            card.data = ModelCardData()
                        if card.data.tags is None:
                            card.data.tags = []
                        card.data.tags.extend(
                            ["heretic", "uncensored", "decensored", "abliterated"]
                        )
                        card.text = (
                            get_readme_intro(
                                settings,
                                trial,
                                evaluator.base_refusals,
                                evaluator.bad_prompts,
                            )
                            + card.text
                        )
                        card.push_to_hub(settings.hf_upload, token=token)

                    print(f"[bold green]Model uploaded to {settings.hf_upload}[/]")
            except Exception as error:
                print(f"[red]Failed to upload to HuggingFace: {error}[/]")

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

                        print("Saving model...")
                        model.model.save_pretrained(save_directory)
                        model.tokenizer.save_pretrained(save_directory)
                        print(f"Model saved to [bold]{save_directory}[/].")

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

            except Exception as error:
                print(f"[red]Error: {error}[/]")


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
