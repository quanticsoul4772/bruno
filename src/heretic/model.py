# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import copy
import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class LayerRangeProfile:
    """Defines abliteration strength for a specific layer range.

    Research shows different layer ranges encode different aspects of refusal:
    - Early layers (0.0-0.4): Basic representations, light abliteration preserves capabilities
    - Middle layers (0.4-0.7): Semantic "what to refuse", primary abliteration target
    - Late layers (0.7-1.0): Behavioral "how to refuse", moderate abliteration

    Attributes:
        range_start: Start of the layer range as fraction of total layers (0.0-1.0)
        range_end: End of the layer range as fraction of total layers (0.0-1.0)
        weight_multiplier: Multiplier applied to abliteration weights in this range
    """

    range_start: float  # 0.0-1.0 (fraction of total layers)
    range_end: float  # 0.0-1.0 (fraction of total layers)
    weight_multiplier: float  # Applied to computed abliteration weight


@dataclass
class PCAExtractionResult:
    """Result from PCA-based refusal direction extraction.

    Contains both the directions and their associated eigenvalues,
    enabling eigenvalue-based weighting for multi-direction ablation.
    """

    directions: Tensor  # Shape: (n_layers, n_components, hidden_dim)
    eigenvalues: Tensor  # Shape: (n_layers, n_components) - per-layer eigenvalues

    def get_eigenvalue_weights(
        self,
        method: str = "softmax",
        temperature: float = 1.0,
    ) -> list[float]:
        """Compute direction weights from eigenvalues.

        Args:
            method: Weight computation method:
                - "softmax": Apply softmax to mean eigenvalues (default)
                - "proportional": Weights proportional to mean eigenvalues
                - "log_proportional": Weights proportional to log of mean eigenvalues
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            List of weights, one per direction component
        """
        # Average eigenvalues across layers
        mean_eigenvalues = self.eigenvalues.mean(dim=0)  # Shape: (n_components,)

        # Ensure positive values (eigenvalues from contrastive PCA can be negative)
        # We use max(0, x) + epsilon to handle negative eigenvalues
        positive_eigenvalues = torch.clamp(mean_eigenvalues, min=1e-8)

        if method == "softmax":
            # Softmax normalization (sum to 1, smooth distribution)
            weights = F.softmax(positive_eigenvalues / temperature, dim=0)
        elif method == "proportional":
            # Direct proportional weights (sum to 1)
            weights = positive_eigenvalues / positive_eigenvalues.sum()
        elif method == "log_proportional":
            # Log-proportional (reduces dominance of largest eigenvalue)
            log_eigenvalues = torch.log1p(positive_eigenvalues)
            weights = log_eigenvalues / log_eigenvalues.sum()
        else:
            raise ValueError(f"Unknown eigenvalue weight method: {method}")

        # Scale so that the first (largest) weight is 1.0
        # This maintains compatibility with existing weight semantics
        if weights[0] > 0:
            weights = weights / weights[0]

        return weights.tolist()


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    torch_dtype=dtype,
                    device_map=settings.device_map,
                )

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            self.loaded_dtype = dtype  # Store successful dtype for potential reload
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

        # Cache original weights in memory for fast reset (avoids reloading from disk)
        # Disabled for very large models that don't fit in memory twice
        if settings.cache_weights:
            print("* Caching original weights in memory...")
            self.original_state_dict = copy.deepcopy(self.model.state_dict())
        else:
            print("* Weight caching disabled (will reload from disk between trials)")
            self.original_state_dict = None

        # Optionally compile the model for faster inference (~1.5-2x speedup)
        if settings.compile:
            print("* Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def reload_model(self):
        # Fast weight reset from cached state_dict (instead of reloading from disk)
        # This is ~5-10x faster than from_pretrained() for large models
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
        else:
            # Reload from disk when weight caching is disabled
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                torch_dtype=self.loaded_dtype,
                device_map=self.settings.device_map,
            )

    def get_layers(self) -> ModuleList:
        # Most multimodal models.
        with suppress(Exception):
            return self.model.model.language_model.layers

        # Text-only models.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight)

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear.weight)

        # We need at least one MLP down-projection.
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

    def get_layer_multiplier(
        self,
        layer_index: int,
        num_layers: int,
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ) -> float:
        """Get the weight multiplier for a specific layer based on layer range profiles.

        Args:
            layer_index: The index of the current layer
            num_layers: Total number of layers in the model
            layer_profiles: List of LayerRangeProfile defining multipliers per range

        Returns:
            Weight multiplier for this layer (default 1.0 if no profile matches)
        """
        if layer_profiles is None or len(layer_profiles) == 0:
            return 1.0

        # Convert layer index to relative position (0.0 to 1.0)
        relative_position = layer_index / max(num_layers - 1, 1)

        # Find matching profile (first match wins)
        for profile in layer_profiles:
            if profile.range_start <= relative_position < profile.range_end:
                return profile.weight_multiplier

        # Handle edge case where relative_position == 1.0 (last layer)
        if relative_position >= 1.0:
            for profile in layer_profiles:
                if profile.range_end >= 1.0:
                    return profile.weight_multiplier

        return 1.0  # Default if no profile matches

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ):
        if direction_index is None:
            refusal_direction = None
            global_projector = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )
            # Pre-compute projector for global direction (reused across all layers)
            global_projector = torch.outer(
                refusal_direction,
                refusal_direction,
            ).to(self.model.dtype)

        # Cache layer matrices to avoid repeated lookups (structure doesn't change)
        num_layers = len(self.get_layers())
        layer_matrices_cache = {
            i: self.get_layer_matrices(i) for i in range(num_layers)
        }

        # Pre-allocate projectors per device to avoid repeated .to() allocations
        # This prevents memory accumulation that causes OOM on multi-GPU setups
        device_projectors_cache: dict[torch.device, Tensor] = {}

        def get_device_projector(projector: Tensor, device: torch.device) -> Tensor:
            if device not in device_projectors_cache:
                device_projectors_cache[device] = projector.to(device)
            return device_projectors_cache[device]

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(num_layers):
            # Compute per-layer projector once per layer (outside component loop)
            if global_projector is not None:
                projector = global_projector
            else:
                # Per-layer direction: compute projector for this layer
                # The index must be shifted by 1 because the first element
                # of refusal_directions is the direction for the embeddings.
                layer_refusal_direction = refusal_directions[layer_index + 1]
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)
                # Clear the per-layer projector cache since projector changed
                device_projectors_cache.clear()

            for component, matrices in layer_matrices_cache[layer_index].items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                # Apply layer-range multiplier for surgical targeting
                layer_multiplier = self.get_layer_multiplier(
                    layer_index, num_layers, layer_profiles
                )
                weight = weight * layer_multiplier

                for matrix in matrices:
                    # Reuse cached projector per device to prevent memory accumulation
                    device_projector = get_device_projector(projector, matrix.device)
                    # In-place subtraction is safe as we're not using Autograd.
                    matrix.sub_(weight * (device_projector @ matrix))

            # Clear CUDA cache periodically to prevent memory fragmentation
            if layer_index % 8 == 7:
                empty_cache()

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )

    def get_responses(
        self, prompts: list[str], max_tokens: int | None = None
    ) -> list[str]:
        if max_tokens is None:
            max_tokens = self.settings.max_response_length

        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=max_tokens,
        )

        # Return only the newly generated part.
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(
        self, prompts: list[str], max_tokens: int | None = None
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch, max_tokens=max_tokens):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[str], n_tokens: int = 1) -> Tensor:
        """Get log probabilities for first n_tokens.

        Args:
            prompts: List of prompts to process
            n_tokens: Number of tokens to generate (default 1 for backward compatibility)

        Returns:
            Tensor of shape (n_prompts, vocab_size) if n_tokens=1,
            or (n_prompts, n_tokens, vocab_size) if n_tokens>1
        """
        _, outputs = self.generate(
            prompts,
            max_new_tokens=n_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if n_tokens == 1:
            # Original behavior: return shape (prompt, vocab_size)
            logits = outputs.scores[0]
            return F.log_softmax(logits, dim=-1)
        else:
            # Multi-token: return shape (prompt, n_tokens, vocab_size)
            # outputs.scores is a tuple of (n_prompts, vocab_size) tensors
            logits = torch.stack(outputs.scores, dim=1)
            return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str], n_tokens: int = 1) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch, n_tokens=n_tokens))

        return torch.cat(logprobs, dim=0)

    # Phase 1: Multi-Direction PCA Extraction
    def get_refusal_directions_pca(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        n_components: int = 3,
        alpha: float = 1.0,
    ) -> PCAExtractionResult:
        """Extract multiple refusal directions using TRUE Contrastive PCA.

        Finds directions that maximize variance in bad residuals while
        minimizing variance in good residuals. This is done by computing
        eigenvectors of (Σ_bad - α*Σ_good).

        Args:
            good_residuals: Shape (n_good, n_layers, hidden_dim)
            bad_residuals: Shape (n_bad, n_layers, hidden_dim)
            n_components: Number of principal components to extract
            alpha: Weight for good covariance subtraction (default 1.0)

        Returns:
            PCAExtractionResult containing:
                - directions: Tensor of shape (n_layers, n_components, hidden_dim)
                - eigenvalues: Tensor of shape (n_layers, n_components)
        """
        n_layers = good_residuals.shape[1]
        directions = []
        all_eigenvalues = []

        for layer_idx in range(n_layers):
            good_layer = good_residuals[:, layer_idx, :].cpu().numpy()
            bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

            # Center the data
            good_centered = good_layer - good_layer.mean(axis=0)
            bad_centered = bad_layer - bad_layer.mean(axis=0)

            # Compute covariance matrices
            # Add small regularization for numerical stability
            n_good = good_centered.shape[0]
            n_bad = bad_centered.shape[0]

            cov_good = (good_centered.T @ good_centered) / max(n_good - 1, 1)
            cov_bad = (bad_centered.T @ bad_centered) / max(n_bad - 1, 1)

            # Contrastive covariance: directions high-variance in bad, low-variance in good
            cov_contrastive = cov_bad - alpha * cov_good

            # Eigendecomposition (eigenvectors with largest eigenvalues)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_contrastive)
            except np.linalg.LinAlgError:
                # Fallback to mean difference if eigendecomposition fails
                diff = bad_layer.mean(axis=0) - good_layer.mean(axis=0)
                diff = diff / (np.linalg.norm(diff) + 1e-8)
                layer_directions = torch.from_numpy(
                    np.tile(diff, (n_components, 1))
                ).float()
                directions.append(layer_directions)
                # Use uniform eigenvalues as fallback
                all_eigenvalues.append(torch.ones(n_components))
                continue

            # Sort by eigenvalue descending, take top n_components
            idx = np.argsort(eigenvalues)[::-1][:n_components]
            top_eigenvalues = eigenvalues[idx]  # Shape: (n_components,)
            top_directions = eigenvectors[:, idx].T  # Shape: (n_components, hidden_dim)

            # Normalize each direction
            layer_directions = torch.from_numpy(top_directions).float()
            layer_directions = F.normalize(layer_directions, p=2, dim=1)
            directions.append(layer_directions)

            # Store eigenvalues for this layer
            layer_eigenvalues = torch.from_numpy(top_eigenvalues.copy()).float()
            all_eigenvalues.append(layer_eigenvalues)

        return PCAExtractionResult(
            directions=torch.stack(directions),
            eigenvalues=torch.stack(all_eigenvalues),
        )

    # Phase 5: Direction Orthogonalization
    def extract_helpfulness_direction(
        self,
        helpful_residuals: Tensor,
        unhelpful_residuals: Tensor,
    ) -> Tensor:
        """Extract direction that encodes 'helpfulness'.

        Uses contrast between helpful and unhelpful/low-quality responses.

        Args:
            helpful_residuals: Residuals from helpful prompts (n_samples, n_layers, hidden_dim)
            unhelpful_residuals: Residuals from unhelpful prompts (n_samples, n_layers, hidden_dim)

        Returns:
            Helpfulness direction tensor of shape (n_layers, hidden_dim)
        """
        helpfulness_direction = F.normalize(
            helpful_residuals.mean(dim=0) - unhelpful_residuals.mean(dim=0),
            p=2,
            dim=1,
        )
        return helpfulness_direction

    def orthogonalize_direction(
        self,
        target_direction: Tensor,
        remove_direction: Tensor,
    ) -> Tensor:
        """Remove component of remove_direction from target_direction.

        Args:
            target_direction: Direction to modify, shape (n_layers, hidden_dim)
            remove_direction: Direction to remove, shape (n_layers, hidden_dim)

        Returns:
            Orthogonalized direction, shape (n_layers, hidden_dim)
        """
        # Compute dot product per layer
        dot_product = (target_direction * remove_direction).sum(dim=1, keepdim=True)

        # Compute projection
        projection = dot_product * remove_direction

        # Remove projection and renormalize
        orthogonal = target_direction - projection
        return F.normalize(orthogonal, p=2, dim=1)

    # Phase 4: Iterative Refinement
    def abliterate_iterative(
        self,
        good_prompts: list[str],
        bad_prompts: list[str],
        parameters: dict[str, "AbliterationParameters"],
        max_rounds: int = 2,
        min_direction_magnitude: float = 0.1,
        max_kl_per_round: float = 0.5,
        base_logprobs: Tensor | None = None,
        kl_check_prompts: list[str] | None = None,
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ) -> tuple[int, list[float]]:
        """Iteratively extract and ablate refusal directions.

        CAUTION: This is experimental. Re-extracting from ablated models
        may find artifacts rather than true refusal directions.

        Args:
            good_prompts: Prompts that don't trigger refusals
            bad_prompts: Prompts that trigger refusals
            parameters: Abliteration parameters for each component
            max_rounds: Maximum number of ablation rounds
            min_direction_magnitude: Minimum magnitude (pre-norm) to continue
            max_kl_per_round: Maximum KL divergence allowed per round
            base_logprobs: Pre-computed logprobs for KL calculation
            kl_check_prompts: Prompts to use for KL checking

        Returns:
            Tuple of (rounds_performed, list of KL values per round)
        """
        kl_values = []

        for round_idx in range(max_rounds):
            print(f"  * Iterative ablation round {round_idx + 1}/{max_rounds}...")

            # Extract residual directions from current model state
            good_residuals = self.get_residuals_batched(good_prompts)
            bad_residuals = self.get_residuals_batched(bad_prompts)

            # Compute raw difference BEFORE normalization for magnitude check
            raw_difference = bad_residuals.mean(dim=0) - good_residuals.mean(dim=0)

            # Check magnitude BEFORE normalization (FIX for original bug)
            direction_magnitude = raw_difference.norm(dim=1).mean().item()
            print(f"    * Direction magnitude (pre-norm): {direction_magnitude:.4f}")

            if direction_magnitude < min_direction_magnitude:
                print(
                    f"    * Below threshold ({min_direction_magnitude}), stopping iteration"
                )
                break

            # Now normalize for abliteration
            refusal_directions = F.normalize(raw_difference, p=2, dim=1)

            # Ablate this round's directions (per-layer mode)
            self.abliterate(refusal_directions, None, parameters, layer_profiles)

            # Capability guard - measure KL after each round
            if (
                round_idx < max_rounds - 1
                and base_logprobs is not None
                and kl_check_prompts is not None
            ):
                current_logprobs = self.get_logprobs_batched(kl_check_prompts)
                round_kl = F.kl_div(
                    current_logprobs,
                    base_logprobs,
                    reduction="batchmean",
                    log_target=True,
                ).item()
                kl_values.append(round_kl)
                print(f"    * Round KL: {round_kl:.3f}")

                if round_kl > max_kl_per_round:
                    print(
                        f"    * Round KL exceeds limit ({max_kl_per_round}), stopping"
                    )
                    break

            # Clean up
            del good_residuals, bad_residuals, raw_difference
            empty_cache()

        return round_idx + 1, kl_values

    def abliterate_multi_direction(
        self,
        refusal_directions: Tensor,
        direction_weights: list[float],
        parameters: dict[str, "AbliterationParameters"],
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ):
        """Abliterate multiple refusal directions with configurable weights.

        Args:
            refusal_directions: Shape (n_layers, n_components, hidden_dim)
            direction_weights: Weight for each component
            parameters: Abliteration parameters for each component
        """
        n_components = refusal_directions.shape[1]

        for component_idx in range(min(n_components, len(direction_weights))):
            weight_multiplier = direction_weights[component_idx]
            if weight_multiplier <= 0:
                continue

            print(
                f"    * Abliterating direction {component_idx + 1}/{n_components} (weight={weight_multiplier:.2f})"
            )

            # Extract single direction for this component: (n_layers, hidden_dim)
            single_direction = refusal_directions[:, component_idx, :]

            # Scale weights in parameters by multiplier
            scaled_parameters = {}
            for comp_name, params in parameters.items():
                scaled_parameters[comp_name] = AbliterationParameters(
                    max_weight=params.max_weight * weight_multiplier,
                    max_weight_position=params.max_weight_position,
                    min_weight=params.min_weight * weight_multiplier,
                    min_weight_distance=params.min_weight_distance,
                )

            # Apply abliteration with scaled weights (per-layer mode)
            self.abliterate(single_direction, None, scaled_parameters, layer_profiles)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
