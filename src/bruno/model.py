# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)
from requests.exceptions import ConnectionError, Timeout
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
from .constants import (
    CACHE_CLEAR_INTERVAL,
    EPSILON,
    NEAR_ZERO,
    SACRED_PCA_ALPHA,
)
from .error_tracker import record_suppressed_error
from .exceptions import (
    ConceptConeError,
    ConfigurationError,
    ModelInferenceError,
    ModelLoadError,
    MoEAbliterationError,
    ResidualExtractionError,
    SacredDirectionError,
    SupervisedProbeError,
    WeightModificationError,
)
from .logging import get_logger
from .utils import batchify, empty_cache, print

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .evaluator import Evaluator


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
class RefusalActivationStats:
    """Statistics about refusal direction activations across prompts.

    Used for calibrating ablation weights based on the strength of refusal
    activations in the specific model/prompt combination. This enables
    adaptive weight scaling instead of fixed ablation strengths.

    Attributes:
        mean_projection: Mean absolute projection onto refusal direction
        std_projection: Standard deviation of projections
        percentile_25: 25th percentile of projection values
        percentile_75: 75th percentile of projection values
        percentile_95: 95th percentile of projection values
        n_samples: Number of samples used for statistics
        target_layer: Which layer these statistics were computed from
    """

    mean_projection: float
    std_projection: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    n_samples: int
    target_layer: int

    def get_percentile_value(self, percentile: float) -> float:
        """Get interpolated percentile value.

        Args:
            percentile: Value between 0.0 and 1.0

        Returns:
            Estimated value at the given percentile
        """
        if percentile <= 0.25:
            # Extrapolate below 25th percentile using mean and std
            return self.mean_projection - self.std_projection * (0.5 - percentile) * 2
        elif percentile <= 0.5:
            # Interpolate between 25th percentile and mean
            t = (percentile - 0.25) / 0.25
            return self.percentile_25 + t * (self.mean_projection - self.percentile_25)
        elif percentile <= 0.75:
            # Interpolate between mean and 75th percentile
            t = (percentile - 0.5) / 0.25
            return self.mean_projection + t * (
                self.percentile_75 - self.mean_projection
            )
        elif percentile <= 0.95:
            # Interpolate between 75th and 95th percentile
            t = (percentile - 0.75) / 0.2
            return self.percentile_75 + t * (self.percentile_95 - self.percentile_75)
        else:
            # Extrapolate above 95th percentile
            return self.percentile_95 + self.std_projection * (percentile - 0.95) * 5


@dataclass
class ConceptCone:
    """A cluster of harmful prompts with their specific refusal direction.

    Represents a "concept cone" - a category of harmful prompts (e.g., violence,
    illegal content, self-harm) that may activate different refusal mechanisms.
    Each cone has its own extracted refusal directions.

    Attributes:
        cluster_id: Unique identifier for this cluster
        centroid: Embedding centroid for this category (mean across layers)
        directions: Refusal directions specific to this cone
            Shape: (n_layers, n_components, hidden_dim)
        eigenvalues: Eigenvalues for each direction per layer
            Shape: (n_layers, n_components)
        prompt_indices: Indices of prompts belonging to this cone
        size: Number of prompts in this cone
    """

    cluster_id: int
    centroid: Tensor  # Mean embedding for this cluster
    directions: Tensor  # Refusal directions for this cone
    eigenvalues: Tensor  # Eigenvalues from PCA
    prompt_indices: list[int]  # Which prompts belong to this cone
    size: int  # Number of prompts in this cone


@dataclass
class ConceptConeExtractionResult:
    """Result from concept cone-based refusal direction extraction.

    Contains multiple concept cones, each representing a category of harmful
    prompts with its own refusal directions.
    """

    cones: list[ConceptCone]
    silhouette_score: float  # Clustering quality metric (-1 to 1, higher is better)
    n_clusters_requested: int  # How many clusters were requested
    n_clusters_valid: int  # How many clusters passed min_size threshold

    def get_global_directions(self) -> Tensor:
        """Get weighted average of all cone directions.

        Weights cones by their size (number of prompts).

        Returns:
            Tensor of shape (n_layers, n_components, hidden_dim)
        """
        if not self.cones:
            raise ValueError("No cones available")

        total_size = sum(cone.size for cone in self.cones)
        if total_size == 0:
            return self.cones[0].directions

        # Weighted average of directions
        weighted_sum = torch.zeros_like(self.cones[0].directions)
        for cone in self.cones:
            weight = cone.size / total_size
            weighted_sum += weight * cone.directions

        return weighted_sum

    def get_primary_directions(self) -> Tensor:
        """Get first principal direction from each cone, stacked.

        Returns:
            Tensor of shape (n_cones, n_layers, hidden_dim)
        """
        if not self.cones:
            raise ValueError("No cones available")

        return torch.stack([cone.directions[:, 0, :] for cone in self.cones])


class ConceptConeExtractor:
    """Extract multiple concept cones from harmful prompts.

    Uses KMeans clustering on residual patterns to identify different
    categories of harmful content, then extracts category-specific
    refusal directions for each cluster.
    """

    def __init__(
        self,
        n_cones: int = 5,
        n_directions_per_cone: int = 2,
        min_cluster_size: int = 10,
        min_silhouette_score: float = 0.1,
    ):
        self.n_cones = n_cones
        self.n_directions_per_cone = n_directions_per_cone
        self.min_cluster_size = min_cluster_size
        self.min_silhouette_score = min_silhouette_score
        self.cones: list[ConceptCone] = []
        self.silhouette_score = 0.0
        self.cluster_labels: np.ndarray | None = None

    def extract(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        model: "Model",
    ) -> ConceptConeExtractionResult:
        """Extract concept cones from bad prompts.

        Clusters bad prompts by their residual patterns and extracts
        refusal directions specific to each cluster.

        Args:
            good_residuals: Residuals from harmless prompts
                Shape: (n_good, n_layers, hidden_dim)
            bad_residuals: Residuals from harmful prompts
                Shape: (n_bad, n_layers, hidden_dim)
            model: Model instance for PCA extraction

        Returns:
            ConceptConeExtractionResult with extracted cones
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Step 1: Cluster bad prompts by their residual patterns
        # Use the mean across layers as the embedding
        bad_embeddings = bad_residuals.mean(dim=1).cpu().numpy()

        # Adjust n_clusters if we don't have enough samples
        n_samples = bad_embeddings.shape[0]
        actual_n_cones = min(self.n_cones, n_samples // self.min_cluster_size)
        if actual_n_cones < 2:
            print(
                f"[yellow]Warning: Not enough samples ({n_samples}) for clustering. "
                f"Need at least {self.min_cluster_size * 2} samples for 2 clusters.[/yellow]"
            )
            actual_n_cones = 1

        if actual_n_cones == 1:
            # Single cluster - just use all bad residuals
            pca_result = model.get_refusal_directions_pca(
                good_residuals, bad_residuals, n_components=self.n_directions_per_cone
            )
            single_cone = ConceptCone(
                cluster_id=0,
                centroid=torch.from_numpy(bad_embeddings.mean(axis=0)).float(),
                directions=pca_result.directions,
                eigenvalues=pca_result.eigenvalues,
                prompt_indices=list(range(n_samples)),
                size=n_samples,
            )
            return ConceptConeExtractionResult(
                cones=[single_cone],
                silhouette_score=0.0,
                n_clusters_requested=self.n_cones,
                n_clusters_valid=1,
            )

        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=actual_n_cones,
            random_state=42,
            n_init=10,
        )
        self.cluster_labels = kmeans.fit_predict(bad_embeddings)

        # Evaluate clustering quality
        if len(set(self.cluster_labels)) > 1:
            self.silhouette_score = silhouette_score(
                bad_embeddings, self.cluster_labels
            )
            print(f"  * Concept cone silhouette score: {self.silhouette_score:.3f}")

            if self.silhouette_score < self.min_silhouette_score:
                print(
                    f"[yellow]Warning: Poor clustering quality "
                    f"(silhouette={self.silhouette_score:.3f} < {self.min_silhouette_score}). "
                    f"Refusal may not have distinct concept cones.[/yellow]"
                )
        else:
            self.silhouette_score = 0.0

        # Extract directions for each cluster
        self.cones = []
        for cluster_id in range(actual_n_cones):
            mask = self.cluster_labels == cluster_id
            cluster_size = int(mask.sum())

            if cluster_size < self.min_cluster_size:
                print(
                    f"  * Skipping cluster {cluster_id} "
                    f"(size={cluster_size} < {self.min_cluster_size})"
                )
                continue

            # Extract directions specific to this cluster
            cluster_residuals = bad_residuals[mask]

            pca_result = model.get_refusal_directions_pca(
                good_residuals,
                cluster_residuals,
                n_components=self.n_directions_per_cone,
            )

            cone = ConceptCone(
                cluster_id=cluster_id,
                centroid=torch.from_numpy(kmeans.cluster_centers_[cluster_id]).float(),
                directions=pca_result.directions,
                eigenvalues=pca_result.eigenvalues,
                prompt_indices=torch.where(torch.tensor(mask))[0].tolist(),
                size=cluster_size,
            )
            self.cones.append(cone)

        print(
            f"  * Extracted {len(self.cones)} concept cones "
            f"from {actual_n_cones} clusters"
        )

        return ConceptConeExtractionResult(
            cones=self.cones,
            silhouette_score=self.silhouette_score,
            n_clusters_requested=self.n_cones,
            n_clusters_valid=len(self.cones),
        )

    def select_cone(self, prompt_residual: Tensor) -> ConceptCone:
        """Select the most relevant cone for a given prompt.

        Finds the cone whose centroid is closest to the prompt's embedding.

        Args:
            prompt_residual: Residual for a single prompt
                Shape: (n_layers, hidden_dim)

        Returns:
            The ConceptCone closest to the prompt

        Raises:
            ValueError: If no cones have been extracted
        """
        if not self.cones:
            raise ValueError("No cones extracted. Call extract() first.")

        # Average across layers to get embedding
        embedding = prompt_residual.mean(dim=0).cpu()

        # Find nearest centroid
        min_dist = float("inf")
        best_cone = self.cones[0]

        for cone in self.cones:
            dist = torch.norm(embedding - cone.centroid).item()
            if dist < min_dist:
                min_dist = dist
                best_cone = cone

        return best_cone

    def get_cone_assignment(self, prompt_residuals: Tensor) -> list[int]:
        """Get cone assignments for multiple prompts.

        Args:
            prompt_residuals: Residuals for multiple prompts
                Shape: (n_prompts, n_layers, hidden_dim)

        Returns:
            List of cone indices (into self.cones) for each prompt
        """
        if not self.cones:
            raise ValueError("No cones extracted. Call extract() first.")

        assignments = []
        for i in range(prompt_residuals.shape[0]):
            cone = self.select_cone(prompt_residuals[i])
            # Find index of this cone in self.cones
            cone_idx = next(
                j for j, c in enumerate(self.cones) if c.cluster_id == cone.cluster_id
            )
            assignments.append(cone_idx)

        return assignments


@dataclass
class RefusalCircuit:
    """A specific attention head identified as mediating refusal.

    Represents a single attention head that contributes to refusal behavior.
    Circuit-level ablation targets these specific heads instead of entire layers,
    potentially reducing capability damage.

    Attributes:
        layer_idx: Index of the transformer layer containing this head
        head_idx: Index of the attention head within the layer
        importance: How much this head contributes to refusal (0.0-1.0)
            Higher values indicate stronger refusal mediation
    """

    layer_idx: int
    head_idx: int
    importance: float


@dataclass
class MoEExpertActivations:
    """Tracked expert activations for MoE models.

    Contains information about which experts were activated during inference
    for a set of prompts. Used for router-aware expert targeting.

    Attributes:
        layer_expert_counts: Dict mapping layer_idx -> expert_idx -> activation count
        total_prompts: Number of prompts used for tracking
        n_experts_per_layer: Number of routed experts per MoE layer
        top_k: Number of experts activated per token
    """

    layer_expert_counts: dict[int, dict[int, int]]
    total_prompts: int
    n_experts_per_layer: int
    top_k: int

    def get_expert_frequencies(self, layer_idx: int) -> dict[int, float]:
        """Get activation frequency (0-1) for each expert at a layer."""
        if layer_idx not in self.layer_expert_counts:
            return {}
        return {
            expert_idx: count / max(self.total_prompts, 1)
            for expert_idx, count in self.layer_expert_counts[layer_idx].items()
        }

    def get_high_activation_experts(
        self, layer_idx: int, threshold: float = 0.1, top_k: int = 0
    ) -> list[int]:
        """Get experts with activation frequency above threshold.

        Args:
            layer_idx: Layer index
            threshold: Minimum activation frequency (0-1)
            top_k: Maximum number of experts to return (0 = no limit)

        Returns:
            List of expert indices sorted by activation frequency (descending)
        """
        freqs = self.get_expert_frequencies(layer_idx)
        # Filter by threshold and sort by frequency
        filtered = [(idx, freq) for idx, freq in freqs.items() if freq >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k > 0:
            filtered = filtered[:top_k]

        return [idx for idx, _ in filtered]


@dataclass
class CircuitAblationResult:
    """Result from circuit-level ablation.

    Contains information about which circuits were ablated and any issues encountered.
    """

    circuits_ablated: int  # Number of circuits successfully ablated
    circuits_skipped: int  # Number of circuits skipped (out of bounds, etc.)
    gqa_detected: bool  # Whether GQA was detected (ablation skipped if True)
    n_heads: int  # Number of attention heads in the model
    n_kv_heads: int  # Number of key-value heads (differs from n_heads in GQA)


@dataclass
class SacredDirectionResult:
    """Result from sacred direction extraction.

    Contains directions that encode capabilities to preserve during ablation.
    These are orthogonalized against before abliterating to prevent capability damage.

    Attributes:
        directions: Sacred directions tensor of shape (n_layers, n_sacred, hidden_dim)
        eigenvalues: Eigenvalues from PCA of shape (n_layers, n_sacred)
    """

    directions: Tensor
    eigenvalues: Tensor


@dataclass
class SupervisedExtractionResult:
    """Result from supervised probe-based refusal direction extraction.

    Contains directions learned by training linear classifiers to predict
    refusal behavior, along with cross-validation accuracy scores.
    """

    directions: Tensor  # Shape: (n_layers, hidden_dim)
    accuracies: list[float]  # Cross-validation accuracy per layer

    def get_mean_accuracy(self) -> float:
        """Get the mean cross-validation accuracy across all layers."""
        return sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0.0

    def get_min_accuracy(self) -> float:
        """Get the minimum cross-validation accuracy across layers."""
        return min(self.accuracies) if self.accuracies else 0.0

    def get_max_accuracy(self) -> float:
        """Get the maximum cross-validation accuracy across layers."""
        return max(self.accuracies) if self.accuracies else 0.0


class RefusalProbe:
    """Linear probe trained to predict refusal behavior.

    Uses logistic regression with L2 regularization and balanced class weights
    to handle potential class imbalance. The learned weights serve as a
    refusal direction that directly captures the decision boundary between
    refuse and comply behaviors.
    """

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        self.device = device
        self.hidden_dim = hidden_dim
        self.weights = torch.zeros(hidden_dim, device=device)
        self.bias = 0.0
        self.accuracy = 0.0  # Track probe quality via cross-validation

    def fit(
        self,
        refusal_residuals: Tensor,  # Residuals from prompts that triggered refusals
        comply_residuals: Tensor,  # Residuals from prompts that got compliance
        layer_idx: int,
    ) -> float:
        """Train probe on labeled residuals using logistic regression.

        Args:
            refusal_residuals: Residuals from prompts where model refused
                Shape: (n_refusal, n_layers, hidden_dim)
            comply_residuals: Residuals from prompts where model complied
                Shape: (n_comply, n_layers, hidden_dim)
            layer_idx: Which layer to extract activations from

        Returns:
            Cross-validation accuracy score (0.0-1.0)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Combine data from the specified layer
        X = (
            torch.cat(
                [
                    refusal_residuals[:, layer_idx, :],
                    comply_residuals[:, layer_idx, :],
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )

        # Labels: 1 for refusal, 0 for compliance
        y = np.concatenate(
            [
                np.ones(len(refusal_residuals)),
                np.zeros(len(comply_residuals)),
            ]
        )

        # Handle class imbalance with balanced class weights
        # Note: L2 regularization is the default; we use C=1.0 to control strength
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",  # Handle imbalanced data
            solver="lbfgs",
            random_state=42,
        )

        # Cross-validate to check probe quality (minimum 10 samples needed)
        if len(X) >= 10:
            n_splits = min(5, len(X) // 2)
            if n_splits >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=n_splits)
                self.accuracy = float(cv_scores.mean())
            else:
                self.accuracy = 0.5  # Insufficient data for CV
        else:
            self.accuracy = 0.5  # Insufficient data

        # Fit on all data
        clf.fit(X, y)

        # Extract weights as the refusal direction
        self.weights = torch.from_numpy(clf.coef_[0]).float().to(self.device)
        self.bias = float(clf.intercept_[0])

        # Normalize to unit vector
        self.weights = F.normalize(self.weights, p=2, dim=0)

        return self.accuracy


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
        positive_eigenvalues = torch.clamp(mean_eigenvalues, min=EPSILON)

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
            raise ConfigurationError(
                f"Unknown eigenvalue weight method: '{method}'. "
                f"Valid options: 'softmax', 'proportional', 'log_proportional'."
            )

        # Scale so that the first (largest) weight is 1.0
        # This maintains compatibility with existing weight semantics
        if weights[0] > 0:
            weights = weights / weights[0]

        return weights.tolist()


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info("Initializing model", model=settings.model)
        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                settings.model,
                trust_remote_code=True,
            )
            logger.debug("Tokenizer loaded successfully", model=settings.model)
        except RepositoryNotFoundError as error:
            logger.error("Tokenizer not found", model=settings.model, error=str(error))
            print(f"[red]Tokenizer Not Found: {settings.model}[/]")
            print("[yellow]Solutions:[/]")
            print(f"  1. Check model name spelling: {settings.model}")
            print("  2. Search HuggingFace Hub: https://huggingface.co/models")
            raise ModelLoadError(
                f"Tokenizer for model '{settings.model}' not found. "
                f"Check model name or search: https://huggingface.co/models"
            ) from error
        except GatedRepoError as error:
            logger.error("Model requires authentication", model=settings.model)
            print(f"[red]Authentication Required: {settings.model}[/]")
            print("[yellow]Run: huggingface-cli login[/]")
            raise ModelLoadError(
                f"Model '{settings.model}' requires authentication. "
                f"Run 'huggingface-cli login' first."
            ) from error
        except (ConnectionError, Timeout) as error:
            logger.error(
                "Network error loading tokenizer",
                model=settings.model,
                error=str(error),
            )
            print("[red]Network Error: Cannot download tokenizer[/]")
            print("[yellow]Check internet connection and try again[/]")
            raise ModelLoadError(
                f"Network error downloading tokenizer for '{settings.model}'. "
                f"Check internet connection."
            ) from error

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None

        logger.debug(
            "Starting dtype fallback chain", dtypes=[str(d) for d in settings.dtypes]
        )
        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")
            logger.debug("Attempting dtype", dtype=str(dtype))

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    torch_dtype=dtype,
                    device_map=settings.device_map,
                    trust_remote_code=True,
                )

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
            except RepositoryNotFoundError as error:
                logger.error("Model repository not found", model=settings.model)
                # Model not found on HuggingFace Hub
                print()
                print(f"[red]Model Not Found: {settings.model}[/]")
                print("[yellow]Solutions:[/]")
                print(f"  1. Check model name spelling: {settings.model}")
                print("  2. Search HuggingFace Hub: https://huggingface.co/models")
                print("  3. Verify model exists and is not private")
                raise ModelLoadError(
                    f"Model '{settings.model}' not found on HuggingFace Hub. "
                    f"Check spelling or search: https://huggingface.co/models"
                ) from error
            except GatedRepoError as error:
                # Model requires authentication
                print()
                print(f"[red]Authentication Required: {settings.model}[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Login with: huggingface-cli login")
                print("  2. Or set HF_TOKEN environment variable")
                print("  3. Accept model terms on HuggingFace Hub if required")
                raise ModelLoadError(
                    f"Model '{settings.model}' is gated and requires authentication. "
                    f"Run 'huggingface-cli login' or set HF_TOKEN environment variable."
                ) from error
            except (ConnectionError, Timeout) as error:
                # Network error during model download
                print()
                print("[red]Network Error: Cannot download model[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Check your internet connection")
                print("  2. Try again in a few minutes")
                print(
                    "  3. Check HuggingFace Hub status: https://status.huggingface.co"
                )
                raise ModelLoadError(
                    f"Network error while downloading model '{settings.model}'. "
                    f"Check internet connection and try again."
                ) from error
            except OSError as error:
                # Check if disk space error
                error_msg = str(error).lower()
                if (
                    "disk" in error_msg
                    or "space" in error_msg
                    or "storage" in error_msg
                ):
                    print()
                    print("[red]Insufficient Disk Space[/]")
                    print("[yellow]Solutions:[/]")
                    print("  1. Free up disk space (check cache directory)")
                    print("  2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface")
                    print("  3. Use smaller model or quantized version")
                    raise ModelLoadError(
                        f"Insufficient disk space to download model '{settings.model}'. "
                        f"Free up space or clear cache: ~/.cache/huggingface"
                    ) from error
                # Other OS errors - let them fall through to general exception
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue
            except RuntimeError as error:
                # Check if dtype incompatibility (probability tensor error)
                error_msg = str(error).lower()
                if (
                    "probability" in error_msg
                    or "inf" in error_msg
                    or "nan" in error_msg
                ):
                    # Dtype incompatibility - try next dtype
                    self.model = None
                    empty_cache()
                    print("[yellow]Incompatible[/] (dtype issue, trying next...)")
                    continue
                # Other runtime errors - let them fall through
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue
            except torch.cuda.OutOfMemoryError:
                # GPU out of memory - try next dtype (lower precision)
                self.model = None
                empty_cache()
                print("[yellow]GPU OOM[/] (trying next dtype...)")
                continue
            except KeyboardInterrupt:
                # CRITICAL: Always let user interrupt
                raise
            except Exception as error:
                # Unexpected error - log and continue to next dtype
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            self.loaded_dtype = dtype  # Store successful dtype for potential reload
            break

        if self.model is None:
            print()
            print("[red]Failed to load model with any configured dtype[/]")
            print(f"[yellow]Tried dtypes: {settings.dtypes}[/]")
            print("[yellow]Solutions:[/]")
            print(
                "  1. Add more dtypes to config: dtypes = [torch.float16, torch.bfloat16, torch.float32]"
            )
            print("  2. Check GPU compatibility (older GPUs may not support bfloat16)")
            print("  3. Try smaller model or quantized version")
            raise ModelLoadError(
                f"Failed to load model '{settings.model}' with any configured dtype. "
                f"Tried: {settings.dtypes}. Check GPU compatibility or try different dtypes."
            )

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")

        # Check multiple layers to detect MoE (layer 0 may be dense in hybrid models)
        is_moe_hybrid = False
        moe_layer_idx = None
        moe_config = None
        for check_idx in range(min(3, len(self.get_layers()))):
            matrices = self.get_layer_matrices(check_idx)
            if matrices.get("mlp.down_proj") and len(matrices["mlp.down_proj"]) > 1:
                is_moe_hybrid = True
                moe_layer_idx = check_idx
                moe_config = self.get_moe_config(check_idx)
                break

        if is_moe_hybrid:
            if moe_config:
                n_experts, top_k = moe_config
                print(
                    f"* [bold cyan]MoE Architecture Detected[/]: "
                    f"{n_experts} routed experts, top-{top_k} routing"
                )
            else:
                print("* [bold cyan]MoE Architecture Detected[/]")

        # Show component info for representative layer
        sample_idx = moe_layer_idx if moe_layer_idx is not None else 0
        for component, matrices in self.get_layer_matrices(sample_idx).items():
            count = len(matrices)
            if is_moe_hybrid and component == "mlp.down_proj":
                # Show MoE-specific info
                dense_count = len(self.get_layer_matrices(0).get("mlp.down_proj", []))
                has_shared = hasattr(
                    self.get_layers()[moe_layer_idx].mlp, "shared_experts"
                )
                shared_info = " + shared_experts" if has_shared else ""
                print(
                    f"  * [bold]{component}[/]: MoE - "
                    f"[bold]{dense_count}[/] (dense) / [bold]{count}[/] (MoE){shared_info}"
                )
            else:
                print(f"  * [bold]{component}[/]: [bold]{count}[/] matrices per layer")

        # Cache abliterable weights in memory for fast reset (avoids reloading from disk)
        # Uses selective layer-wise caching (~55% memory reduction vs full state_dict)
        # Enables caching for 32B+ models that would OOM with full caching
        if settings.cache_weights:
            print("* Caching abliterable weights in memory (selective mode)...")
            self.layer_weights_cache = self._create_layer_weights_cache()
            self.original_state_dict = None  # Legacy attribute, no longer used

            # Log cache statistics for verification
            cache_size_mb = sum(
                sum(m.numel() * m.element_size() for m in matrices)
                for layer_cache in self.layer_weights_cache.values()
                for matrices in layer_cache.values()
            ) / (1024**2)

            total_params = sum(p.numel() for p in self.model.parameters())
            cached_params = sum(
                sum(m.numel() for m in matrices)
                for layer_cache in self.layer_weights_cache.values()
                for matrices in layer_cache.values()
            )
            cache_ratio = (
                (cached_params / total_params) * 100 if total_params > 0 else 0
            )

            print(
                f"  * Cache size: {cache_size_mb:.1f} MB "
                f"({cached_params:,} params, {cache_ratio:.1f}% of model)"
            )
        else:
            print("* Weight caching disabled (will reload from disk between trials)")
            self.layer_weights_cache = None
            self.original_state_dict = None

        # Optionally compile the model for faster inference (~1.5-2x speedup)
        if settings.compile:
            print("* Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  * Model compiled successfully")
            except RuntimeError as e:
                # torch.compile() can fail on Windows, old CUDA, or unsupported ops
                if "compile" in str(e).lower() or "dynamo" in str(e).lower():
                    logger.warning(
                        f"torch.compile() failed: {e}. Continuing with uncompiled model."
                    )
                    print(
                        "[yellow]  * Compilation failed, using uncompiled model "
                        "(inference will be slower)[/]"
                    )
                else:
                    # Unexpected RuntimeError - re-raise
                    raise ModelLoadError(
                        f"Failed to compile model: {e}. Try --compile false"
                    ) from e
            except Exception as e:
                # Truly unexpected errors
                logger.error(f"Unexpected error during model compilation: {e}")
                print(
                    f"[yellow]  * Unexpected compilation error: {e}. "
                    f"Using uncompiled model.[/]"
                )
                # Continue without compilation rather than crashing

    def _create_layer_weights_cache(self) -> dict[int, dict[str, list[Tensor]]]:
        """Create selective cache of abliterable component weights.

        Caches only attn.o_proj and mlp.down_proj components per layer,
        reducing memory footprint by ~55% vs full state_dict caching.
        Also caches gate weights for MoE models.

        This enables weight caching for 32B+ models that would otherwise
        experience OOM when caching the full state_dict (e.g., 62GB model
        + 62GB cache = 124GB > 141GB H200 memory).

        Returns:
            Nested dict mapping layer_idx -> component_name -> list of weight tensors.
            Structure matches get_layer_matrices() return format for seamless integration.

        Raises:
            ModelLoadError: If cache creation fails (OOM, layer access errors, etc.)

        Example structure:
            {
                0: {
                    "attn.o_proj": [tensor1],  # Always 1 tensor
                    "mlp.down_proj": [tensor1, tensor2, ...]  # 1 for dense, 8 for MoE
                    "gate.weight": [tensor1],  # MoE gate weight (if present)
                    "gate.bias": [tensor1],  # MoE gate bias (if present)
                },
                1: { ... },
                ...
            }
        """
        cache = {}
        num_layers = len(self.get_layers())

        try:
            for layer_idx in range(num_layers):
                try:
                    layer_matrices = self.get_layer_matrices(layer_idx)
                except Exception as e:
                    raise ModelLoadError(
                        f"Failed to access layer {layer_idx} matrices during cache creation. "
                        f"Model architecture may be incompatible. Error: {e}"
                    ) from e

                # Clone tensors: detach from computation graph, preserve device placement
                try:
                    cache[layer_idx] = {
                        component: [matrix.clone().detach() for matrix in matrices]
                        for component, matrices in layer_matrices.items()
                    }
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise ModelLoadError(
                            f"Insufficient memory to create weight cache at layer {layer_idx}. "
                            f"Try --cache-weights false to disable caching. "
                            f"Original error: {e}"
                        ) from e
                    else:
                        raise ModelLoadError(
                            f"Failed to clone weights at layer {layer_idx}. Error: {e}"
                        ) from e

                # Add gate weight caching for MoE models
                layer = self.get_layers()[layer_idx]
                if hasattr(layer.mlp, "gate"):
                    gate = layer.mlp.gate
                    if hasattr(gate, "weight"):
                        try:
                            cache[layer_idx]["gate.weight"] = [
                                gate.weight.clone().detach()
                            ]
                        except RuntimeError:
                            pass  # Not critical if gate caching fails

                    # Also cache bias if present (e_score_correction in DeepSeek)
                    if hasattr(gate, "e_score_correction"):
                        try:
                            cache[layer_idx]["gate.bias"] = [
                                gate.e_score_correction.clone().detach()
                            ]
                        except RuntimeError:
                            pass  # Not critical if bias caching fails

            # Validation: Ensure cache is not empty and has expected structure
            if not cache:
                raise ModelLoadError(
                    "Weight cache creation resulted in empty cache. "
                    "Model may have 0 layers or get_layers() failed."
                )

            # Validate cache structure for first layer (representative check)
            if 0 in cache:
                if not cache[0]:
                    raise ModelLoadError(
                        "Layer 0 cache is empty. No abliterable components found. "
                        "Model architecture may be incompatible."
                    )

                # Verify expected components exist
                if "attn.o_proj" not in cache[0]:
                    raise ModelLoadError(
                        "Required component 'attn.o_proj' not found in layer 0 cache. "
                        "Model architecture may be incompatible with abliteration."
                    )

        except ModelLoadError:
            # Re-raise ModelLoadError as-is
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise ModelLoadError(
                f"Unexpected error during weight cache creation: {e}. "
                f"Try --cache-weights false to disable caching."
            ) from e

        return cache

    def reload_model(self):
        """Reset model weights to original state.

        Uses selective layer-wise cache if available (fast, ~10-15s for 32B models),
        otherwise reloads from disk (slow, ~60-120s for 32B models).

        The selective cache only stores abliterable components (attn.o_proj, mlp.down_proj)
        and gate weights for MoE models, reducing memory usage by ~55% vs full state_dict
        caching while maintaining fast reload performance.

        Raises:
            ModelLoadError: If cache restoration fails (missing layers, shape mismatches, etc.)
        """
        if self.layer_weights_cache is not None:
            # Fast selective weight restoration (layer-wise)
            num_layers = len(self.get_layers())

            # Validate cache matches current model structure
            if len(self.layer_weights_cache) != num_layers:
                raise ModelLoadError(
                    f"Cache layer count mismatch: cache has {len(self.layer_weights_cache)} layers, "
                    f"model has {num_layers} layers. Cache may be corrupted or from different model."
                )

            with (
                torch.no_grad()
            ):  # Disable gradient tracking for in-place copy operations
                for layer_idx in range(num_layers):
                    # Verify layer exists in cache
                    if layer_idx not in self.layer_weights_cache:
                        raise ModelLoadError(
                            f"Layer {layer_idx} missing from weight cache. "
                            f"Cache may be corrupted. Available layers: {sorted(self.layer_weights_cache.keys())}"
                        )

                    try:
                        current_matrices = self.get_layer_matrices(layer_idx)
                        cached_layer = self.layer_weights_cache[layer_idx]
                    except Exception as e:
                        raise ModelLoadError(
                            f"Failed to access layer {layer_idx} during cache restoration: {e}"
                        ) from e

                    for component, matrices in current_matrices.items():
                        # Verify component exists in cache
                        if component not in cached_layer:
                            raise ModelLoadError(
                                f"Component '{component}' missing from layer {layer_idx} cache. "
                                f"Available components: {list(cached_layer.keys())}. "
                                f"Model architecture may have changed."
                            )

                        cached_matrices = cached_layer[component]

                        # Verify matrix count matches
                        if len(matrices) != len(cached_matrices):
                            raise ModelLoadError(
                                f"Matrix count mismatch at layer {layer_idx}, component '{component}': "
                                f"model has {len(matrices)} matrices, cache has {len(cached_matrices)}. "
                                f"Model architecture may have changed (e.g., expert count in MoE)."
                            )

                        # Restore each matrix from cache
                        for matrix_idx, (matrix, cached) in enumerate(
                            zip(matrices, cached_matrices)
                        ):
                            # Verify shapes match
                            if matrix.shape != cached.shape:
                                raise ModelLoadError(
                                    f"Shape mismatch at layer {layer_idx}, component '{component}', "
                                    f"matrix {matrix_idx}: model shape {matrix.shape}, "
                                    f"cache shape {cached.shape}. Model architecture may have changed."
                                )

                            # Verify dtypes match
                            if matrix.dtype != cached.dtype:
                                logger.warning(
                                    f"dtype mismatch at layer {layer_idx}, component '{component}', "
                                    f"matrix {matrix_idx}: model dtype {matrix.dtype}, "
                                    f"cache dtype {cached.dtype}. Converting cached tensor."
                                )
                                cached = cached.to(dtype=matrix.dtype)

                            # Verify devices match
                            if matrix.device != cached.device:
                                logger.warning(
                                    f"Device mismatch at layer {layer_idx}, component '{component}', "
                                    f"matrix {matrix_idx}: model device {matrix.device}, "
                                    f"cache device {cached.device}. Moving cached tensor."
                                )
                                cached = cached.to(device=matrix.device)

                            # Perform restoration
                            try:
                                matrix.copy_(cached)
                            except RuntimeError as e:
                                raise ModelLoadError(
                                    f"Failed to restore weights at layer {layer_idx}, "
                                    f"component '{component}', matrix {matrix_idx}: {e}"
                                ) from e

                    # Restore gate weights for MoE models
                    layer = self.get_layers()[layer_idx]
                    if hasattr(layer.mlp, "gate"):
                        gate = layer.mlp.gate

                        # Restore gate weight
                        if (
                            "gate.weight" in cached_layer
                            and hasattr(gate, "weight")
                            and cached_layer["gate.weight"]
                        ):
                            cached_gate_weight = cached_layer["gate.weight"][0]
                            if gate.weight.shape == cached_gate_weight.shape:
                                if gate.weight.device != cached_gate_weight.device:
                                    cached_gate_weight = cached_gate_weight.to(
                                        device=gate.weight.device
                                    )
                                if gate.weight.dtype != cached_gate_weight.dtype:
                                    cached_gate_weight = cached_gate_weight.to(
                                        dtype=gate.weight.dtype
                                    )
                                gate.weight.copy_(cached_gate_weight)

                        # Restore gate bias (e_score_correction)
                        if (
                            "gate.bias" in cached_layer
                            and hasattr(gate, "e_score_correction")
                            and cached_layer["gate.bias"]
                        ):
                            cached_gate_bias = cached_layer["gate.bias"][0]
                            if gate.e_score_correction.shape == cached_gate_bias.shape:
                                if (
                                    gate.e_score_correction.device
                                    != cached_gate_bias.device
                                ):
                                    cached_gate_bias = cached_gate_bias.to(
                                        device=gate.e_score_correction.device
                                    )
                                if (
                                    gate.e_score_correction.dtype
                                    != cached_gate_bias.dtype
                                ):
                                    cached_gate_bias = cached_gate_bias.to(
                                        dtype=gate.e_score_correction.dtype
                                    )
                                gate.e_score_correction.copy_(cached_gate_bias)
        else:
            # Fallback: Reload from disk when caching disabled
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                torch_dtype=self.loaded_dtype,
                device_map=self.settings.device_map,
                trust_remote_code=True,
            )

    def get_layers(self) -> ModuleList:
        """Get the transformer layers from the model.

        Tries multimodal model structure first, then falls back to text-only.

        Returns:
            ModuleList of transformer layers

        Raises:
            ModelInferenceError: If layers cannot be accessed from model
        """
        # Most multimodal models.
        try:
            return self.model.model.language_model.layers
        except (AttributeError, TypeError):
            pass

        # Text-only models.
        try:
            return self.model.model.layers
        except (AttributeError, TypeError) as e:
            raise ModelInferenceError(
                f"Cannot access transformer layers from model. "
                f"Model architecture may be unsupported. Error: {e}"
            ) from e

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        """Get abliterable weight matrices from a specific layer.

        Extracts attention output projection and MLP down projection weights
        that can be modified during abliteration.

        Args:
            layer_index: Index of the layer to extract matrices from

        Returns:
            Dict mapping component names to lists of weight tensors

        Raises:
            ModelInferenceError: If required matrices cannot be accessed
        """
        layer = self.get_layers()[layer_index]

        matrices: dict[str, list[Tensor]] = {}

        def try_add(component: str, matrix: Any) -> bool:
            """Try to add a matrix, return True if successful."""
            if not torch.is_tensor(matrix):
                return False
            if component not in matrices:
                matrices[component] = []
            matrices[component].append(matrix)
            return True

        # Attention output projection - required for all models
        try:
            try_add("attn.o_proj", layer.self_attn.o_proj.weight)
        except (AttributeError, TypeError) as e:
            raise ModelInferenceError(
                f"Cannot access attention output projection (attn.o_proj) at layer {layer_index}. "
                f"Model architecture may be unsupported. Error: {e}"
            ) from e

        # Track which MLP architecture patterns we tried and found
        mlp_found = False

        # Most dense models.
        try:
            if try_add("mlp.down_proj", layer.mlp.down_proj.weight):
                mlp_found = True
        except (AttributeError, TypeError):
            pass  # Expected for MoE models

        # Some MoE models (e.g. Qwen3).
        try:
            for expert in layer.mlp.experts:
                if try_add("mlp.down_proj", expert.down_proj.weight):
                    mlp_found = True
        except (AttributeError, TypeError):
            pass  # Not this architecture

        # Phi-3.5-MoE (and possibly others).
        try:
            for expert in layer.block_sparse_moe.experts:
                if try_add("mlp.down_proj", expert.w2.weight):
                    mlp_found = True
        except (AttributeError, TypeError):
            pass  # Not this architecture

        # gpt-oss MoE.
        try:
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            if try_add("mlp.down_proj", layer.mlp.experts.down_proj):
                mlp_found = True
        except (AttributeError, TypeError):
            pass  # Not this architecture

        # Granite MoE Hybrid - attention layers with shared_mlp.
        try:
            if try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight):
                mlp_found = True
        except (AttributeError, TypeError):
            pass  # Not this architecture

        # Granite MoE Hybrid - MoE layers with experts.
        try:
            for expert in layer.moe.experts:
                if try_add("mlp.down_proj", expert.output_linear.weight):
                    mlp_found = True
        except (AttributeError, TypeError):
            pass  # Not this architecture

        # DeepSeekV3/Moonlight shared experts (always active for every token).
        # These are critical for MoE abliteration since they process ALL tokens.
        if self.settings.use_moe_shared_experts:
            try:
                if try_add("mlp.down_proj", layer.mlp.shared_experts.down_proj.weight):
                    mlp_found = True
            except (AttributeError, TypeError) as e:
                # Log but don't fail - shared experts are optional enhancement
                record_suppressed_error(
                    error=e,
                    context="get_layer_matrices_shared_experts",
                    module="model",
                    severity="info",
                    details={
                        "layer_index": layer_index,
                        "reason": "Shared experts not found, may not be DeepSeekV3/Moonlight",
                    },
                )

        # We need at least one MLP down-projection.
        if (
            not mlp_found
            or "mlp.down_proj" not in matrices
            or not matrices["mlp.down_proj"]
        ):
            raise ModelInferenceError(
                f"No MLP down-projection weights found at layer {layer_index}. "
                f"Model architecture may be unsupported. "
                f"Tried: dense MLP, Qwen3 MoE, Phi-3.5 MoE, gpt-oss MoE, Granite MoE."
            )

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
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> dict[str, int]:
        """Abliterate the model by projecting out refusal directions.

        Args:
            refusal_directions: Refusal directions tensor
            direction_index: Index for global direction mode, or None for per-layer
            parameters: Abliteration parameters per component
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA ('row', 'column', 'frobenius')
            mpoa_min_scale: Minimum scaling factor for MPOA (default 0.5)
            mpoa_max_scale: Maximum scaling factor for MPOA (default 2.0)

        Returns:
            Dict with abliteration statistics:
                - 'layers_processed': Number of layers that had weights modified
                - 'matrices_modified': Total number of weight matrices modified
                - 'errors_suppressed': Number of non-fatal errors encountered

        Raises:
            WeightModificationError: If critical error occurs during weight modification
        """
        stats = {
            "layers_processed": 0,
            "matrices_modified": 0,
            "errors_suppressed": 0,
        }

        # Validate inputs
        if refusal_directions is None or refusal_directions.numel() == 0:
            raise WeightModificationError(
                "Refusal directions tensor is empty or None. "
                "Direction extraction may have failed."
            )

        if not parameters:
            raise WeightModificationError(
                "No abliteration parameters provided. "
                "At least one component must have parameters."
            )

        if direction_index is None:
            refusal_direction = None
            global_projector = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)

            # Validate direction index bounds
            if int(index) + 1 >= len(refusal_directions):
                raise WeightModificationError(
                    f"Direction index {direction_index} out of bounds. "
                    f"Refusal directions has {len(refusal_directions)} entries."
                )

            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

            # Check for NaN/Inf in direction
            if (
                torch.isnan(refusal_direction).any()
                or torch.isinf(refusal_direction).any()
            ):
                raise WeightModificationError(
                    f"Refusal direction at index {direction_index} contains NaN or Inf values. "
                    f"Direction extraction may have produced invalid results."
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
            """Get device-specific projector with OOM protection.

            Returns:
                Projector on specified device, or CPU fallback if OOM
            """
            if device not in device_projectors_cache:
                try:
                    device_projectors_cache[device] = projector.to(device)
                except torch.cuda.OutOfMemoryError:
                    # OOM during transfer - use CPU fallback
                    empty_cache()
                    record_suppressed_error(
                        error=None,
                        context="abliterate_projector_transfer",
                        module="model",
                        severity="warning",
                        details={
                            "device": str(device),
                            "reason": "GPU OOM, using CPU fallback",
                        },
                    )
                    logger.warning(
                        f"GPU OOM transferring projector to {device}. Using CPU fallback."
                    )
                    device_projectors_cache[device] = projector.to("cpu")
                    stats["errors_suppressed"] += 1
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuda" in str(e):
                        raise WeightModificationError(
                            f"CUDA error transferring projector to {device}: {e}"
                        ) from e
                    raise
            return device_projectors_cache[device]

        # Log which ablation mode is being used (prevents silent failure confusion)
        if use_mpoa:
            logger.info(
                "Applying MPOA (Norm-Preserving Biprojected Abliteration)",
                norm_mode=mpoa_norm_mode,
                min_scale=mpoa_min_scale,
                max_scale=mpoa_max_scale,
            )
        else:
            logger.debug("Using standard ablation (MPOA disabled)")

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(num_layers):
            layer_modified = False

            # Compute per-layer projector once per layer (outside component loop)
            if global_projector is not None:
                projector = global_projector
            else:
                # Per-layer direction: compute projector for this layer
                # The index must be shifted by 1 because the first element
                # of refusal_directions is the direction for the embeddings.
                layer_refusal_direction = refusal_directions[layer_index + 1]

                # Check for NaN/Inf in per-layer direction
                if (
                    torch.isnan(layer_refusal_direction).any()
                    or torch.isinf(layer_refusal_direction).any()
                ):
                    record_suppressed_error(
                        error=None,
                        context="abliterate_layer_direction",
                        module="model",
                        severity="warning",
                        details={
                            "layer_index": layer_index,
                            "reason": "Layer direction contains NaN/Inf, skipping layer",
                        },
                    )
                    logger.warning(
                        f"Skipping layer {layer_index}: direction contains NaN/Inf"
                    )
                    stats["errors_suppressed"] += 1
                    continue

                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)
                # Clear the per-layer projector cache since projector changed
                device_projectors_cache.clear()

            for component, matrices in layer_matrices_cache[layer_index].items():
                if component not in parameters:
                    record_suppressed_error(
                        error=None,
                        context="abliterate_missing_params",
                        module="model",
                        severity="info",
                        details={
                            "layer_index": layer_index,
                            "component": component,
                            "reason": "No parameters for component, skipping",
                        },
                    )
                    continue

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

                for matrix_idx, matrix in enumerate(matrices):
                    try:
                        # Reuse cached projector per device to prevent memory accumulation
                        device_projector = get_device_projector(
                            projector, matrix.device
                        )

                        if use_mpoa:
                            # MPOA: Norm-Preserving Biprojected Abliteration
                            # Preserve norms to prevent capability degradation
                            self._apply_mpoa_projection(
                                matrix,
                                device_projector,
                                weight,
                                mpoa_norm_mode,
                                mpoa_min_scale,
                                mpoa_max_scale,
                            )
                        else:
                            # Standard abliteration: in-place subtraction
                            matrix.sub_(weight * (device_projector @ matrix))

                        # Verify no NaN/Inf introduced
                        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
                            record_suppressed_error(
                                error=None,
                                context="abliterate_nan_inf_detected",
                                module="model",
                                severity="error",
                                details={
                                    "layer_index": layer_index,
                                    "component": component,
                                    "matrix_idx": matrix_idx,
                                    "weight": weight,
                                    "reason": "NaN/Inf values detected after abliteration",
                                },
                            )
                            logger.error(
                                f"NaN/Inf detected in {component} at layer {layer_index}, matrix {matrix_idx} after abliteration"
                            )
                            stats["errors_suppressed"] += 1
                        else:
                            stats["matrices_modified"] += 1
                            layer_modified = True

                    except torch.cuda.OutOfMemoryError as e:
                        record_suppressed_error(
                            error=e,
                            context="abliterate_matrix_oom",
                            module="model",
                            severity="error",
                            details={
                                "layer_index": layer_index,
                                "component": component,
                                "matrix_idx": matrix_idx,
                            },
                            include_traceback=True,
                        )
                        empty_cache()
                        stats["errors_suppressed"] += 1
                        # Continue to next matrix - partial abliteration is better than none
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        if "shape" in error_msg or "size" in error_msg:
                            raise WeightModificationError(
                                f"Shape mismatch during abliteration at layer {layer_index}, "
                                f"component '{component}', matrix {matrix_idx}: {e}"
                            ) from e
                        record_suppressed_error(
                            error=e,
                            context="abliterate_matrix_error",
                            module="model",
                            severity="error",
                            details={
                                "layer_index": layer_index,
                                "component": component,
                                "matrix_idx": matrix_idx,
                            },
                            include_traceback=True,
                        )
                        stats["errors_suppressed"] += 1

            if layer_modified:
                stats["layers_processed"] += 1

            # Clear CUDA cache periodically to prevent memory fragmentation
            if layer_index % CACHE_CLEAR_INTERVAL == (CACHE_CLEAR_INTERVAL - 1):
                empty_cache()

        # Clear the device projector cache to prevent memory leaks
        device_projectors_cache.clear()

        # Log summary
        if stats["errors_suppressed"] > 0:
            logger.warning(
                "Abliteration completed with errors",
                layers_processed=stats["layers_processed"],
                matrices_modified=stats["matrices_modified"],
                errors_suppressed=stats["errors_suppressed"],
            )
        else:
            logger.info(
                "Abliteration completed successfully",
                layers_processed=stats["layers_processed"],
                matrices_modified=stats["matrices_modified"],
            )

        return stats

    def _apply_mpoa_projection(
        self,
        matrix: Tensor,
        projector: Tensor,
        weight: float,
        norm_mode: str = "row",
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        """Apply MPOA (Norm-Preserving Biprojected Abliteration) to a weight matrix.

        MPOA preserves the original norms of the weight matrix after projection,
        preventing capability degradation that occurs with standard abliteration.

        The key insight is that standard abliteration `W' = W - (r @ r^T @ W)` reduces
        the matrix norm, which damages the model's learned feature representations.
        MPOA rescales the matrix to preserve the original norms.

        Args:
            matrix: Weight matrix to modify in-place
            projector: Projection matrix (r @ r^T)
            weight: Abliteration weight/strength
            norm_mode: How to preserve norms:
                - 'row': Preserve each row's norm (recommended for attention o_proj)
                        Each row corresponds to an output neuron's "loudness"
                - 'column': Preserve each column's norm
                        Each column corresponds to an input feature's contribution
                - 'frobenius': Preserve overall matrix Frobenius norm
                        Simplest but least targeted preservation
            min_scale: Minimum scaling factor to prevent extreme shrinking (default 0.5)
                       Prevents capability damage from over-aggressive ablation
            max_scale: Maximum scaling factor to prevent extreme inflation (default 2.0)
                       Prevents instability from over-aggressive norm restoration
        """
        # Step 1: Store original norms based on mode
        if norm_mode == "row":
            # Preserve each output neuron's magnitude
            original_norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
        elif norm_mode == "column":
            # Preserve each input feature's contribution
            original_norms = torch.linalg.norm(matrix, dim=0, keepdim=True)
        else:  # frobenius
            # Preserve overall matrix magnitude (scalar)
            original_norm = torch.linalg.norm(matrix)

        # Step 2: Apply standard projection removal
        # W' = W - weight * (projector @ W)
        projection = weight * (projector @ matrix)
        matrix.sub_(projection)

        # Step 3: Rescale to preserve original norms within bounds
        if norm_mode == "row":
            new_norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
            # Compute scaling to restore original norms
            # We want: min_scale <= (new_norms * scaling) / original_norms <= max_scale
            # Solving for scaling bounds:
            # scaling >= min_scale * original_norms / new_norms
            # scaling <= max_scale * original_norms / new_norms
            scaling = torch.where(
                new_norms > EPSILON,
                original_norms / new_norms,
                torch.ones_like(original_norms),
            )
            # Compute actual bounds that enforce final norm ratio constraints
            min_allowed = min_scale * original_norms / (new_norms + EPSILON)
            max_allowed = max_scale * original_norms / (new_norms + EPSILON)
            scaling = torch.clamp(scaling, min=min_allowed, max=max_allowed)
            matrix.mul_(scaling)
        elif norm_mode == "column":
            new_norms = torch.linalg.norm(matrix, dim=0, keepdim=True)
            scaling = torch.where(
                new_norms > EPSILON,
                original_norms / new_norms,
                torch.ones_like(original_norms),
            )
            min_allowed = min_scale * original_norms / (new_norms + EPSILON)
            max_allowed = max_scale * original_norms / (new_norms + EPSILON)
            scaling = torch.clamp(scaling, min=min_allowed, max=max_allowed)
            matrix.mul_(scaling)
        else:  # frobenius
            new_norm = torch.linalg.norm(matrix)
            # Check both original and new norms to handle edge cases
            if original_norm > EPSILON and new_norm > EPSILON:
                scaling = original_norm / new_norm
                scaling = torch.clamp(
                    torch.tensor(scaling), min=min_scale, max=max_scale
                ).item()
                matrix.mul_(scaling)
            # If original_norm is near-zero, matrix was already near-zero - no scaling needed
            # If new_norm is near-zero but original wasn't, something went wrong - skip scaling to avoid instability

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        max_input_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "return_token_type_ids": False,
        }
        if max_input_length is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_input_length

        inputs = self.tokenizer(
            chat_prompts,
            **tokenizer_kwargs,
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
        """Extract residual (hidden state) vectors for prompts.

        Generates one token and returns the hidden states at that position
        for each prompt and layer.

        Args:
            prompts: List of prompts to extract residuals from

        Returns:
            Tensor of shape (n_prompts, n_layers, hidden_dim)

        Raises:
            ResidualExtractionError: If hidden state extraction fails
        """
        if not prompts:
            raise ResidualExtractionError(
                "Cannot extract residuals from empty prompt list."
            )

        try:
            # We only generate one token, and we return the residual vectors
            # at that token position, for each prompt and layer.
            # Truncate inputs to residual_max_tokens to bound memory usage --
            # output_hidden_states=True stores (n_layers * batch * seq_len * hidden_dim)
            # which can OOM on large models (14B+) with long sequences.
            _, outputs = self.generate(
                prompts,
                max_input_length=self.settings.residual_max_tokens,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            raise ResidualExtractionError(
                f"GPU out of memory during residual extraction for {len(prompts)} prompts. "
                f"Try reducing batch_size or residual_max_tokens."
            ) from e
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise ResidualExtractionError(
                    f"Out of memory during residual extraction: {e}"
                ) from e
            raise ResidualExtractionError(
                f"Runtime error during residual extraction: {e}"
            ) from e

        # Validate hidden states were returned
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ResidualExtractionError(
                "Model did not return hidden states. "
                "The model may not support output_hidden_states=True."
            )

        if len(outputs.hidden_states) == 0:
            raise ResidualExtractionError(
                "Model returned empty hidden_states. No tokens were generated."
            )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]
        del outputs  # Free the massive output object immediately

        if not hidden_states:
            raise ResidualExtractionError("Hidden states for generated token is empty.")

        try:
            # The returned tensor has shape (prompt, layer, component).
            # Use .clone() to break views into the original hidden state tensors,
            # allowing them to be freed by the garbage collector.
            residuals = torch.stack(
                # layer_hidden_states has shape (prompt, position, component),
                # so this extracts the hidden states at the end of each prompt,
                # and stacks them up over the layers.
                [
                    layer_hidden_states[:, -1, :].clone()
                    for layer_hidden_states in hidden_states
                ],
                dim=1,
            )
        except (IndexError, RuntimeError) as e:
            raise ResidualExtractionError(
                f"Failed to stack hidden states into residual tensor: {e}"
            ) from e
        finally:
            del hidden_states  # Free layer hidden states

        # Verify no NaN/Inf in residuals
        if torch.isnan(residuals).any() or torch.isinf(residuals).any():
            nan_count = torch.isnan(residuals).sum().item()
            inf_count = torch.isinf(residuals).sum().item()
            record_suppressed_error(
                error=None,
                context="get_residuals_nan_inf",
                module="model",
                severity="warning",
                details={
                    "n_prompts": len(prompts),
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "reason": "Residuals contain NaN/Inf values",
                },
            )
            logger.warning(
                f"Residuals contain {nan_count} NaN and {inf_count} Inf values"
            )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        """Extract residuals for prompts in batches.

        Args:
            prompts: List of prompts to extract residuals from

        Returns:
            Tensor of shape (n_prompts, n_layers, hidden_dim)

        Raises:
            ResidualExtractionError: If extraction fails for any batch
        """
        if not prompts:
            raise ResidualExtractionError(
                "Cannot extract residuals from empty prompt list."
            )

        residuals = []
        failed_batches = 0

        # Residual extraction uses output_hidden_states=True which stores
        # O(n_layers * seq_len * hidden_dim) per sample -- far more memory than
        # normal inference. Cap batch size to avoid OOM on large models (14B+).
        residual_batch_size = min(self.settings.batch_size, 16)
        batches = list(batchify(prompts, residual_batch_size))

        for batch_idx, batch in enumerate(batches):
            self._extract_batch_with_retry(batch, batch_idx, residuals, failed_batches)

        if not residuals:
            raise ResidualExtractionError(
                "No residuals extracted. All batches may have failed."
            )

        try:
            # Move CPU-offloaded tensors back to GPU for concatenation
            device = next(self.model.parameters()).device
            return torch.cat([r.to(device) for r in residuals], dim=0)
        except RuntimeError as e:
            raise ResidualExtractionError(
                f"Failed to concatenate residual batches: {e}"
            ) from e

    def _extract_batch_with_retry(
        self,
        batch: list[str],
        batch_idx: int,
        residuals: list,
        failed_batches: int,
    ) -> None:
        """Helper to extract residuals for a batch with retry logic."""
        try:
            # Move result to CPU to free GPU memory for subsequent batches.
            # Critical for large models (14B+) where hidden state extraction
            # uses most of the GPU memory.
            residuals.append(self.get_residuals(batch).cpu())
        except ResidualExtractionError:
            # Re-raise residual extraction errors directly
            raise
        except torch.cuda.OutOfMemoryError as e:
            failed_batches += 1
            if failed_batches > 3:
                raise ResidualExtractionError(
                    f"GPU OOM during residual extraction after {failed_batches} failed batches. "
                    f"Try reducing batch_size (current: {self.settings.batch_size})."
                ) from e
            record_suppressed_error(
                error=e,
                context="get_residuals_batched_oom",
                module="model",
                severity="warning",
                details={
                    "batch_idx": batch_idx,
                    "batch_size": len(batch),
                    "failed_batches": failed_batches,
                },
            )
            empty_cache()
            # Retry this batch after clearing cache
            try:
                residuals.append(self.get_residuals(batch).cpu())
            except torch.cuda.OutOfMemoryError as retry_e:
                raise ResidualExtractionError(
                    f"GPU OOM during residual extraction at batch {batch_idx} even after cache clear. "
                    f"Batch size: {len(batch)}. Try reducing batch_size."
                ) from retry_e

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
            batch_logprobs = self.get_logprobs(batch, n_tokens=n_tokens)

            # Handle variable-length generation when n_tokens > 1
            # Some batches may generate fewer tokens due to EOS
            if n_tokens > 1 and batch_logprobs.dim() == 3:
                actual_tokens = batch_logprobs.shape[1]
                if actual_tokens < n_tokens:
                    # Pad with very negative log-prob (near-zero probability)
                    # Using -100.0 instead of 0.0 since these are log probabilities
                    padding = torch.full(
                        (
                            batch_logprobs.shape[0],
                            n_tokens - actual_tokens,
                            batch_logprobs.shape[2],
                        ),
                        fill_value=-100.0,
                        device=batch_logprobs.device,
                        dtype=batch_logprobs.dtype,
                    )
                    batch_logprobs = torch.cat([batch_logprobs, padding], dim=1)
                elif actual_tokens > n_tokens:
                    # Truncate to expected n_tokens
                    batch_logprobs = batch_logprobs[:, :n_tokens, :]

            logprobs.append(batch_logprobs)

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

        Uses batched GPU operations with torch.einsum for ~20-30% speedup
        over sequential per-layer computation.

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
        device = good_residuals.device

        # Upcast to float32 for numerical stability
        # Shape: (n_samples, n_layers, hidden_dim)
        good_float = good_residuals.float()
        bad_float = bad_residuals.float()

        n_good = good_float.shape[0]
        n_bad = bad_float.shape[0]

        # Compute means for all layers at once
        # Shape: (n_layers, hidden_dim)
        good_means = good_float.mean(dim=0)
        bad_means = bad_float.mean(dim=0)

        # Center data for all layers in one operation
        # Shape: (n_samples, n_layers, hidden_dim)
        good_centered = good_float - good_means.unsqueeze(0)
        bad_centered = bad_float - bad_means.unsqueeze(0)

        # Batched covariance computation using einsum
        # For each layer l, compute: X^T @ X where X is (n_samples, hidden_dim)
        # einsum 'nld,nlD->ldD' computes: sum over n of (x_l[n,d] * x_l[n,D]) for each layer l
        # Result shape: (n_layers, hidden_dim, hidden_dim)
        cov_good = torch.einsum("nld,nlD->ldD", good_centered, good_centered) / max(
            n_good - 1, 1
        )
        cov_bad = torch.einsum("nld,nlD->ldD", bad_centered, bad_centered) / max(
            n_bad - 1, 1
        )

        # Contrastive covariance for all layers
        # Shape: (n_layers, hidden_dim, hidden_dim)
        cov_contrastive = cov_bad - alpha * cov_good

        # Batched eigendecomposition - torch.linalg.eigh supports batch dimensions
        # Input shape: (n_layers, hidden_dim, hidden_dim)
        # Output eigenvalues: (n_layers, hidden_dim), eigenvectors: (n_layers, hidden_dim, hidden_dim)
        try:
            eigenvalues_all, eigenvectors_all = torch.linalg.eigh(cov_contrastive)
        except (RuntimeError, torch.linalg.LinAlgError):
            # Fallback to mean difference for all layers if batched eigendecomposition fails
            diff = bad_means - good_means  # Shape: (n_layers, hidden_dim)
            diff = diff / (torch.linalg.norm(diff, dim=1, keepdim=True) + EPSILON)
            # Repeat for n_components
            directions = diff.unsqueeze(1).expand(
                -1, n_components, -1
            )  # (n_layers, n_components, hidden_dim)
            eigenvalues = torch.ones(n_layers, n_components, device=device)
            return PCAExtractionResult(
                directions=directions,
                eigenvalues=eigenvalues,
            )

        # eigenvalues are in ascending order, we want descending
        # Take last n_components (largest eigenvalues)
        # eigenvalues_all: (n_layers, hidden_dim) -> top_eigenvalues: (n_layers, n_components)
        top_eigenvalues = eigenvalues_all[:, -n_components:].flip(dims=[1])

        # eigenvectors_all: (n_layers, hidden_dim, hidden_dim)
        # eigenvectors_all[:, :, i] is the eigenvector for eigenvalue i
        # We want the last n_components eigenvectors (corresponding to largest eigenvalues)
        # Shape: (n_layers, hidden_dim, n_components) -> transpose to (n_layers, n_components, hidden_dim)
        top_eigenvectors = (
            eigenvectors_all[:, :, -n_components:].flip(dims=[2]).transpose(1, 2)
        )

        # Normalize each direction vector
        # Shape: (n_layers, n_components, hidden_dim)
        directions = F.normalize(top_eigenvectors, p=2, dim=2)

        return PCAExtractionResult(
            directions=directions,
            eigenvalues=top_eigenvalues,
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

    def extract_sacred_directions(
        self,
        capability_residuals: Tensor,
        baseline_residuals: Tensor,
        n_directions: int = 5,
    ) -> SacredDirectionResult:
        """Extract directions that encode capabilities to preserve.

        Uses contrastive PCA to find directions that maximize variance in
        capability prompts (e.g., MMLU) vs baseline text.

        Args:
            capability_residuals: Residuals from capability prompts (MMLU, etc.)
                Shape: (n_samples, n_layers, hidden_dim)
            baseline_residuals: Residuals from generic baseline text
                Shape: (n_samples, n_layers, hidden_dim)
            n_directions: Number of sacred directions to extract per layer

        Returns:
            SacredDirectionResult with directions and eigenvalues

        Raises:
            SacredDirectionError: If extraction fails
        """
        n_layers = capability_residuals.shape[1]
        hidden_dim = capability_residuals.shape[2]
        device = capability_residuals.device

        sacred_directions = torch.zeros(
            n_layers, n_directions, hidden_dim, device=device
        )
        eigenvalues = torch.zeros(n_layers, n_directions, device=device)

        # Contrastive PCA alpha - favor capability-specific variance
        alpha = SACRED_PCA_ALPHA

        for layer_idx in range(n_layers):
            try:
                # Extract layer activations and convert to float32 for stability
                cap_layer = capability_residuals[:, layer_idx, :].float()
                base_layer = baseline_residuals[:, layer_idx, :].float()

                # Center the data
                cap_centered = cap_layer - cap_layer.mean(dim=0)
                base_centered = base_layer - base_layer.mean(dim=0)

                # Compute covariance matrices using einsum for efficiency
                n_cap = cap_centered.shape[0]
                n_base = base_centered.shape[0]

                cov_cap = torch.einsum("ni,nj->ij", cap_centered, cap_centered) / (
                    n_cap - 1
                )
                cov_base = torch.einsum("ni,nj->ij", base_centered, base_centered) / (
                    n_base - 1
                )

                # Contrastive covariance: maximize capability variance, minimize baseline
                cov_contrastive = cov_cap - alpha * cov_base

                # Make symmetric (numerical stability)
                cov_contrastive = (cov_contrastive + cov_contrastive.T) / 2

                # GPU-accelerated eigendecomposition
                eig_vals, eig_vecs = torch.linalg.eigh(cov_contrastive)

                # Take top n_directions (eigenvalues are in ascending order)
                top_indices = torch.argsort(eig_vals, descending=True)[:n_directions]
                sacred_directions[layer_idx] = eig_vecs[:, top_indices].T
                eigenvalues[layer_idx] = eig_vals[top_indices]

            except Exception as e:
                logger.warning(
                    "Sacred direction extraction failed for layer",
                    layer=layer_idx,
                    error=str(e),
                )
                # Use mean difference as fallback
                mean_diff = capability_residuals[:, layer_idx, :].mean(
                    dim=0
                ) - baseline_residuals[:, layer_idx, :].mean(dim=0)
                sacred_directions[layer_idx, 0] = F.normalize(
                    mean_diff.float(), p=2, dim=0
                )
                # Zero out remaining directions
                if n_directions > 1:
                    sacred_directions[layer_idx, 1:] = 0
                eigenvalues[layer_idx] = 0

        # Normalize all directions
        sacred_directions = F.normalize(sacred_directions, p=2, dim=2)

        return SacredDirectionResult(
            directions=sacred_directions,
            eigenvalues=eigenvalues,
        )

    def _orthogonalize_single_against_sacred(
        self,
        direction: Tensor,
        sacred_directions: Tensor,
    ) -> Tensor:
        """Orthogonalize a single 2D direction against all sacred directions.

        Uses iterative Gram-Schmidt orthogonalization.

        Args:
            direction: Direction to orthogonalize, shape (n_layers, hidden_dim)
            sacred_directions: Sacred directions, shape (n_layers, n_sacred, hidden_dim)

        Returns:
            Orthogonalized direction, shape (n_layers, hidden_dim)
        """
        result = direction.clone()
        n_sacred = sacred_directions.shape[1]

        for sacred_idx in range(n_sacred):
            sacred = sacred_directions[:, sacred_idx, :]  # (n_layers, hidden_dim)

            # Project result onto sacred direction
            dot_product = (result * sacred).sum(dim=1, keepdim=True)
            projection = dot_product * sacred

            # Remove projection
            result = result - projection

        return result

    def orthogonalize_against_sacred(
        self,
        refusal_direction: Tensor,
        sacred_directions: Tensor,
    ) -> Tensor:
        """Orthogonalize refusal direction(s) against all sacred directions.

        Removes any component of the refusal direction that lies in the span
        of the sacred directions, mathematically guaranteeing that ablation
        cannot damage the encoded capabilities.

        Handles both 2D (single direction) and 3D (multiple directions) input.

        Args:
            refusal_direction: Direction(s) to orthogonalize
                - 2D shape: (n_layers, hidden_dim) - single direction
                - 3D shape: (n_layers, n_components, hidden_dim) - multiple directions
            sacred_directions: Sacred capability directions
                Shape: (n_layers, n_sacred, hidden_dim)

        Returns:
            Orthogonalized and renormalized direction(s), same shape as input

        Raises:
            SacredDirectionError: If orthogonalization results in near-zero direction
        """
        # Handle 3D case (multiple refusal directions)
        if refusal_direction.dim() == 3:
            n_components = refusal_direction.shape[1]
            result = torch.zeros_like(refusal_direction)

            for comp_idx in range(n_components):
                single_dir = refusal_direction[:, comp_idx, :]
                orthogonalized = self._orthogonalize_single_against_sacred(
                    single_dir, sacred_directions
                )
                result[:, comp_idx, :] = orthogonalized

            # Check for near-zero directions
            magnitudes = result.norm(dim=2)  # (n_layers, n_components)
            if (magnitudes < NEAR_ZERO).any():
                zero_count = (magnitudes < NEAR_ZERO).sum().item()
                raise SacredDirectionError(
                    f"Orthogonalization resulted in {zero_count} near-zero direction(s). "
                    "Refusal direction may lie within sacred direction span. "
                    "Try reducing n_sacred_directions or disabling sacred direction preservation."
                )

            # Renormalize
            return F.normalize(result, p=2, dim=2)

        # Handle 2D case (single refusal direction)
        result = self._orthogonalize_single_against_sacred(
            refusal_direction, sacred_directions
        )

        # Check for near-zero result
        magnitudes = result.norm(dim=1)  # (n_layers,)
        if (magnitudes < NEAR_ZERO).any():
            zero_layers = (magnitudes < NEAR_ZERO).sum().item()
            raise SacredDirectionError(
                f"Orthogonalization resulted in near-zero direction in {zero_layers} layer(s). "
                "Refusal direction may lie within sacred direction span. "
                "Try reducing n_sacred_directions or disabling sacred direction preservation."
            )

        # Renormalize
        return F.normalize(result, p=2, dim=1)

    def compute_sacred_overlap(
        self,
        refusal_direction: Tensor,
        sacred_directions: Tensor,
    ) -> dict[str, float]:
        """Compute overlap between refusal direction and sacred directions.

        Higher overlap means more potential for capability damage during ablation.

        Args:
            refusal_direction: Refusal direction, shape (n_layers, hidden_dim)
            sacred_directions: Sacred directions, shape (n_layers, n_sacred, hidden_dim)

        Returns:
            Dictionary with overlap statistics:
                - mean_overlap: Mean cosine similarity across all layers and sacred directions
                - max_overlap: Maximum cosine similarity
                - per_layer_mean: Mean overlap per layer
        """
        n_layers = refusal_direction.shape[0]
        n_sacred = sacred_directions.shape[1]

        overlaps = torch.zeros(n_layers, n_sacred, device=refusal_direction.device)

        for sacred_idx in range(n_sacred):
            sacred = sacred_directions[:, sacred_idx, :]  # (n_layers, hidden_dim)
            # Cosine similarity per layer
            cos_sim = F.cosine_similarity(refusal_direction, sacred, dim=1)
            overlaps[:, sacred_idx] = cos_sim.abs()  # Use absolute value

        return {
            "mean_overlap": overlaps.mean().item(),
            "max_overlap": overlaps.max().item(),
            "per_layer_mean": overlaps.mean(dim=1).tolist(),
        }

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
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> tuple[int, list[float], dict[str, int]]:
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
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA

        Returns:
            Tuple of (rounds_performed, list of KL values per round, aggregated stats)
        """
        kl_values = []
        aggregated_stats = {
            "rounds_completed": 0,
            "total_layers_processed": 0,
            "total_matrices_modified": 0,
            "errors_suppressed": 0,
        }

        for round_idx in range(max_rounds):
            print(f"  * Iterative ablation round {round_idx + 1}/{max_rounds}...")

            # Extract residual directions from current model state
            try:
                good_residuals = self.get_residuals_batched(good_prompts)
                bad_residuals = self.get_residuals_batched(bad_prompts)
            except ResidualExtractionError as e:
                record_suppressed_error(
                    error=e,
                    context="abliterate_iterative_residuals",
                    module="model",
                    severity="error",
                    details={"round_idx": round_idx},
                    include_traceback=True,
                )
                aggregated_stats["errors_suppressed"] += 1
                logger.error(f"Residual extraction failed at round {round_idx}: {e}")
                break

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
            try:
                round_stats = self.abliterate(
                    refusal_directions,
                    None,
                    parameters,
                    layer_profiles,
                    use_mpoa=use_mpoa,
                    mpoa_norm_mode=mpoa_norm_mode,
                    mpoa_min_scale=mpoa_min_scale,
                    mpoa_max_scale=mpoa_max_scale,
                )
                aggregated_stats["total_layers_processed"] += round_stats.get(
                    "layers_processed", 0
                )
                aggregated_stats["total_matrices_modified"] += round_stats.get(
                    "matrices_modified", 0
                )
                aggregated_stats["errors_suppressed"] += round_stats.get(
                    "errors_suppressed", 0
                )
                aggregated_stats["rounds_completed"] += 1
            except WeightModificationError as e:
                record_suppressed_error(
                    error=e,
                    context="abliterate_iterative_ablation",
                    module="model",
                    severity="error",
                    details={"round_idx": round_idx},
                    include_traceback=True,
                )
                aggregated_stats["errors_suppressed"] += 1
                logger.error(f"Abliteration failed at round {round_idx}: {e}")
                break

            # Capability guard - measure KL after each round
            if (
                round_idx < max_rounds - 1
                and base_logprobs is not None
                and kl_check_prompts is not None
            ):
                try:
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
                except Exception as e:
                    record_suppressed_error(
                        error=e,
                        context="abliterate_iterative_kl_check",
                        module="model",
                        severity="warning",
                        details={"round_idx": round_idx},
                    )
                    aggregated_stats["errors_suppressed"] += 1

            # Clean up
            del good_residuals, bad_residuals, raw_difference
            empty_cache()

        # Log summary
        if aggregated_stats["errors_suppressed"] > 0:
            logger.warning(
                "Iterative ablation completed with errors",
                rounds_completed=aggregated_stats["rounds_completed"],
                errors_suppressed=aggregated_stats["errors_suppressed"],
            )
        else:
            logger.info(
                "Iterative ablation completed",
                rounds_completed=aggregated_stats["rounds_completed"],
                total_matrices_modified=aggregated_stats["total_matrices_modified"],
            )

        return round_idx + 1, kl_values, aggregated_stats

    def abliterate_multi_direction(
        self,
        refusal_directions: Tensor,
        direction_weights: list[float],
        parameters: dict[str, "AbliterationParameters"],
        layer_profiles: list["LayerRangeProfile"] | None = None,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> dict[str, int]:
        """Abliterate multiple refusal directions with configurable weights.

        Args:
            refusal_directions: Shape (n_layers, n_components, hidden_dim)
            direction_weights: Weight for each component
            parameters: Abliteration parameters for each component
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA
            mpoa_min_scale: Minimum scaling factor for MPOA (default 0.5)
            mpoa_max_scale: Maximum scaling factor for MPOA (default 2.0)

        Returns:
            Dict with ablation statistics:
                - 'directions_processed': Number of directions abliterated
                - 'directions_skipped': Number of directions skipped (weight <= 0)
                - 'total_layers_processed': Total layers across all directions
                - 'total_matrices_modified': Total matrices across all directions
                - 'errors_suppressed': Number of non-fatal errors encountered
        """
        stats = {
            "directions_processed": 0,
            "directions_skipped": 0,
            "total_layers_processed": 0,
            "total_matrices_modified": 0,
            "errors_suppressed": 0,
        }

        n_components = refusal_directions.shape[1]

        for component_idx in range(min(n_components, len(direction_weights))):
            weight_multiplier = direction_weights[component_idx]
            if weight_multiplier <= 0:
                stats["directions_skipped"] += 1
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
            try:
                direction_stats = self.abliterate(
                    single_direction,
                    None,
                    scaled_parameters,
                    layer_profiles,
                    use_mpoa=use_mpoa,
                    mpoa_norm_mode=mpoa_norm_mode,
                    mpoa_min_scale=mpoa_min_scale,
                    mpoa_max_scale=mpoa_max_scale,
                )
                stats["directions_processed"] += 1
                stats["total_layers_processed"] += direction_stats.get(
                    "layers_processed", 0
                )
                stats["total_matrices_modified"] += direction_stats.get(
                    "matrices_modified", 0
                )
                stats["errors_suppressed"] += direction_stats.get(
                    "errors_suppressed", 0
                )
            except WeightModificationError as e:
                record_suppressed_error(
                    error=e,
                    context="abliterate_multi_direction",
                    module="model",
                    severity="error",
                    details={
                        "component_idx": component_idx,
                        "weight_multiplier": weight_multiplier,
                    },
                    include_traceback=True,
                )
                stats["errors_suppressed"] += 1
                logger.error(
                    f"Multi-direction ablation failed at component {component_idx}: {e}"
                )
                # Continue with remaining directions

        # Log summary
        if stats["errors_suppressed"] > 0:
            logger.warning(
                "Multi-direction ablation completed with errors",
                directions_processed=stats["directions_processed"],
                directions_skipped=stats["directions_skipped"],
                errors_suppressed=stats["errors_suppressed"],
            )
        else:
            logger.info(
                "Multi-direction ablation completed",
                directions_processed=stats["directions_processed"],
                directions_skipped=stats["directions_skipped"],
                total_matrices_modified=stats["total_matrices_modified"],
            )

        return stats

    # Phase 2: Supervised Refusal Probing
    def get_refusal_directions_supervised(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        bad_prompts: list[str],
        evaluator: "Evaluator",
        min_probe_accuracy: float = 0.65,
    ) -> SupervisedExtractionResult:
        """Extract refusal directions using supervised probing.

        Unlike PCA which finds variance directions, this finds the actual
        decision boundary between refuse/comply by training linear classifiers.

        The method:
        1. Generates responses to bad prompts
        2. Labels each prompt as "triggered refusal" or "got compliance"
        3. Trains a logistic regression probe per layer
        4. Uses the learned weights as refusal directions

        Raises SupervisedProbeError if:
        - Insufficient class balance (< 10 samples per class)
        - Mean probe accuracy below min_probe_accuracy threshold

        Args:
            good_residuals: Residuals from harmless prompts
                Shape: (n_good, n_layers, hidden_dim)
            bad_residuals: Residuals from harmful prompts
                Shape: (n_bad, n_layers, hidden_dim)
            bad_prompts: List of bad prompts (must match bad_residuals length)
            evaluator: Evaluator instance for refusal detection
            min_probe_accuracy: Minimum cross-validation accuracy required
                Falls back to PCA if mean accuracy is below this threshold

        Returns:
            SupervisedExtractionResult containing directions and accuracies

        Raises:
            SupervisedProbeError: If class imbalance (< 10 samples per class) or
                mean probe accuracy below min_probe_accuracy threshold
        """
        n_layers = good_residuals.shape[1]
        device = good_residuals.device
        directions: list[Tensor] = []
        accuracies: list[float] = []

        # Validate prompt/residual alignment
        if len(bad_prompts) != bad_residuals.shape[0]:
            print(
                f"[yellow]Warning: Prompt count ({len(bad_prompts)}) != residual count "
                f"({bad_residuals.shape[0]}). Using first {min(len(bad_prompts), bad_residuals.shape[0])} samples.[/yellow]"
            )
            n_samples = min(len(bad_prompts), bad_residuals.shape[0])
            bad_prompts = bad_prompts[:n_samples]
            bad_residuals = bad_residuals[:n_samples]

        print("  * Generating responses to classify refusal behavior...")
        bad_responses = self.get_responses_batched(bad_prompts)

        # Label based on actual responses using the evaluator
        refusal_mask = torch.tensor(
            [evaluator.is_refusal(r) for r in bad_responses],
            dtype=torch.bool,
            device=device,
        )

        # Check for class balance
        n_refusals = int(refusal_mask.sum().item())
        n_comply = len(refusal_mask) - n_refusals

        print(
            f"  * Response classification: {n_refusals} refusals, {n_comply} compliant"
        )

        if n_refusals < 10 or n_comply < 10:
            raise SupervisedProbeError(
                f"Class imbalance prevents supervised probing: refusals={n_refusals}, comply={n_comply}. "
                f"Need at least 10 samples per class. Either your bad_prompts dataset doesn't trigger "
                f"enough refusals, or the model complies with too many harmful prompts. "
                f"Try using a different bad_prompts dataset or disable supervised probing."
            )

        # Separate residuals by actual behavior
        refusal_residuals = bad_residuals[refusal_mask]
        comply_residuals = torch.cat(
            [
                good_residuals,
                bad_residuals[~refusal_mask],
            ],
            dim=0,
        )

        print(f"  * Training {n_layers} layer probes in parallel...")

        # GPU-accelerated batched probe training: trains ALL layer probes simultaneously
        # on GPU using PyTorch, replacing the CPU-bound sklearn + joblib approach.
        # For 14B models (49 layers, 5120 hidden_dim), this reduces training from
        # ~40 minutes on CPU to seconds on GPU.
        use_gpu_probes = torch.cuda.is_available() and device.type == "cuda"

        if use_gpu_probes:
            directions, accuracies = self._train_probes_gpu_batched(
                refusal_residuals, comply_residuals, n_layers, device
            )
        else:
            # CPU fallback: use sklearn LogisticRegression with joblib parallelism
            directions, accuracies = self._train_probes_sklearn_fallback(
                refusal_residuals, comply_residuals, n_layers, device
            )

        mean_acc = sum(accuracies) / len(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)

        print(
            f"  * Probe accuracies: min={min_acc:.2f}, max={max_acc:.2f}, mean={mean_acc:.2f}"
        )

        # Check accuracy threshold - raise error if too low
        if mean_acc < min_probe_accuracy:
            raise SupervisedProbeError(
                f"Probe accuracy ({mean_acc:.2f}) below threshold ({min_probe_accuracy}). "
                f"The model's internal representations don't clearly distinguish refusal from compliance. "
                f"This may indicate the model uses different mechanisms for refusal, or the prompts "
                f"don't trigger consistent behavior. Try lowering min_probe_accuracy or disable supervised probing."
            )

        return SupervisedExtractionResult(
            directions=torch.stack(directions),
            accuracies=accuracies,
        )

    def _train_probes_gpu_batched(
        self,
        refusal_residuals: Tensor,
        comply_residuals: Tensor,
        n_layers: int,
        device: torch.device,
    ) -> tuple[list[Tensor], list[float]]:
        """Train all layer probes simultaneously on GPU using batched PyTorch logistic regression.

        Stacks all layer data into batch tensors and runs gradient-based optimization
        for all probes in parallel. For 49 layers with 5120 hidden_dim, this completes
        in seconds vs ~40 minutes with CPU sklearn.

        Args:
            refusal_residuals: Shape (n_refusal, n_layers, hidden_dim)
            comply_residuals: Shape (n_comply, n_layers, hidden_dim)
            n_layers: Number of layers
            device: CUDA device

        Returns:
            Tuple of (directions list, accuracies list) matching sklearn output format
        """
        n_refusal = refusal_residuals.shape[0]
        n_comply = comply_residuals.shape[0]
        hidden_dim = refusal_residuals.shape[2]

        # Build X: (n_layers, n_samples, hidden_dim) and y: (n_layers, n_samples)
        # refusal_residuals is (n_refusal, n_layers, hidden_dim) -> permute to (n_layers, n_refusal, hidden_dim)
        X_refusal = refusal_residuals.permute(
            1, 0, 2
        ).float()  # (n_layers, n_refusal, hidden_dim)
        X_comply = comply_residuals.permute(
            1, 0, 2
        ).float()  # (n_layers, n_comply, hidden_dim)
        X = torch.cat([X_refusal, X_comply], dim=1)  # (n_layers, n_samples, hidden_dim)

        n_samples = n_refusal + n_comply

        # Labels: 1 for refusal, 0 for comply -- broadcast across layers
        y = torch.cat(
            [
                torch.ones(n_refusal, device=device),
                torch.zeros(n_comply, device=device),
            ]
        ).float()  # (n_samples,)
        y = y.unsqueeze(0).expand(n_layers, -1)  # (n_layers, n_samples)

        # Balanced class weights: weight_refusal = n_samples / (2 * n_refusal),
        # weight_comply = n_samples / (2 * n_comply)
        w_refusal = n_samples / (2.0 * n_refusal)
        w_comply = n_samples / (2.0 * n_comply)
        sample_weights = torch.cat(
            [
                torch.full((n_refusal,), w_refusal, device=device),
                torch.full((n_comply,), w_comply, device=device),
            ]
        ).float()  # (n_samples,)
        sample_weights = sample_weights.unsqueeze(0).expand(
            n_layers, -1
        )  # (n_layers, n_samples)

        # Initialize weights and bias for all layers: W (n_layers, hidden_dim), b (n_layers,)
        W = torch.zeros(n_layers, hidden_dim, device=device, dtype=torch.float32)
        b = torch.zeros(n_layers, device=device, dtype=torch.float32)
        W.requires_grad_(True)
        b.requires_grad_(True)

        # L2 regularization strength (sklearn C=1.0 means lambda=1/C=1.0,
        # but sklearn normalizes differently; match their scale)
        l2_lambda = 1.0 / n_samples

        # Use LBFGS for fast convergence (matches sklearn's lbfgs solver)
        optimizer = torch.optim.LBFGS(
            [W, b],
            max_iter=200,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            # Logits: (n_layers, n_samples) = batched matmul + bias
            logits = torch.bmm(X, W.unsqueeze(2)).squeeze(2) + b.unsqueeze(1)
            # Weighted binary cross-entropy with logits
            bce = F.binary_cross_entropy_with_logits(
                logits, y, weight=sample_weights, reduction="none"
            )
            loss = bce.mean(dim=1).sum()  # sum across layers, mean across samples
            # L2 regularization on weights
            loss = loss + l2_lambda * (W * W).sum()
            loss.backward()
            return loss

        # Run LBFGS optimization
        optimizer.step(closure)

        # Extract trained weights (detach from computation graph)
        W_final = W.detach()  # (n_layers, hidden_dim)

        # Normalize each layer's weight vector
        norms = W_final.norm(dim=1, keepdim=True) + EPSILON  # (n_layers, 1)
        W_normalized = W_final / norms  # (n_layers, hidden_dim)

        # Compute per-layer accuracies using k-fold cross-validation on GPU
        # Use a simple 5-fold CV approximation
        n_splits = min(5, n_samples // 2) if n_samples >= 10 else 0

        if n_splits >= 2:
            accuracies = self._gpu_cross_validate(
                X, y, sample_weights, n_layers, hidden_dim, n_samples, n_splits, device
            )
        else:
            accuracies = [0.5] * n_layers

        # Convert to list of tensors
        directions = [W_normalized[i].to(device) for i in range(n_layers)]

        print(f"  * GPU batched probe training complete ({n_layers} layers)")

        return directions, accuracies

    def _gpu_cross_validate(
        self,
        X: Tensor,
        y: Tensor,
        sample_weights: Tensor,
        n_layers: int,
        hidden_dim: int,
        n_samples: int,
        n_splits: int,
        device: torch.device,
    ) -> list[float]:
        """Run approximate k-fold cross-validation for all layers on GPU.

        Trains n_splits models per layer (all layers batched) and computes held-out accuracy.
        Uses fewer LBFGS iterations for CV since we only need accuracy estimates.

        Args:
            X: Input features, shape (n_layers, n_samples, hidden_dim)
            y: Labels, shape (n_layers, n_samples)
            sample_weights: Per-sample weights, shape (n_layers, n_samples)
            n_layers: Number of layers
            hidden_dim: Hidden dimension size
            n_samples: Total number of samples
            n_splits: Number of CV folds
            device: CUDA device

        Returns:
            List of mean CV accuracy per layer
        """
        # Create fold indices
        indices = torch.randperm(n_samples, device=device)
        fold_size = n_samples // n_splits

        # Accumulate per-layer accuracy across folds
        fold_accuracies = torch.zeros(n_layers, n_splits, device=device)

        for fold in range(n_splits):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_splits - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = torch.cat([indices[:val_start], indices[val_end:]])

            X_train = X[:, train_idx, :]  # (n_layers, n_train, hidden_dim)
            y_train = y[:, train_idx]  # (n_layers, n_train)
            w_train = sample_weights[:, train_idx]
            X_val = X[:, val_idx, :]  # (n_layers, n_val, hidden_dim)
            y_val = y[:, val_idx]  # (n_layers, n_val)

            n_train = X_train.shape[1]
            l2_lambda = 1.0 / n_train

            # Train fold model (fewer iterations for speed)
            W_cv = torch.zeros(n_layers, hidden_dim, device=device, dtype=torch.float32)
            b_cv = torch.zeros(n_layers, device=device, dtype=torch.float32)
            W_cv.requires_grad_(True)
            b_cv.requires_grad_(True)

            optimizer_cv = torch.optim.LBFGS(
                [W_cv, b_cv],
                max_iter=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-5,
                tolerance_change=1e-7,
            )

            def cv_closure() -> Tensor:
                optimizer_cv.zero_grad()
                logits = torch.bmm(X_train, W_cv.unsqueeze(2)).squeeze(
                    2
                ) + b_cv.unsqueeze(1)
                bce = F.binary_cross_entropy_with_logits(
                    logits, y_train, weight=w_train, reduction="none"
                )
                loss = bce.mean(dim=1).sum()
                loss = loss + l2_lambda * (W_cv * W_cv).sum()
                loss.backward()
                return loss

            optimizer_cv.step(cv_closure)

            # Evaluate on validation set
            with torch.no_grad():
                val_logits = torch.bmm(X_val, W_cv.unsqueeze(2)).squeeze(
                    2
                ) + b_cv.unsqueeze(1)
                val_preds = (val_logits > 0.0).float()
                fold_accuracies[:, fold] = (val_preds == y_val).float().mean(dim=1)

        # Mean accuracy across folds for each layer
        mean_accs = fold_accuracies.mean(dim=1)  # (n_layers,)
        return mean_accs.cpu().tolist()

    def _train_probes_sklearn_fallback(
        self,
        refusal_residuals: Tensor,
        comply_residuals: Tensor,
        n_layers: int,
        device: torch.device,
    ) -> tuple[list[Tensor], list[float]]:
        """CPU fallback: train probes using sklearn LogisticRegression with joblib parallelism.

        Used when CUDA is not available. Identical to the original implementation.

        Args:
            refusal_residuals: Shape (n_refusal, n_layers, hidden_dim)
            comply_residuals: Shape (n_comply, n_layers, hidden_dim)
            n_layers: Number of layers
            device: Target device for output tensors

        Returns:
            Tuple of (directions list, accuracies list)
        """
        from joblib import Parallel, delayed

        refusal_np = refusal_residuals.cpu().numpy()
        comply_np = comply_residuals.cpu().numpy()

        def train_probe_for_layer(
            layer_idx: int,
            refusal_data: np.ndarray,
            comply_data: np.ndarray,
        ) -> tuple[np.ndarray, float]:
            """Train a single probe for one layer (designed for parallel execution)."""
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            X = np.concatenate(
                [refusal_data[:, layer_idx, :], comply_data[:, layer_idx, :]],
                axis=0,
            )
            y = np.concatenate([np.ones(len(refusal_data)), np.zeros(len(comply_data))])

            clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )

            if len(X) >= 10:
                n_splits = min(5, len(X) // 2)
                if n_splits >= 2:
                    cv_scores = cross_val_score(clf, X, y, cv=n_splits)
                    accuracy = float(cv_scores.mean())
                else:
                    accuracy = 0.5
            else:
                accuracy = 0.5

            clf.fit(X, y)

            weights = clf.coef_[0]
            weights = weights / (np.linalg.norm(weights) + EPSILON)

            return weights, accuracy

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(train_probe_for_layer)(layer_idx, refusal_np, comply_np)
            for layer_idx in range(n_layers)
        )

        directions: list[Tensor] = []
        accuracies: list[float] = []
        for weights_np, accuracy in results:
            directions.append(torch.from_numpy(weights_np).float().to(device))
            accuracies.append(accuracy)

        print(f"  * sklearn CPU fallback probe training complete ({n_layers} layers)")

        return directions, accuracies

    # Phase 3: Activation-Scaled Weight Calibration
    def compute_refusal_activation_stats(
        self,
        refusal_directions: Tensor,
        bad_residuals: Tensor,
        target_layer_frac: float = 0.6,
    ) -> RefusalActivationStats:
        """Compute statistics about refusal activations for weight calibration.

        Measures how strongly the bad prompts activate the refusal direction,
        which can be used to calibrate ablation weights appropriately.

        Args:
            refusal_directions: Refusal directions, shape (n_layers, hidden_dim)
            bad_residuals: Residuals from harmful prompts,
                shape (n_prompts, n_layers, hidden_dim)
            target_layer_frac: Which layer to measure (as fraction of total layers).
                Default 0.6 targets middle-to-late layers where refusal is typically strongest.

        Returns:
            RefusalActivationStats with projection statistics
        """
        n_layers = bad_residuals.shape[1]
        target_layer = int(n_layers * target_layer_frac)
        target_layer = min(target_layer, n_layers - 1)  # Clamp to valid range

        # Get direction at target layer
        direction = refusal_directions[target_layer]

        # Compute projections for all prompts (vectorized)
        layer_residuals = bad_residuals[:, target_layer, :]  # (n_prompts, hidden_dim)
        projections = torch.abs(
            torch.matmul(layer_residuals, direction)  # (n_prompts,)
        )

        # Convert to numpy for percentile calculation
        projections_np = projections.cpu().numpy()

        return RefusalActivationStats(
            mean_projection=float(projections_np.mean()),
            std_projection=float(projections_np.std()),
            percentile_25=float(np.percentile(projections_np, 25)),
            percentile_75=float(np.percentile(projections_np, 75)),
            percentile_95=float(np.percentile(projections_np, 95)),
            n_samples=len(projections_np),
            target_layer=target_layer,
        )

    def compute_calibrated_weight(
        self,
        stats: RefusalActivationStats,
        base_weight: float = 1.0,
        target_percentile: float = 0.75,
        min_factor: float = 0.5,
        max_factor: float = 2.0,
    ) -> float:
        """Compute calibrated ablation weight based on activation statistics.

        The idea: instead of fixed weights, scale based on how strongly refusal
        activations appear in this specific model/prompt combination.

        The calibration factor is computed as: target_percentile_value / mean_projection
        - Higher percentile (e.g., 0.95) → targets stronger activations → higher weight
        - Lower percentile (e.g., 0.5) → targets average activations → weight ≈ 1.0
        - Lower percentile (e.g., 0.25) → targets weaker activations → lower weight

        Args:
            stats: Activation statistics from compute_refusal_activation_stats
            base_weight: Starting weight before calibration (default 1.0)
            target_percentile: Which percentile of activations to target.
                Higher = more aggressive ablation (scales up weight).
                Lower = more conservative ablation (scales down weight).
            min_factor: Minimum calibration factor (prevents under-ablation)
            max_factor: Maximum calibration factor (prevents over-ablation)

        Returns:
            Calibrated weight = base_weight * calibration_factor
        """
        # Get target value from statistics
        target_value = stats.get_percentile_value(target_percentile)

        # Compute calibration factor
        # Scale weight inversely to activation strength relative to mean
        if stats.mean_projection > EPSILON:
            calibration_factor = target_value / stats.mean_projection
        else:
            # If mean projection is near zero, refusal direction may be weak
            calibration_factor = 1.0
            logger.warning(
                "Activation calibration fallback triggered",
                mean_projection=stats.mean_projection,
                reason="Mean projection near zero, refusal direction may be weak",
                calibration_factor=calibration_factor,
            )
            print(
                f"[yellow]  * Warning: Weak refusal activation (mean={stats.mean_projection:.6f}), "
                f"using calibration factor 1.0[/yellow]"
            )

        # Clamp to reasonable range to prevent extreme values
        calibration_factor = max(min_factor, min(max_factor, calibration_factor))

        return base_weight * calibration_factor

    def get_calibrated_parameters(
        self,
        parameters: dict[str, "AbliterationParameters"],
        stats: RefusalActivationStats,
        target_percentile: float = 0.75,
        min_factor: float = 0.5,
        max_factor: float = 2.0,
    ) -> dict[str, "AbliterationParameters"]:
        """Apply activation-based calibration to abliteration parameters.

        Scales the max_weight and min_weight of each component based on
        the computed activation statistics.

        Args:
            parameters: Original abliteration parameters per component
            stats: Activation statistics from compute_refusal_activation_stats
            target_percentile: Which percentile of activations to target
            min_factor: Minimum calibration factor
            max_factor: Maximum calibration factor

        Returns:
            New dict of AbliterationParameters with calibrated weights
        """
        calibration_factor = self.compute_calibrated_weight(
            stats,
            base_weight=1.0,
            target_percentile=target_percentile,
            min_factor=min_factor,
            max_factor=max_factor,
        )

        print(
            f"  * Activation calibration: factor={calibration_factor:.3f} "
            f"(target_percentile={target_percentile}, "
            f"mean_proj={stats.mean_projection:.4f}, "
            f"p75={stats.percentile_75:.4f})"
        )

        calibrated_parameters = {}
        for component, params in parameters.items():
            calibrated_parameters[component] = AbliterationParameters(
                max_weight=params.max_weight * calibration_factor,
                max_weight_position=params.max_weight_position,
                min_weight=params.min_weight * calibration_factor,
                min_weight_distance=params.min_weight_distance,
            )

        return calibrated_parameters

    # Phase 4: Concept Cones Clustering
    def get_refusal_directions_concept_cones(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        n_cones: int = 5,
        min_cone_size: int = 10,
        directions_per_cone: int = 2,
        min_silhouette_score: float = 0.1,
    ) -> ConceptConeExtractionResult:
        """Extract refusal directions using concept cone clustering.

        Clusters harmful prompts by their residual patterns to identify
        different categories of harmful content (e.g., violence, fraud,
        self-harm), then extracts category-specific refusal directions.

        Raises ConceptConeError if:
        - Silhouette score below threshold (poor clustering quality)
        - No valid cones extracted (all clusters below min_cone_size)

        Args:
            good_residuals: Residuals from harmless prompts
                Shape: (n_good, n_layers, hidden_dim)
            bad_residuals: Residuals from harmful prompts
                Shape: (n_bad, n_layers, hidden_dim)
            n_cones: Number of concept cones to extract
            min_cone_size: Minimum prompts required per cone
            directions_per_cone: Number of PCA directions per cone
            min_silhouette_score: Minimum clustering quality score

        Returns:
            ConceptConeExtractionResult containing extracted concept cones

        Raises:
            ConceptConeError: If silhouette score below threshold or no valid cones extracted
        """
        print("  * Extracting concept cones...")

        extractor = ConceptConeExtractor(
            n_cones=n_cones,
            n_directions_per_cone=directions_per_cone,
            min_cluster_size=min_cone_size,
            min_silhouette_score=min_silhouette_score,
        )

        result = extractor.extract(good_residuals, bad_residuals, self)

        # Check if clustering quality is acceptable
        if (
            result.silhouette_score < min_silhouette_score
            and result.n_clusters_valid > 1
        ):
            raise ConceptConeError(
                f"Poor clustering quality: silhouette score ({result.silhouette_score:.3f}) "
                f"below threshold ({min_silhouette_score}). The harmful prompts don't form "
                f"distinct categories. This may indicate the prompts are too homogeneous or "
                f"the model uses similar refusal patterns for all categories. "
                f"Try lowering min_silhouette_score or disable concept cones."
            )

        # Check if we have any valid cones
        if not result.cones:
            raise ConceptConeError(
                "No valid concept cones extracted. All clusters were below the minimum size "
                f"threshold ({min_cone_size}). Either increase the number of bad_prompts, "
                f"lower min_cone_size, or disable concept cones."
            )

        return result

    def abliterate_concept_cones(
        self,
        cone_result: "ConceptConeExtractionResult",
        parameters: dict[str, "AbliterationParameters"],
        direction_weights: list[float] | None = None,
        layer_profiles: list["LayerRangeProfile"] | None = None,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> dict[str, int]:
        """Abliterate using concept cone-specific directions.

        Applies ablation for each concept cone's primary direction,
        weighted by the cone's size (number of prompts).

        Args:
            cone_result: Result from get_refusal_directions_concept_cones
            parameters: Abliteration parameters for each component
            direction_weights: Optional weights for each direction component.
                If None, uses first direction only for each cone.
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA
            mpoa_min_scale: Minimum scaling factor for MPOA (default 0.5)
            mpoa_max_scale: Maximum scaling factor for MPOA (default 2.0)

        Returns:
            Dict with ablation statistics:
                - 'cones_processed': Number of cones abliterated
                - 'total_layers_processed': Total layers across all cones
                - 'total_matrices_modified': Total matrices across all cones
                - 'errors_suppressed': Number of non-fatal errors encountered
        """
        stats = {
            "cones_processed": 0,
            "total_layers_processed": 0,
            "total_matrices_modified": 0,
            "errors_suppressed": 0,
        }

        if not cone_result.cones:
            print("[yellow]Warning: No concept cones to abliterate.[/yellow]")
            logger.warning("Concept cone ablation skipped: no cones extracted")
            return stats

        total_size = sum(cone.size for cone in cone_result.cones)

        print(f"  * Abliterating {len(cone_result.cones)} concept cones...")
        logger.info(
            "Applying concept cone ablation",
            n_cones=len(cone_result.cones),
            total_prompts=total_size,
            use_mpoa=use_mpoa,
        )

        for cone in cone_result.cones:
            # Weight by relative size of this cone
            cone_weight = cone.size / total_size if total_size > 0 else 1.0

            print(
                f"    * Cone {cone.cluster_id}: size={cone.size}, "
                f"weight={cone_weight:.2f}"
            )

            # Scale parameters by cone weight
            scaled_parameters = {}
            for component, params in parameters.items():
                scaled_parameters[component] = AbliterationParameters(
                    max_weight=params.max_weight * cone_weight,
                    max_weight_position=params.max_weight_position,
                    min_weight=params.min_weight * cone_weight,
                    min_weight_distance=params.min_weight_distance,
                )

            # Use the primary direction for this cone
            # cone.directions has shape (n_layers, n_components, hidden_dim)
            try:
                if direction_weights is None:
                    # Use first direction only
                    primary_direction = cone.directions[:, 0, :]
                    cone_stats = self.abliterate(
                        primary_direction,
                        None,
                        scaled_parameters,
                        layer_profiles,
                        use_mpoa=use_mpoa,
                        mpoa_norm_mode=mpoa_norm_mode,
                        mpoa_min_scale=mpoa_min_scale,
                        mpoa_max_scale=mpoa_max_scale,
                    )
                    stats["total_layers_processed"] += cone_stats.get(
                        "layers_processed", 0
                    )
                    stats["total_matrices_modified"] += cone_stats.get(
                        "matrices_modified", 0
                    )
                    stats["errors_suppressed"] += cone_stats.get("errors_suppressed", 0)
                else:
                    # Use multiple directions with weights
                    multi_stats = self.abliterate_multi_direction(
                        cone.directions,
                        direction_weights,
                        scaled_parameters,
                        layer_profiles,
                        use_mpoa=use_mpoa,
                        mpoa_norm_mode=mpoa_norm_mode,
                        mpoa_min_scale=mpoa_min_scale,
                        mpoa_max_scale=mpoa_max_scale,
                    )
                    stats["total_layers_processed"] += multi_stats.get(
                        "total_layers_processed", 0
                    )
                    stats["total_matrices_modified"] += multi_stats.get(
                        "total_matrices_modified", 0
                    )
                    stats["errors_suppressed"] += multi_stats.get(
                        "errors_suppressed", 0
                    )
                stats["cones_processed"] += 1
            except (WeightModificationError, Exception) as e:
                record_suppressed_error(
                    error=e,
                    context="abliterate_concept_cones",
                    module="model",
                    severity="error",
                    details={
                        "cluster_id": cone.cluster_id,
                        "cone_size": cone.size,
                        "cone_weight": cone_weight,
                    },
                    include_traceback=True,
                )
                stats["errors_suppressed"] += 1
                logger.error(
                    f"Concept cone ablation failed for cone {cone.cluster_id}: {e}"
                )
                # Continue with remaining cones

        # Log summary
        if stats["errors_suppressed"] > 0:
            logger.warning(
                "Concept cone ablation completed with errors",
                cones_processed=stats["cones_processed"],
                errors_suppressed=stats["errors_suppressed"],
            )
        else:
            logger.info(
                "Concept cone ablation completed",
                cones_processed=stats["cones_processed"],
                total_matrices_modified=stats["total_matrices_modified"],
            )

        return stats

    def get_refusal_directions_ensemble(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        bad_prompts: list[str],
        evaluator: "Evaluator",
        probe_weight: float = 0.7,
        pca_weight: float = 0.3,
        min_probe_accuracy: float = 0.65,
    ) -> Tensor:
        """Extract refusal directions using ensemble of supervised probe and PCA.

        Combines the strengths of both approaches:
        - Supervised probe: directly learns the refuse/comply decision boundary
        - PCA: finds directions of maximum variance difference

        Raises RuntimeError if supervised probing fails (class imbalance or low accuracy).

        Args:
            good_residuals: Residuals from harmless prompts
            bad_residuals: Residuals from harmful prompts
            bad_prompts: List of bad prompts (must match bad_residuals length)
            evaluator: Evaluator instance for refusal detection
            probe_weight: Weight for supervised probe direction (default 0.7)
            pca_weight: Weight for PCA direction (default 0.3)
            min_probe_accuracy: Minimum accuracy for supervised probe

        Returns:
            Ensemble directions tensor of shape (n_layers, hidden_dim)

        Raises:
            RuntimeError: If supervised probing fails due to class imbalance or low accuracy
        """
        # Normalize weights to sum to 1.0
        total_weight = probe_weight + pca_weight
        if total_weight != 1.0:
            probe_weight = probe_weight / total_weight
            pca_weight = pca_weight / total_weight

        # Get PCA directions (always computed)
        pca_result = self.get_refusal_directions_pca(
            good_residuals, bad_residuals, n_components=1
        )
        pca_directions = pca_result.directions[:, 0, :]  # Shape: (n_layers, hidden_dim)

        # Get supervised directions (raises SupervisedProbeError on failure)
        supervised_result = self.get_refusal_directions_supervised(
            good_residuals, bad_residuals, bad_prompts, evaluator, min_probe_accuracy
        )
        supervised_directions = supervised_result.directions

        # Weighted combination
        ensemble_directions = (
            probe_weight * supervised_directions + pca_weight * pca_directions
        )

        # Re-normalize after combination
        ensemble_directions = F.normalize(ensemble_directions, p=2, dim=-1)

        print(
            f"  * Ensemble directions: probe_weight={probe_weight:.2f}, pca_weight={pca_weight:.2f}"
        )

        return ensemble_directions

    # Phase 5: Contrastive Activation Addition (CAA)
    def extract_compliance_direction(
        self,
        compliant_residuals: Tensor,
        refusing_residuals: Tensor,
    ) -> Tensor:
        """Extract a direction that encodes 'compliance' behavior.

        This is conceptually the OPPOSITE of the refusal direction, but computed
        from different data (actual compliance vs refusal responses). The compliance
        direction points from "refused" to "complied" in activation space.

        Args:
            compliant_residuals: Residuals from prompts where model complied
                Shape: (n_comply, n_layers, hidden_dim)
            refusing_residuals: Residuals from prompts where model refused
                Shape: (n_refuse, n_layers, hidden_dim)

        Returns:
            Compliance direction tensor of shape (n_layers, hidden_dim)
        """
        compliance_direction = F.normalize(
            compliant_residuals.mean(dim=0) - refusing_residuals.mean(dim=0),
            p=2,
            dim=-1,  # Normalize across hidden_dim
        )
        return compliance_direction

    def abliterate_with_caa(
        self,
        refusal_directions: Tensor,
        compliance_direction: Tensor,
        parameters: dict[str, "AbliterationParameters"],
        removal_strength: float = 1.0,
        addition_strength: float = 0.3,
        max_overlap: float = 0.5,
        layer_profiles: list["LayerRangeProfile"] | None = None,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> tuple[bool, dict[str, int]]:
        """Combined ablation: remove refusal direction + add compliance direction.

        This implements Contrastive Activation Addition (CAA) which:
        1. Removes the refusal direction (standard ablation)
        2. Adds a compliance direction to encourage helpful behavior

        WARNING: Addition may partially undo removal if directions aren't orthogonal.
        The method checks cosine similarity and skips addition if overlap is too high.

        Args:
            refusal_directions: Refusal directions to remove
                Shape: (n_layers, hidden_dim) or (n_layers, n_components, hidden_dim)
            compliance_direction: Compliance direction to add
                Shape: (n_layers, hidden_dim)
            parameters: Abliteration parameters per component
            removal_strength: Multiplier for refusal direction removal (default 1.0)
            addition_strength: Multiplier for compliance direction addition (default 0.3)
            max_overlap: Maximum allowed cosine similarity (default 0.5)
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA
            mpoa_min_scale: Minimum scaling factor for MPOA (default 0.5)
            mpoa_max_scale: Maximum scaling factor for MPOA (default 2.0)

        Returns:
            Tuple of (caa_applied, stats):
                - caa_applied: True if CAA addition was applied, False if skipped
                - stats: Dict with ablation statistics
        """
        stats = {
            "removal_layers_processed": 0,
            "removal_matrices_modified": 0,
            "addition_layers_processed": 0,
            "errors_suppressed": 0,
        }

        # Handle both single-direction and multi-direction refusal tensors
        if refusal_directions.dim() == 2:
            # Single direction: (n_layers, hidden_dim)
            refusal_for_sim = refusal_directions
        else:
            # Multi-direction: (n_layers, n_components, hidden_dim) - use first component
            refusal_for_sim = refusal_directions[:, 0, :]

        # Check orthogonality between refusal and compliance directions
        cosine_sim = (
            F.cosine_similarity(
                refusal_for_sim,
                compliance_direction,
                dim=-1,
            )
            .mean()
            .item()
        )

        print(f"  * CAA: Refusal-compliance cosine similarity: {cosine_sim:.3f}")
        logger.info(
            "Applying CAA (Contrastive Activation Addition)",
            cosine_similarity=cosine_sim,
            removal_strength=removal_strength,
            addition_strength=addition_strength,
            max_overlap=max_overlap,
            use_mpoa=use_mpoa,
        )

        caa_applied = True
        if abs(cosine_sim) > max_overlap:
            print(
                f"[yellow]Warning: High direction overlap (|cosine|={abs(cosine_sim):.2f} > {max_overlap}). "
                f"Skipping CAA addition to prevent undoing ablation.[/yellow]"
            )
            caa_applied = False

        # Step 1: Remove refusal direction (standard ablation)
        # Scale parameters by removal_strength
        if removal_strength != 1.0:
            scaled_parameters = {}
            for component, params in parameters.items():
                scaled_parameters[component] = AbliterationParameters(
                    max_weight=params.max_weight * removal_strength,
                    max_weight_position=params.max_weight_position,
                    min_weight=params.min_weight * removal_strength,
                    min_weight_distance=params.min_weight_distance,
                )
        else:
            scaled_parameters = parameters

        # Handle multi-direction vs single-direction
        try:
            if refusal_directions.dim() == 3:
                # Multi-direction: use first component (primary direction)
                removal_stats = self.abliterate(
                    refusal_directions[:, 0, :],
                    None,  # Per-layer mode
                    scaled_parameters,
                    layer_profiles=layer_profiles,
                    use_mpoa=use_mpoa,
                    mpoa_norm_mode=mpoa_norm_mode,
                    mpoa_min_scale=mpoa_min_scale,
                    mpoa_max_scale=mpoa_max_scale,
                )
            else:
                # Single direction
                removal_stats = self.abliterate(
                    refusal_directions,
                    None,  # Per-layer mode
                    scaled_parameters,
                    layer_profiles=layer_profiles,
                    use_mpoa=use_mpoa,
                    mpoa_norm_mode=mpoa_norm_mode,
                    mpoa_min_scale=mpoa_min_scale,
                    mpoa_max_scale=mpoa_max_scale,
                )
            stats["removal_layers_processed"] = removal_stats.get("layers_processed", 0)
            stats["removal_matrices_modified"] = removal_stats.get(
                "matrices_modified", 0
            )
            stats["errors_suppressed"] += removal_stats.get("errors_suppressed", 0)
        except WeightModificationError as e:
            record_suppressed_error(
                error=e,
                context="abliterate_with_caa_removal",
                module="model",
                severity="error",
                details={"removal_strength": removal_strength},
                include_traceback=True,
            )
            stats["errors_suppressed"] += 1
            logger.error(f"CAA refusal removal failed: {e}")
            # Continue to attempt compliance addition if overlap is acceptable

        # Step 2: Add compliance direction (only if overlap is acceptable)
        if caa_applied:
            print(
                f"  * CAA: Adding compliance direction (strength={addition_strength:.2f})"
            )

            num_layers = len(self.get_layers())

            for layer_idx in range(num_layers):
                try:
                    layer = self.get_layers()[layer_idx]

                    # Get layer-specific compliance direction
                    compliance_dir = compliance_direction[layer_idx]

                    # Check for NaN/Inf
                    if (
                        torch.isnan(compliance_dir).any()
                        or torch.isinf(compliance_dir).any()
                    ):
                        record_suppressed_error(
                            error=None,
                            context="abliterate_with_caa_nan_direction",
                            module="model",
                            severity="warning",
                            details={"layer_idx": layer_idx},
                        )
                        stats["errors_suppressed"] += 1
                        continue

                    # Compute projector for this direction
                    projector = torch.outer(
                        compliance_dir,
                        compliance_dir,
                    ).to(self.model.dtype)

                    # Apply layer multiplier if profiles are provided
                    layer_multiplier = self.get_layer_multiplier(
                        layer_idx, num_layers, layer_profiles
                    )
                    effective_strength = addition_strength * layer_multiplier

                    # Add to attention output projection
                    # This encourages the model to produce compliance-like activations
                    o_proj = layer.self_attn.o_proj.weight
                    device_projector = projector.to(o_proj.device)

                    # In-place addition: W += strength * (projector @ W)
                    # This biases the output toward the compliance direction
                    o_proj.data.add_(
                        effective_strength * (device_projector @ o_proj.data)
                    )
                    stats["addition_layers_processed"] += 1

                    # Verify no NaN/Inf introduced
                    if torch.isnan(o_proj.data).any() or torch.isinf(o_proj.data).any():
                        record_suppressed_error(
                            error=None,
                            context="abliterate_with_caa_nan_result",
                            module="model",
                            severity="error",
                            details={"layer_idx": layer_idx},
                        )
                        stats["errors_suppressed"] += 1

                except torch.cuda.OutOfMemoryError as e:
                    record_suppressed_error(
                        error=e,
                        context="abliterate_with_caa_addition_oom",
                        module="model",
                        severity="error",
                        details={"layer_idx": layer_idx},
                    )
                    stats["errors_suppressed"] += 1
                    empty_cache()
                except Exception as e:
                    record_suppressed_error(
                        error=e,
                        context="abliterate_with_caa_addition_error",
                        module="model",
                        severity="error",
                        details={"layer_idx": layer_idx},
                        include_traceback=True,
                    )
                    stats["errors_suppressed"] += 1

                # Clean up periodically
                if layer_idx % CACHE_CLEAR_INTERVAL == (CACHE_CLEAR_INTERVAL - 1):
                    empty_cache()

        # Log summary
        if stats["errors_suppressed"] > 0:
            logger.warning(
                "CAA ablation completed with errors",
                caa_applied=caa_applied,
                removal_matrices_modified=stats["removal_matrices_modified"],
                addition_layers_processed=stats["addition_layers_processed"],
                errors_suppressed=stats["errors_suppressed"],
            )
        else:
            logger.info(
                "CAA ablation completed",
                caa_applied=caa_applied,
                removal_matrices_modified=stats["removal_matrices_modified"],
                addition_layers_processed=stats["addition_layers_processed"],
            )

        return caa_applied, stats

    def get_compliance_directions_from_responses(
        self,
        bad_prompts: list[str],
        bad_residuals: Tensor,
        evaluator: "Evaluator",
    ) -> tuple[Tensor, Tensor] | None:
        """Generate responses and separate into compliant vs refusing for CAA.

        This method generates responses to bad prompts and uses the evaluator
        to classify them as compliant or refusing, then returns the corresponding
        residuals separated into two groups.

        Args:
            bad_prompts: List of harmful prompts
            bad_residuals: Residuals from harmful prompts
                Shape: (n_bad, n_layers, hidden_dim)
            evaluator: Evaluator instance for refusal detection

        Returns:
            Tuple of (compliant_residuals, refusing_residuals), or None if
            there aren't enough samples in each category (< 10)
        """
        # Validate alignment
        if len(bad_prompts) != bad_residuals.shape[0]:
            print(
                f"[yellow]Warning: Prompt count ({len(bad_prompts)}) != residual count "
                f"({bad_residuals.shape[0]}). Using first {min(len(bad_prompts), bad_residuals.shape[0])} samples.[/yellow]"
            )
            n_samples = min(len(bad_prompts), bad_residuals.shape[0])
            bad_prompts = bad_prompts[:n_samples]
            bad_residuals = bad_residuals[:n_samples]

        print("  * CAA: Generating responses to classify compliance behavior...")
        bad_responses = self.get_responses_batched(bad_prompts)

        # Classify responses
        refusal_mask = torch.tensor(
            [evaluator.is_refusal(r) for r in bad_responses],
            dtype=torch.bool,
            device=bad_residuals.device,
        )

        n_refusals = int(refusal_mask.sum().item())
        n_comply = len(refusal_mask) - n_refusals

        print(
            f"  * CAA: Response classification: {n_refusals} refusals, {n_comply} compliant"
        )

        if n_refusals < 10 or n_comply < 10:
            record_suppressed_error(
                error=None,
                context="get_compliance_directions_from_responses",
                module="model",
                severity="info",
                details={
                    "n_refusals": n_refusals,
                    "n_comply": n_comply,
                    "min_required": 10,
                    "reason": "Insufficient samples for CAA",
                },
            )
            print(
                f"[yellow]Warning: Insufficient samples for CAA "
                f"(refusals={n_refusals}, comply={n_comply}, need 10+ each). "
                f"Skipping CAA.[/yellow]"
            )
            return None

        compliant_residuals = bad_residuals[~refusal_mask]
        refusing_residuals = bad_residuals[refusal_mask]

        return compliant_residuals, refusing_residuals

    # Phase 6: Circuit-Level Ablation
    def is_gqa_model(self) -> tuple[bool, int, int]:
        """Check if the model uses Grouped Query Attention (GQA).

        GQA models have fewer key-value heads than query heads, which makes
        circuit-level ablation more complex and potentially incompatible.

        Returns:
            Tuple of (is_gqa, n_heads, n_kv_heads)
        """
        n_heads = getattr(self.model.config, "num_attention_heads", 32)
        n_kv_heads = getattr(self.model.config, "num_key_value_heads", n_heads)

        return n_kv_heads != n_heads, n_heads, n_kv_heads

    def discover_refusal_circuits(
        self,
        refusal_directions: Tensor,
        n_circuits: int = 20,
        importance_threshold: float = 0.1,
    ) -> list[RefusalCircuit]:
        """Discover attention heads that mediate refusal behavior.

        Uses a simplified heuristic based on the alignment between attention
        head output projections and the refusal direction. Heads whose output
        projections are highly aligned with the refusal direction are likely
        to be mediating refusal behavior.

        Note: This is a simplified approach. Full circuit discovery would require
        activation patching with TransformerLens or similar infrastructure.

        Args:
            refusal_directions: Refusal directions, shape (n_layers, hidden_dim)
            n_circuits: Maximum number of circuits to return
            importance_threshold: Minimum importance score to include

        Returns:
            List of RefusalCircuit objects, sorted by importance descending
        """
        is_gqa, n_heads, n_kv_heads = self.is_gqa_model()

        if is_gqa:
            print(
                f"[yellow]Warning: GQA model detected (n_heads={n_heads}, "
                f"n_kv_heads={n_kv_heads}). Circuit discovery may be inaccurate.[/yellow]"
            )

        circuits: list[RefusalCircuit] = []
        num_layers = len(self.get_layers())

        for layer_idx in range(num_layers):
            layer = self.get_layers()[layer_idx]
            o_proj = layer.self_attn.o_proj.weight

            hidden_dim = o_proj.shape[0]
            head_dim = hidden_dim // n_heads

            # Get refusal direction for this layer
            direction = refusal_directions[layer_idx].to(o_proj.device)

            for head_idx in range(n_heads):
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim

                if end_idx > hidden_dim:
                    continue

                # Get the output projection weights for this head
                head_weights = o_proj[start_idx:end_idx, :]

                # Compute alignment with refusal direction
                # Higher alignment means this head's output is more aligned with refusal
                head_direction = direction[start_idx:end_idx]
                head_direction_norm = F.normalize(head_direction.float(), p=2, dim=0)

                # Compute mean alignment of head's output weights with refusal direction
                weights_norm = F.normalize(head_weights.float(), p=2, dim=0)
                alignment = (
                    torch.abs(torch.matmul(head_direction_norm, weights_norm))
                    .mean()
                    .item()
                )

                if alignment >= importance_threshold:
                    circuits.append(
                        RefusalCircuit(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            importance=alignment,
                        )
                    )

        # Sort by importance and take top n_circuits
        circuits.sort(key=lambda c: c.importance, reverse=True)
        circuits = circuits[:n_circuits]

        print(
            f"  * Discovered {len(circuits)} refusal circuits "
            f"(threshold={importance_threshold}, max={n_circuits})"
        )

        if circuits:
            top_circuit = circuits[0]
            print(
                f"    * Top circuit: layer={top_circuit.layer_idx}, "
                f"head={top_circuit.head_idx}, importance={top_circuit.importance:.3f}"
            )

        return circuits

    def abliterate_circuits(
        self,
        circuits: list[RefusalCircuit],
        refusal_directions: Tensor,
        ablation_strength: float = 1.0,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> CircuitAblationResult:
        """Abliterate only specific attention heads, not entire layers.

        This is a more surgical approach to ablation that targets individual
        attention heads identified as mediating refusal behavior. This can
        potentially reduce capability damage compared to full-layer ablation.

        WARNING: This method is NOT supported for GQA (Grouped Query Attention)
        models like Qwen2.5 and Llama 3.x. It will detect GQA and skip ablation.

        Args:
            circuits: List of RefusalCircuit objects to ablate
            refusal_directions: Refusal directions, shape (n_layers, hidden_dim)
            ablation_strength: Base ablation strength multiplier (default 1.0)
            use_mpoa: Use MPOA (Norm-Preserving Biprojected Abliteration)
            mpoa_norm_mode: Norm preservation mode for MPOA ('row', 'column', 'frobenius')
            mpoa_min_scale: Minimum scaling factor for MPOA (default 0.5)
            mpoa_max_scale: Maximum scaling factor for MPOA (default 2.0)

        Returns:
            CircuitAblationResult with ablation statistics
        """
        # Check for GQA and skip if detected
        is_gqa, n_heads, n_kv_heads = self.is_gqa_model()

        if is_gqa:
            print(
                f"[yellow]Warning: GQA model detected "
                f"(n_heads={n_heads}, n_kv_heads={n_kv_heads}).[/yellow]"
            )
            print(
                "[yellow]Circuit ablation is not supported for GQA models. "
                "Skipping circuit-level ablation.[/yellow]"
            )
            return CircuitAblationResult(
                circuits_ablated=0,
                circuits_skipped=len(circuits),
                gqa_detected=True,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
            )

        if not circuits:
            print("[yellow]Warning: No circuits provided for ablation.[/yellow]")
            return CircuitAblationResult(
                circuits_ablated=0,
                circuits_skipped=0,
                gqa_detected=False,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
            )

        print(f"  * Abliterating {len(circuits)} circuits...")

        circuits_ablated = 0
        circuits_skipped = 0
        num_layers = len(self.get_layers())

        for circuit in circuits:
            # Validate layer index
            if circuit.layer_idx < 0 or circuit.layer_idx >= num_layers:
                print(
                    f"[red]Error: Layer {circuit.layer_idx} out of bounds "
                    f"(0-{num_layers - 1}). Skipping circuit.[/red]"
                )
                circuits_skipped += 1
                continue

            layer = self.get_layers()[circuit.layer_idx]
            o_proj = layer.self_attn.o_proj.weight

            hidden_dim = o_proj.shape[0]
            head_dim = hidden_dim // n_heads

            # Compute head-specific indices
            start_idx = circuit.head_idx * head_dim
            end_idx = start_idx + head_dim

            # Validate head indices
            if circuit.head_idx < 0 or circuit.head_idx >= n_heads:
                print(
                    f"[red]Error: Head {circuit.head_idx} out of bounds "
                    f"(0-{n_heads - 1}) for layer {circuit.layer_idx}. "
                    f"Skipping circuit.[/red]"
                )
                circuits_skipped += 1
                continue

            if end_idx > hidden_dim:
                print(
                    f"[red]Error: Head {circuit.head_idx} indices "
                    f"({start_idx}:{end_idx}) exceed hidden_dim ({hidden_dim}). "
                    f"Skipping circuit.[/red]"
                )
                circuits_skipped += 1
                continue

            # Get refusal direction for this layer
            direction = refusal_directions[circuit.layer_idx].to(o_proj.device)

            # Compute effective ablation strength
            strength = ablation_strength * circuit.importance

            # Project direction to head subspace
            head_direction = direction[start_idx:end_idx]
            head_direction = F.normalize(head_direction, p=2, dim=0)

            # Create projector for this head's subspace
            projector = torch.outer(head_direction, head_direction).to(o_proj.dtype)

            # Ablate only this head's contribution
            head_weights = o_proj.data[start_idx:end_idx, :]

            if use_mpoa:
                # MPOA: Norm-Preserving Biprojected Abliteration for circuit-level
                # Store original norms based on mode
                if mpoa_norm_mode == "row":
                    original_norms = torch.linalg.norm(
                        head_weights, dim=1, keepdim=True
                    )
                elif mpoa_norm_mode == "column":
                    original_norms = torch.linalg.norm(
                        head_weights, dim=0, keepdim=True
                    )
                else:  # frobenius
                    original_norm = torch.linalg.norm(head_weights)

                # Apply projection removal
                new_weights = head_weights - strength * (projector @ head_weights)

                # Rescale to preserve norms
                if mpoa_norm_mode == "row":
                    new_norms = torch.linalg.norm(new_weights, dim=1, keepdim=True)
                    scaling = original_norms / (new_norms + EPSILON)
                    scaling = torch.clamp(
                        scaling, min=mpoa_min_scale, max=mpoa_max_scale
                    )
                    new_weights = new_weights * scaling
                elif mpoa_norm_mode == "column":
                    new_norms = torch.linalg.norm(new_weights, dim=0, keepdim=True)
                    scaling = original_norms / (new_norms + EPSILON)
                    scaling = torch.clamp(
                        scaling, min=mpoa_min_scale, max=mpoa_max_scale
                    )
                    new_weights = new_weights * scaling
                else:  # frobenius
                    new_norm = torch.linalg.norm(new_weights)
                    if original_norm > EPSILON and new_norm > EPSILON:
                        scaling = original_norm / new_norm
                        scaling = torch.clamp(
                            scaling, min=mpoa_min_scale, max=mpoa_max_scale
                        )
                        new_weights = new_weights * scaling

                o_proj.data[start_idx:end_idx, :] = new_weights
            else:
                # Standard abliteration
                o_proj.data[start_idx:end_idx, :] = head_weights - strength * (
                    projector @ head_weights
                )

            circuits_ablated += 1

        # Clean up
        empty_cache()

        print(
            f"  * Circuit ablation complete: {circuits_ablated} ablated, "
            f"{circuits_skipped} skipped"
        )

        return CircuitAblationResult(
            circuits_ablated=circuits_ablated,
            circuits_skipped=circuits_skipped,
            gqa_detected=False,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )

    def save_circuits_to_cache(
        self,
        circuits: list[RefusalCircuit],
        cache_path: str,
    ) -> None:
        """Save discovered circuits to a JSON cache file with error handling.

        Args:
            circuits: List of RefusalCircuit objects to save
            cache_path: Path to the JSON cache file

        Note:
            Uses atomic write (temp file + rename) to prevent corruption.
            Non-critical failures (disk full, permission denied) are logged
            but don't raise exceptions - circuit discovery continues.
        """
        import json
        from pathlib import Path

        cache_data = {
            "model": self.settings.model,
            "circuits": [
                {
                    "layer_idx": c.layer_idx,
                    "head_idx": c.head_idx,
                    "importance": c.importance,
                }
                for c in circuits
            ],
        }

        try:
            # Atomic write using temp file to prevent corruption
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            # Atomic rename (POSIX) or best-effort (Windows)
            Path(temp_path).replace(cache_path)

            print(f"  * Saved {len(circuits)} circuits to {cache_path}")

        except OSError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg:
                logger.warning(
                    f"Insufficient disk space for circuit cache: {e}. "
                    f"Circuits will be recomputed on next run."
                )
                print(
                    "[yellow]  * Warning: Insufficient disk space, circuit cache not saved[/]"
                )
            elif "permission" in error_msg:
                logger.warning(
                    f"Permission denied writing circuit cache to {cache_path}: {e}. "
                    f"Check directory permissions."
                )
                print(
                    f"[yellow]  * Warning: Permission denied for {cache_path}, cache not saved[/]"
                )
            else:
                # Other OSErrors - log but don't crash (cache is optional)
                record_suppressed_error(
                    error=e,
                    context="circuit_cache_save",
                    module="model",
                    severity="warning",
                    details={"cache_path": cache_path, "n_circuits": len(circuits)},
                )
                logger.warning(f"Failed to save circuit cache: {e}")
                print(f"[yellow]  * Warning: Failed to save circuit cache: {e}[/]")

        except Exception as e:
            # Unexpected errors - log with traceback but don't crash
            record_suppressed_error(
                error=e,
                context="circuit_cache_save",
                module="model",
                severity="error",
                details={"cache_path": cache_path, "n_circuits": len(circuits)},
                include_traceback=True,
            )
            logger.error(f"Unexpected error saving circuit cache: {e}")
            print(f"[yellow]  * Warning: Unexpected error saving cache: {e}[/]")

    def load_circuits_from_cache(
        self,
        cache_path: str,
    ) -> list[RefusalCircuit] | None:
        """Load circuits from a JSON cache file.

        Args:
            cache_path: Path to the JSON cache file

        Returns:
            List of RefusalCircuit objects, or None if cache doesn't exist
            or is for a different model
        """
        import json
        import os

        if not os.path.exists(cache_path):
            record_suppressed_error(
                error=None,
                context="load_circuits_from_cache",
                module="model",
                severity="debug",
                details={
                    "cache_path": cache_path,
                    "reason": "Cache file not found",
                },
            )
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Verify model matches
            if cache_data.get("model") != self.settings.model:
                record_suppressed_error(
                    error=None,
                    context="load_circuits_from_cache",
                    module="model",
                    severity="info",
                    details={
                        "cache_model": cache_data.get("model"),
                        "current_model": self.settings.model,
                        "cache_path": cache_path,
                        "reason": "Model mismatch",
                    },
                )
                print(
                    f"[yellow]Warning: Circuit cache is for model "
                    f"'{cache_data.get('model')}', not '{self.settings.model}'. "
                    f"Ignoring cache.[/yellow]"
                )
                return None

            circuits = [
                RefusalCircuit(
                    layer_idx=c["layer_idx"],
                    head_idx=c["head_idx"],
                    importance=c["importance"],
                )
                for c in cache_data.get("circuits", [])
            ]

            print(f"  * Loaded {len(circuits)} circuits from cache: {cache_path}")
            return circuits

        except (json.JSONDecodeError, KeyError) as e:
            record_suppressed_error(
                error=e,
                context="load_circuits_from_cache",
                module="model",
                severity="warning",
                details={
                    "cache_path": cache_path,
                    "reason": "Failed to parse cache file",
                },
            )
            print(
                f"[yellow]Warning: Failed to load circuit cache: {e}. "
                f"Will rediscover circuits.[/yellow]"
            )
            return None

    def get_or_discover_circuits(
        self,
        refusal_directions: Tensor,
        n_circuits: int = 20,
        importance_threshold: float = 0.1,
        cache_path: str | None = None,
    ) -> list[RefusalCircuit]:
        """Get circuits from cache or discover them.

        Convenience method that checks the cache first, and if not found,
        discovers circuits and optionally saves them to cache.

        Args:
            refusal_directions: Refusal directions for circuit discovery
            n_circuits: Maximum number of circuits to discover
            importance_threshold: Minimum importance score
            cache_path: Optional path to cache file

        Returns:
            List of RefusalCircuit objects
        """
        # Try loading from cache
        if cache_path is not None:
            circuits = self.load_circuits_from_cache(cache_path)
            if circuits is not None:
                return circuits

        # Discover circuits
        circuits = self.discover_refusal_circuits(
            refusal_directions,
            n_circuits=n_circuits,
            importance_threshold=importance_threshold,
        )

        # Save to cache if path provided
        if cache_path is not None and circuits:
            self.save_circuits_to_cache(circuits, cache_path)

        return circuits

    # ==================== MoE Router-Aware Expert Targeting ====================

    def should_use_moe_targeting(self) -> bool:
        """Check if MoE-specific targeting should be used.

        Returns True if:
        - Model is an MoE model
        - Router-aware targeting is enabled in settings
        """
        return self.settings.use_router_aware_targeting and self.is_moe_model()

    def is_moe_model(self) -> bool:
        """Check if the model uses Mixture of Experts architecture.

        Detects MoE by checking for expert modules in the first few layers.

        Returns:
            True if model has MoE layers, False otherwise
        """
        # Check first 3 layers (layer 0 might be dense in hybrid models like Moonlight)
        for layer_idx in range(min(3, len(self.get_layers()))):
            layer = self.get_layers()[layer_idx]
            # Check for various MoE patterns
            if hasattr(layer.mlp, "experts") or hasattr(layer.mlp, "shared_experts"):
                return True
            if hasattr(layer, "block_sparse_moe"):
                return True
            if hasattr(layer, "moe"):
                return True
        return False

    def get_moe_config(self, layer_idx: int = 1) -> tuple[int, int] | None:
        """Get MoE configuration (n_experts, top_k) from a layer.

        Args:
            layer_idx: Which layer to check (default 1 since layer 0 may be dense)

        Returns:
            Tuple of (n_routed_experts, top_k) or None if not MoE
        """
        if layer_idx >= len(self.get_layers()):
            return None

        layer = self.get_layers()[layer_idx]

        # DeepSeekV3/Moonlight pattern
        if hasattr(layer.mlp, "gate"):
            gate = layer.mlp.gate
            n_experts = getattr(gate, "n_routed_experts", None)
            top_k = getattr(gate, "top_k", None)
            if n_experts is not None and top_k is not None:
                return (n_experts, top_k)

        # Mixtral/general pattern
        if hasattr(layer.mlp, "experts"):
            n_experts = len(layer.mlp.experts)
            # Default top_k is usually 2 for Mixtral-style
            top_k = getattr(layer.mlp, "num_experts_per_tok", 2)
            return (n_experts, top_k)

        return None

    def track_expert_activations(
        self,
        prompts: list[str],
    ) -> MoEExpertActivations | None:
        """Track which experts are activated for a set of prompts.

        Uses forward hooks on MoE gate modules to capture expert selection
        during inference. This is used for router-aware expert targeting.

        Args:
            prompts: List of prompts to track activations for

        Returns:
            MoEExpertActivations containing per-layer expert activation counts,
            or None if model is not MoE or tracking fails

        Note:
            Returns None (not raises) for non-fatal failures to allow
            fallback to standard abliteration.
        """
        if not prompts:
            record_suppressed_error(
                error=None,
                context="track_expert_activations",
                module="model",
                severity="warning",
                details={"reason": "Empty prompt list provided"},
            )
            return None

        if not self.is_moe_model():
            logger.info("Model is not MoE, skipping expert activation tracking")
            return None

        moe_config = self.get_moe_config()
        if moe_config is None:
            record_suppressed_error(
                error=None,
                context="track_expert_activations",
                module="model",
                severity="warning",
                details={"reason": "Could not determine MoE config"},
            )
            logger.warning("Could not determine MoE config")
            return None

        n_experts, top_k = moe_config
        num_layers = len(self.get_layers())

        # Initialize counts
        layer_expert_counts: dict[int, dict[int, int]] = {
            i: {j: 0 for j in range(n_experts)} for i in range(num_layers)
        }

        # Storage for captured activations
        captured_indices: list[tuple[int, Tensor]] = []
        hook_errors: list[dict] = []

        def make_hook(layer_idx: int):
            """Create a forward hook for capturing expert indices.

            NOTE: Hook output format assumption:
            This assumes the MoE gate returns a tuple where output[0] contains
            the selected expert indices tensor of shape (batch, seq_len, top_k).

            Known compatible architectures:
            - DeepSeekV3 / Moonlight: Returns (topk_idx, topk_weight, aux_loss)
            - Mixtral-style: May use different format - not yet tested
            - Phi-3.5-MoE: May use different format - not yet tested

            If your MoE model uses a different gate output format, this hook
            may silently fail to capture activations.
            """

            def hook(module, input, output):
                try:
                    # DeepSeekV3 MoEGate returns (topk_idx, topk_weight, aux_loss)
                    # or similar tuple with expert indices
                    if isinstance(output, tuple) and len(output) >= 2:
                        topk_idx = output[0]  # Shape: (batch, seq_len, top_k)
                        if topk_idx is not None and torch.is_tensor(topk_idx):
                            captured_indices.append((layer_idx, topk_idx.detach()))
                    else:
                        # Track unexpected output format
                        hook_errors.append(
                            {
                                "layer_idx": layer_idx,
                                "output_type": type(output).__name__,
                                "output_len": len(output)
                                if isinstance(output, tuple)
                                else "N/A",
                            }
                        )
                except Exception as e:
                    hook_errors.append(
                        {
                            "layer_idx": layer_idx,
                            "error": str(e),
                        }
                    )

            return hook

        # Register hooks on all MoE gates
        hooks = []
        hooks_registered = 0
        for layer_idx in range(num_layers):
            layer = self.get_layers()[layer_idx]
            if hasattr(layer.mlp, "gate"):
                try:
                    hook = layer.mlp.gate.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)
                    hooks_registered += 1
                except Exception as e:
                    record_suppressed_error(
                        error=e,
                        context="track_expert_activations_hook_register",
                        module="model",
                        severity="warning",
                        details={"layer_idx": layer_idx},
                    )

        if not hooks:
            record_suppressed_error(
                error=None,
                context="track_expert_activations",
                module="model",
                severity="warning",
                details={
                    "reason": "No MoE gates found to hook",
                    "num_layers": num_layers,
                },
            )
            logger.warning("No MoE gates found to hook")
            return None

        batches_processed = 0
        batches_failed = 0

        try:
            # Run inference to capture activations
            print(f"  * Tracking expert activations for {len(prompts)} prompts...")
            for batch in batchify(prompts, self.settings.batch_size):
                captured_indices.clear()
                try:
                    # Generate just 1 token to capture gate activations
                    with torch.no_grad():
                        self.generate(batch, max_new_tokens=1)
                    batches_processed += 1
                except torch.cuda.OutOfMemoryError as e:
                    batches_failed += 1
                    record_suppressed_error(
                        error=e,
                        context="track_expert_activations_oom",
                        module="model",
                        severity="warning",
                        details={
                            "batch_size": len(batch),
                            "batches_failed": batches_failed,
                        },
                    )
                    empty_cache()
                    if batches_failed > 3:
                        logger.warning(
                            f"Too many OOM errors ({batches_failed}) during expert tracking, aborting"
                        )
                        break
                    continue
                except Exception as e:
                    batches_failed += 1
                    record_suppressed_error(
                        error=e,
                        context="track_expert_activations_batch",
                        module="model",
                        severity="warning",
                        details={"batch_size": len(batch)},
                        include_traceback=True,
                    )
                    continue

                # Process captured indices
                for layer_idx, indices in captured_indices:
                    # indices shape: (batch, seq_len, top_k)
                    # Flatten and count
                    try:
                        flat_indices = indices.view(-1).cpu().tolist()
                        for expert_idx in flat_indices:
                            if 0 <= expert_idx < n_experts:
                                layer_expert_counts[layer_idx][expert_idx] += 1
                    except Exception as e:
                        record_suppressed_error(
                            error=e,
                            context="track_expert_activations_process",
                            module="model",
                            severity="warning",
                            details={"layer_idx": layer_idx},
                        )

        finally:
            # Remove hooks
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass  # Ignore errors during cleanup

        # Log hook errors if any
        if hook_errors:
            record_suppressed_error(
                error=None,
                context="track_expert_activations_hook_errors",
                module="model",
                severity="warning",
                details={
                    "n_hook_errors": len(hook_errors),
                    "sample_errors": hook_errors[:3],  # First 3 errors
                },
            )

        # Check for silent failure - no activations captured despite having hooks
        total_captured = sum(
            sum(counts.values()) for counts in layer_expert_counts.values()
        )
        if total_captured == 0:
            record_suppressed_error(
                error=None,
                context="track_expert_activations",
                module="model",
                severity="warning",
                details={
                    "reason": "No expert activations captured",
                    "hooks_registered": hooks_registered,
                    "batches_processed": batches_processed,
                    "batches_failed": batches_failed,
                    "hook_errors": len(hook_errors),
                    "advice": "MoE gate output format may not match expected pattern",
                },
            )
            logger.warning(
                "No expert activations captured during tracking. "
                "MoE gate output format may not match expected pattern (topk_idx, topk_weight, ...). "
                "This model's MoE architecture may not be compatible with router-aware targeting."
            )
            print(
                "[yellow]  * Warning: No expert activations captured - "
                "MoE gate format may be incompatible[/yellow]"
            )

        # Log summary
        logger.info(
            "Expert activation tracking complete",
            total_captured=total_captured,
            batches_processed=batches_processed,
            batches_failed=batches_failed,
            hooks_registered=hooks_registered,
        )

        return MoEExpertActivations(
            layer_expert_counts=layer_expert_counts,
            total_prompts=len(prompts),
            n_experts_per_layer=n_experts,
            top_k=top_k,
        )

    def get_moe_targeted_experts(
        self,
        activations: MoEExpertActivations,
        threshold: float = 0.1,
        top_k: int = 0,
    ) -> dict[int, list[int]]:
        """Get targeted experts per layer based on activation patterns.

        Args:
            activations: Expert activation tracking results
            threshold: Minimum activation frequency to target
            top_k: Maximum experts per layer (0 = no limit)

        Returns:
            Dict mapping layer_idx -> list of expert indices to target
        """
        targeted = {}
        for layer_idx in activations.layer_expert_counts.keys():
            experts = activations.get_high_activation_experts(
                layer_idx, threshold, top_k
            )
            if experts:
                targeted[layer_idx] = experts
        return targeted

    def abliterate_moe_targeted(
        self,
        refusal_directions: Tensor,
        parameters: dict[str, AbliterationParameters],
        targeted_experts: dict[int, list[int]],
        layer_profiles: list["LayerRangeProfile"] | None = None,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
    ) -> dict[str, int]:
        """Abliterate only targeted experts in MoE layers.

        This performs router-aware abliteration: instead of abliterating all
        experts (most of which never activate for refusal), only abliterate
        experts that were identified as processing refusal prompts.

        Also always abliterates:
        - Attention o_proj (all tokens pass through this)
        - Shared experts (always active in DeepSeekV3/Moonlight)

        Args:
            refusal_directions: Refusal direction tensor
            parameters: Abliteration parameters per component
            targeted_experts: Dict mapping layer_idx -> expert indices to target
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA norm preservation
            mpoa_norm_mode: MPOA norm mode
            mpoa_min_scale: MPOA min scale
            mpoa_max_scale: MPOA max scale

        Returns:
            Dict with ablation statistics:
                - 'attn_ablated': Number of attention layers ablated
                - 'experts_ablated': Number of expert MLPs ablated
                - 'shared_experts_ablated': Number of shared experts ablated
                - 'experts_skipped': Number of experts skipped (not targeted)
                - 'errors_suppressed': Number of non-fatal errors encountered

        Raises:
            MoEAbliterationError: If critical error occurs during MoE abliteration
        """
        # Validate inputs
        if refusal_directions is None or refusal_directions.numel() == 0:
            raise MoEAbliterationError(
                "Refusal directions tensor is empty or None for MoE abliteration."
            )

        if not parameters:
            raise MoEAbliterationError(
                "No abliteration parameters provided for MoE abliteration."
            )

        num_layers = len(self.get_layers())
        stats = {
            "attn_ablated": 0,
            "experts_ablated": 0,
            "shared_experts_ablated": 0,
            "experts_skipped": 0,
            "errors_suppressed": 0,
        }

        # Pre-compute projectors per device
        device_projectors_cache: dict[torch.device, Tensor] = {}

        def get_device_projector(projector: Tensor, device: torch.device) -> Tensor:
            if device not in device_projectors_cache:
                try:
                    device_projectors_cache[device] = projector.to(device)
                except torch.cuda.OutOfMemoryError:
                    empty_cache()
                    record_suppressed_error(
                        error=None,
                        context="abliterate_moe_projector_oom",
                        module="model",
                        severity="warning",
                        details={"device": str(device)},
                    )
                    stats["errors_suppressed"] += 1
                    device_projectors_cache[device] = projector.to("cpu")
            return device_projectors_cache[device]

        print(
            f"  * MoE targeted abliteration: {len(targeted_experts)} layers with targets"
        )
        if targeted_experts:
            sample_layer = next(iter(targeted_experts.keys()))
            print(
                f"    * Example layer {sample_layer}: targeting {len(targeted_experts[sample_layer])} experts"
            )

        logger.info(
            "Starting MoE targeted abliteration",
            num_layers=num_layers,
            layers_with_targets=len(targeted_experts),
            use_mpoa=use_mpoa,
        )

        for layer_idx in range(num_layers):
            layer = self.get_layers()[layer_idx]

            # Get layer-specific refusal direction
            layer_direction = refusal_directions[
                layer_idx + 1
            ]  # +1 for embedding offset

            # Validate direction
            if torch.isnan(layer_direction).any() or torch.isinf(layer_direction).any():
                record_suppressed_error(
                    error=None,
                    context="abliterate_moe_nan_direction",
                    module="model",
                    severity="warning",
                    details={"layer_idx": layer_idx},
                )
                stats["errors_suppressed"] += 1
                continue

            projector = torch.outer(layer_direction, layer_direction).to(
                self.model.dtype
            )
            device_projectors_cache.clear()

            # Get layer multiplier
            layer_multiplier = self.get_layer_multiplier(
                layer_idx, num_layers, layer_profiles
            )

            # Get parameters for attention component
            params = parameters.get("attn.o_proj")
            if params:
                distance = abs(layer_idx - params.max_weight_position)
                if distance <= params.min_weight_distance:
                    weight = params.max_weight + (
                        distance / params.min_weight_distance
                    ) * (params.min_weight - params.max_weight)
                    weight = weight * layer_multiplier

                    # Always abliterate attention (all tokens pass through)
                    try:
                        o_proj = layer.self_attn.o_proj.weight
                        device_projector = get_device_projector(
                            projector, o_proj.device
                        )

                        if use_mpoa:
                            self._apply_mpoa_projection(
                                o_proj,
                                device_projector,
                                weight,
                                mpoa_norm_mode,
                                mpoa_min_scale,
                                mpoa_max_scale,
                            )
                        else:
                            o_proj.sub_(weight * (device_projector @ o_proj))
                        stats["attn_ablated"] += 1
                    except torch.cuda.OutOfMemoryError as e:
                        record_suppressed_error(
                            error=e,
                            context="abliterate_moe_attn_oom",
                            module="model",
                            severity="error",
                            details={"layer_idx": layer_idx},
                        )
                        empty_cache()
                        stats["errors_suppressed"] += 1
                    except Exception as e:
                        record_suppressed_error(
                            error=e,
                            context="abliterate_moe_attn_error",
                            module="model",
                            severity="error",
                            details={"layer_idx": layer_idx},
                            include_traceback=True,
                        )
                        stats["errors_suppressed"] += 1

            # MLP abliteration - targeted
            params = parameters.get("mlp.down_proj")
            if params:
                distance = abs(layer_idx - params.max_weight_position)
                if distance <= params.min_weight_distance:
                    weight = params.max_weight + (
                        distance / params.min_weight_distance
                    ) * (params.min_weight - params.max_weight)
                    weight = weight * layer_multiplier

                    # Get targeted experts for this layer
                    layer_targets = targeted_experts.get(layer_idx, [])

                    # Abliterate targeted routed experts
                    if hasattr(layer.mlp, "experts"):
                        for expert_idx, expert in enumerate(layer.mlp.experts):
                            if expert_idx in layer_targets:
                                if hasattr(expert, "down_proj"):
                                    try:
                                        down_proj = expert.down_proj.weight
                                        device_projector = get_device_projector(
                                            projector, down_proj.device
                                        )
                                        if use_mpoa:
                                            self._apply_mpoa_projection(
                                                down_proj,
                                                device_projector,
                                                weight,
                                                mpoa_norm_mode,
                                                mpoa_min_scale,
                                                mpoa_max_scale,
                                            )
                                        else:
                                            down_proj.sub_(
                                                weight * (device_projector @ down_proj)
                                            )
                                        stats["experts_ablated"] += 1
                                    except torch.cuda.OutOfMemoryError as e:
                                        record_suppressed_error(
                                            error=e,
                                            context="abliterate_moe_expert_oom",
                                            module="model",
                                            severity="error",
                                            details={
                                                "layer_idx": layer_idx,
                                                "expert_idx": expert_idx,
                                            },
                                        )
                                        empty_cache()
                                        stats["errors_suppressed"] += 1
                                    except Exception as e:
                                        record_suppressed_error(
                                            error=e,
                                            context="abliterate_moe_expert_error",
                                            module="model",
                                            severity="error",
                                            details={
                                                "layer_idx": layer_idx,
                                                "expert_idx": expert_idx,
                                            },
                                            include_traceback=True,
                                        )
                                        stats["errors_suppressed"] += 1
                                else:
                                    record_suppressed_error(
                                        error=None,
                                        context="abliterate_moe_expert_missing_down_proj",
                                        module="model",
                                        severity="info",
                                        details={
                                            "layer_idx": layer_idx,
                                            "expert_idx": expert_idx,
                                            "reason": "Expert has no down_proj attribute",
                                        },
                                    )
                            else:
                                stats["experts_skipped"] += 1

                    # Always abliterate shared experts (always active)
                    if self.settings.use_moe_shared_experts and hasattr(
                        layer.mlp, "shared_experts"
                    ):
                        shared = layer.mlp.shared_experts
                        if hasattr(shared, "down_proj"):
                            try:
                                down_proj = shared.down_proj.weight
                                device_projector = get_device_projector(
                                    projector, down_proj.device
                                )
                                if use_mpoa:
                                    self._apply_mpoa_projection(
                                        down_proj,
                                        device_projector,
                                        weight,
                                        mpoa_norm_mode,
                                        mpoa_min_scale,
                                        mpoa_max_scale,
                                    )
                                else:
                                    down_proj.sub_(
                                        weight * (device_projector @ down_proj)
                                    )
                                stats["shared_experts_ablated"] += 1
                            except torch.cuda.OutOfMemoryError as e:
                                record_suppressed_error(
                                    error=e,
                                    context="abliterate_moe_shared_oom",
                                    module="model",
                                    severity="error",
                                    details={"layer_idx": layer_idx},
                                )
                                empty_cache()
                                stats["errors_suppressed"] += 1
                            except Exception as e:
                                record_suppressed_error(
                                    error=e,
                                    context="abliterate_moe_shared_error",
                                    module="model",
                                    severity="error",
                                    details={"layer_idx": layer_idx},
                                    include_traceback=True,
                                )
                                stats["errors_suppressed"] += 1

                    # Dense layers (like layer 0 in Moonlight)
                    if hasattr(layer.mlp, "down_proj") and not hasattr(
                        layer.mlp, "experts"
                    ):
                        try:
                            down_proj = layer.mlp.down_proj.weight
                            device_projector = get_device_projector(
                                projector, down_proj.device
                            )
                            if use_mpoa:
                                self._apply_mpoa_projection(
                                    down_proj,
                                    device_projector,
                                    weight,
                                    mpoa_norm_mode,
                                    mpoa_min_scale,
                                    mpoa_max_scale,
                                )
                            else:
                                down_proj.sub_(weight * (device_projector @ down_proj))
                        except Exception as e:
                            record_suppressed_error(
                                error=e,
                                context="abliterate_moe_dense_error",
                                module="model",
                                severity="error",
                                details={"layer_idx": layer_idx},
                                include_traceback=True,
                            )
                            stats["errors_suppressed"] += 1

            # Periodic cache clear
            if layer_idx % CACHE_CLEAR_INTERVAL == (CACHE_CLEAR_INTERVAL - 1):
                empty_cache()

        device_projectors_cache.clear()

        # Log summary with error tracking
        if stats["errors_suppressed"] > 0:
            logger.warning(
                "MoE abliteration completed with errors",
                attn_ablated=stats["attn_ablated"],
                experts_ablated=stats["experts_ablated"],
                shared_experts_ablated=stats["shared_experts_ablated"],
                experts_skipped=stats["experts_skipped"],
                errors_suppressed=stats["errors_suppressed"],
            )
            print(
                f"[yellow]  * MoE abliteration completed with {stats['errors_suppressed']} errors suppressed[/yellow]"
            )
        else:
            logger.info(
                "MoE abliteration completed successfully",
                attn_ablated=stats["attn_ablated"],
                experts_ablated=stats["experts_ablated"],
                shared_experts_ablated=stats["shared_experts_ablated"],
                experts_skipped=stats["experts_skipped"],
            )

        print(
            f"  * MoE abliteration complete: "
            f"{stats['attn_ablated']} attn, "
            f"{stats['experts_ablated']} experts, "
            f"{stats['shared_experts_ablated']} shared, "
            f"{stats['experts_skipped']} skipped"
        )

        # Raise error if too many failures
        total_attempted = (
            stats["attn_ablated"]
            + stats["experts_ablated"]
            + stats["shared_experts_ablated"]
            + stats["errors_suppressed"]
        )
        if total_attempted > 0 and stats["errors_suppressed"] > total_attempted * 0.5:
            raise MoEAbliterationError(
                f"MoE abliteration had too many failures: {stats['errors_suppressed']} errors out of "
                f"{total_attempted} attempted operations. Check error tracker for details."
            )

        return stats

    # ==================== MoE Gate Abliteration ====================

    def compute_routing_entropy(self, prompts: list[str]) -> dict[int, float]:
        """Compute entropy of expert routing distribution per layer.

        Higher entropy = more uniform routing (healthy)
        Lower entropy = routing collapse (unhealthy)

        Args:
            prompts: List of prompts to analyze routing for

        Returns:
            Dict mapping layer_idx -> entropy (0.0 to log(n_experts))
        """
        activations = self.track_expert_activations(prompts)
        if activations is None:
            return {}

        entropies = {}
        for layer_idx, counts in activations.layer_expert_counts.items():
            total = sum(counts.values())
            if total == 0:
                entropies[layer_idx] = 0.0
                continue

            probs = [count / total for count in counts.values() if count > 0]
            entropy = -sum(p * math.log(p) for p in probs)
            entropies[layer_idx] = entropy

        return entropies

    def check_routing_health(
        self, prompts: list[str], min_entropy: float = 0.5
    ) -> bool:
        """Check if routing is healthy (not collapsed).

        Returns True if mean entropy > min_entropy.

        Args:
            prompts: List of prompts to check routing health for
            min_entropy: Minimum normalized entropy threshold (0.0 to 1.0)

        Returns:
            True if routing is healthy, False if collapsed
        """
        entropies = self.compute_routing_entropy(prompts)
        if not entropies:
            return True  # Non-MoE model or no entropy data

        # Guard against empty dict after filtering
        if len(entropies) == 0:
            logger.warning("No entropy values computed, assuming healthy routing")
            return True

        mean_entropy = sum(entropies.values()) / len(entropies)

        moe_config = self.get_moe_config()
        if moe_config is None or moe_config[0] <= 1:
            logger.warning("Invalid MoE config for entropy normalization")
            return True

        max_possible = math.log(moe_config[0])
        if max_possible <= 0:
            logger.warning(
                "Max possible entropy is non-positive, assuming healthy routing"
            )
            return True

        normalized_entropy = mean_entropy / max_possible

        logger.debug(
            "Routing health check",
            mean_entropy=mean_entropy,
            max_possible=max_possible,
            normalized_entropy=normalized_entropy,
            threshold=min_entropy,
        )

        return normalized_entropy > min_entropy

    def abliterate_gate(
        self,
        layer_idx: int,
        refusal_direction: Tensor,
        strength: float = 1.0,
    ) -> bool:
        """Abliterate the gate's ability to detect refusal-related content.

        The gate projects inputs to expert affinity scores. By removing the
        refusal direction from the gate weights, we prevent the gate from
        "seeing" refusal-triggering patterns in the input.

        Args:
            layer_idx: Which layer's gate to modify
            refusal_direction: Direction to remove (hidden_dim,)
            strength: Ablation strength (0.0 to 1.0)

        Returns:
            True if gate was abliterated, False if layer has no gate
        """
        layer = self.get_layers()[layer_idx]

        if not hasattr(layer.mlp, "gate"):
            logger.debug(f"Layer {layer_idx} has no gate attribute, skipping")
            return False

        gate = layer.mlp.gate
        if not hasattr(gate, "weight"):
            logger.debug(f"Layer {layer_idx} gate has no weight attribute, skipping")
            return False

        # Gate weight: [n_experts, hidden_dim]
        gate_weight = gate.weight.data

        # Normalize direction
        direction = refusal_direction.to(gate_weight.device)
        direction = direction / (direction.norm() + EPSILON)

        # Create projector
        projector = torch.outer(direction, direction).to(gate_weight.dtype)

        # Remove refusal direction from gate's input detection
        # W' = W - strength * (W @ projector)
        gate.weight.data = gate_weight - strength * (gate_weight @ projector)

        logger.debug(f"Abliterated gate at layer {layer_idx} with strength {strength}")
        return True

    def abliterate_gates(
        self,
        refusal_directions: Tensor,
        strength: float = 1.0,
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ) -> int:
        """Abliterate gates across all MoE layers.

        Args:
            refusal_directions: Per-layer directions (n_layers+1, hidden_dim)
            strength: Base ablation strength
            layer_profiles: Optional per-layer weight profiles

        Returns:
            Number of gates abliterated
        """
        num_layers = len(self.get_layers())
        gates_ablated = 0

        logger.info(
            "Abliterating MoE gates",
            num_layers=num_layers,
            strength=strength,
        )

        for layer_idx in range(num_layers):
            # Get layer-specific direction (+1 for embedding offset)
            direction = refusal_directions[layer_idx + 1]

            # Skip if direction contains NaN/Inf
            if torch.isnan(direction).any() or torch.isinf(direction).any():
                logger.warning(
                    f"Skipping gate at layer {layer_idx}: direction contains NaN/Inf"
                )
                continue

            # Apply layer profile multiplier
            layer_multiplier = self.get_layer_multiplier(
                layer_idx, num_layers, layer_profiles
            )
            effective_strength = strength * layer_multiplier

            if self.abliterate_gate(layer_idx, direction, effective_strength):
                gates_ablated += 1

        logger.info(f"Abliterated {gates_ablated} MoE gates")
        return gates_ablated

    def _track_expert_activations_per_prompt(
        self,
        prompts: list[str],
    ) -> dict[int, dict[int, set[int]]] | None:
        """Track which experts activate for each individual prompt.

        NOTE: The hook implementation handles several common MoE output formats,
        but may need architecture-specific customization. The exact output format
        of DeepSeek/Moonlight gates should be verified against the actual model
        before production use.

        Args:
            prompts: List of prompts to track activations for

        Returns:
            layer_idx -> prompt_idx -> set of expert indices that activated
            Returns None if model is not MoE or tracking fails.
        """
        if not self.is_moe_model():
            logger.debug("Not an MoE model, skipping per-prompt activation tracking")
            return None

        per_prompt_activations: dict[int, dict[int, set[int]]] = {}

        # Register hooks to capture expert selections per prompt
        hooks = []
        captured_experts: dict[int, set[int]] = {}  # layer_idx -> expert indices
        missing_router_indices_warned = (
            False  # Track if we've warned about unrecognized output format
        )

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                nonlocal missing_router_indices_warned
                # Capture which experts were selected
                # Implementation depends on specific MoE architecture
                try:
                    if hasattr(output, "router_indices"):
                        indices = output.router_indices.flatten().tolist()
                        captured_experts.setdefault(layer_idx, set()).update(indices)
                    elif hasattr(output, "topk_indices"):
                        # Alternative attribute name used by some MoE implementations
                        indices = output.topk_indices.flatten().tolist()
                        captured_experts.setdefault(layer_idx, set()).update(indices)
                    elif isinstance(output, tuple) and len(output) >= 2:
                        # Some gates return (scores, indices) tuple
                        potential_indices = output[
                            0
                        ]  # DeepSeek: (topk_idx, topk_weight, ...)
                        if hasattr(potential_indices, "flatten") and torch.is_tensor(
                            potential_indices
                        ):
                            indices = potential_indices.flatten().tolist()
                            captured_experts.setdefault(layer_idx, set()).update(
                                indices
                            )
                    else:
                        # Gate output format not recognized
                        if not missing_router_indices_warned:
                            logger.warning(
                                f"Gate output at layer {layer_idx} has no recognized routing indices. "
                                f"Output type: {type(output)}. Per-prompt tracking may be incomplete."
                            )
                            missing_router_indices_warned = True
                except Exception as e:
                    record_suppressed_error(
                        error=e,
                        context="_track_expert_activations_per_prompt_hook",
                        module="model",
                        severity="warning",
                        details={"layer_idx": layer_idx},
                    )

            return hook

        try:
            # Register hooks on each MoE layer's gate
            for layer_idx, layer in enumerate(self.get_layers()):
                if hasattr(layer.mlp, "gate"):
                    hook = layer.mlp.gate.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

            if not hooks:
                logger.debug("No MoE gates found for per-prompt tracking")
                return None

            # Process each prompt individually
            for prompt_idx, prompt in enumerate(prompts):
                captured_experts.clear()

                # Forward pass to capture expert activations
                try:
                    with torch.no_grad():
                        self.generate([prompt], max_new_tokens=1)
                except Exception as e:
                    record_suppressed_error(
                        error=e,
                        context="_track_expert_activations_per_prompt_forward",
                        module="model",
                        severity="warning",
                        details={"prompt_idx": prompt_idx},
                    )
                    continue

                # Store per-prompt results
                for layer_idx, experts in captured_experts.items():
                    if layer_idx not in per_prompt_activations:
                        per_prompt_activations[layer_idx] = {}
                    per_prompt_activations[layer_idx][prompt_idx] = experts.copy()

            logger.debug(
                f"Tracked per-prompt activations for {len(prompts)} prompts "
                f"across {len(per_prompt_activations)} layers"
            )
            return per_prompt_activations

        except Exception as e:
            logger.warning(f"Failed to track per-prompt expert activations: {e}")
            record_suppressed_error(
                error=e,
                context="_track_expert_activations_per_prompt",
                module="model",
                severity="error",
                details={"n_prompts": len(prompts)},
                include_traceback=True,
            )
            return None
        finally:
            # Clean up hooks
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass

    def compute_expert_compliance_scores(
        self,
        bad_prompts: list[str],
        evaluator: "Evaluator",
    ) -> dict[int, dict[int, float]]:
        """Compute compliance scores for each expert based on per-prompt activation correlation.

        For each expert, correlate its activation frequency with refusal outcomes
        across individual prompts. Experts that activate more on prompts that
        result in refusals get negative scores.

        Args:
            bad_prompts: List of harmful prompts
            evaluator: Evaluator instance for refusal detection

        Returns:
            layer_idx -> expert_idx -> score (-1 to 1)
            Returns empty dict if tracking fails or no data available.
        """
        logger.info("Computing expert compliance scores")

        # Track per-prompt expert activations
        per_prompt_activations = self._track_expert_activations_per_prompt(bad_prompts)
        if per_prompt_activations is None:
            logger.warning(
                "Could not track per-prompt activations, returning empty scores"
            )
            return {}

        if not per_prompt_activations:
            logger.warning("No activation data captured, returning empty scores")
            return {}

        # Generate responses and check for refusals
        logger.debug(f"Generating responses for {len(bad_prompts)} prompts")
        responses = self.get_responses_batched(bad_prompts)
        refusal_mask = [evaluator.is_refusal(r) for r in responses]

        n_refusals = sum(refusal_mask)
        logger.debug(
            f"Refusal detection: {n_refusals}/{len(bad_prompts)} prompts refused"
        )

        scores: dict[int, dict[int, float]] = {}
        for layer_idx in per_prompt_activations.keys():
            layer_scores: dict[int, float] = {}

            # per_prompt_activations[layer_idx] is dict: prompt_idx -> set of expert_idx
            prompt_experts = per_prompt_activations[layer_idx]

            # For each expert, compute correlation with refusal outcomes
            all_experts: set[int] = set()
            for experts in prompt_experts.values():
                all_experts.update(experts)

            for exp_idx in all_experts:
                # Count: how often does this expert activate on refusing vs compliant prompts?
                refusal_activations = 0
                comply_activations = 0

                for prompt_idx, experts in prompt_experts.items():
                    if exp_idx in experts:
                        if prompt_idx < len(refusal_mask) and refusal_mask[prompt_idx]:
                            refusal_activations += 1
                        else:
                            comply_activations += 1

                total = refusal_activations + comply_activations
                if total > 0:
                    # Score: -1 if only activates on refusals, +1 if only on compliance
                    compliance_score = (
                        comply_activations - refusal_activations
                    ) / total
                else:
                    compliance_score = 0.0

                layer_scores[exp_idx] = compliance_score

            scores[layer_idx] = layer_scores

        # Log summary statistics
        total_experts = sum(len(s) for s in scores.values())
        if total_experts > 0:
            avg_score = (
                sum(s for layer in scores.values() for s in layer.values())
                / total_experts
            )
            logger.info(
                f"Computed compliance scores for {total_experts} experts "
                f"across {len(scores)} layers (avg score: {avg_score:.3f})"
            )
        else:
            logger.warning("No expert compliance scores computed")

        return scores

    def modify_expert_biases(
        self,
        expert_scores: dict[int, dict[int, float]],
        bias_delta: float = 0.3,
    ) -> int:
        """Modify gate bias terms to prefer compliant experts.

        Args:
            expert_scores: layer_idx -> expert_idx -> compliance score (-1 to 1)
                Negative = refusing, Positive = compliant
            bias_delta: Maximum bias adjustment

        Returns:
            Number of biases modified
        """
        biases_modified = 0
        layers_without_bias = 0

        for layer_idx, scores in expert_scores.items():
            layer = self.get_layers()[layer_idx]

            if not hasattr(layer.mlp, "gate"):
                continue

            gate = layer.mlp.gate
            if not hasattr(gate, "e_score_correction"):
                layers_without_bias += 1
                continue

            bias = gate.e_score_correction.data

            for exp_idx, score in scores.items():
                if exp_idx < len(bias):
                    # score > 0 means compliant, increase bias
                    # score < 0 means refusing, decrease bias
                    adjustment = score * bias_delta
                    bias[exp_idx] += adjustment
                    biases_modified += 1

        if layers_without_bias > 0:
            logger.warning(
                f"Bias manipulation skipped for {layers_without_bias} layers "
                f"(no e_score_correction attribute)"
            )

        logger.info(f"Modified {biases_modified} expert biases")
        return biases_modified

    def abliterate_moe_two_stage(
        self,
        refusal_directions: Tensor,
        bad_prompts: list[str],
        parameters: dict[str, AbliterationParameters],
        evaluator: "Evaluator",
        gate_strength: float = 0.3,
        expert_threshold: float = 0.1,
        layer_profiles: list["LayerRangeProfile"] | None = None,
        use_mpoa: bool = False,
        mpoa_norm_mode: str = "row",
        mpoa_min_scale: float = 0.5,
        mpoa_max_scale: float = 2.0,
        use_bias_manipulation: bool = True,
        bias_delta: float = 0.3,
    ) -> dict[str, int]:
        """Two-stage MoE abliteration: gate first, then re-track and abliterate experts.

        Stage 1: Abliterate gates to disrupt refusal detection
        Stage 2: Re-track expert activations (routing patterns changed!)
        Stage 3: Abliterate remaining high-activation experts
        Stage 4: (Optional) Modify biases to prefer compliant experts

        This addresses the whack-a-mole problem where abliterating one expert
        causes routing to shift to another refusing expert.

        Args:
            refusal_directions: Per-layer refusal directions
            bad_prompts: Prompts that trigger refusals
            parameters: Abliteration parameters per component
            evaluator: Evaluator instance for refusal detection (used in bias manipulation)
            gate_strength: Strength for gate abliteration
            expert_threshold: Activation threshold for expert targeting
            layer_profiles: Optional per-layer weight profiles
            use_mpoa: Use MPOA norm preservation
            mpoa_norm_mode: MPOA norm mode
            mpoa_min_scale: MPOA minimum scale
            mpoa_max_scale: MPOA maximum scale
            use_bias_manipulation: Whether to apply bias manipulation
            bias_delta: Bias manipulation strength

        Returns:
            Statistics dict with counts of abliterated components
        """
        stats: dict[str, int] = {
            "gates_ablated": 0,
            "experts_ablated": 0,
            "shared_experts_ablated": 0,
            "attn_ablated": 0,
            "biases_modified": 0,
            "routing_collapsed": 0,
        }

        # Stage 1: Gate abliteration
        logger.info("Two-stage MoE abliteration: Stage 1 - Abliterating gates")
        stats["gates_ablated"] = self.abliterate_gates(
            refusal_directions,
            strength=gate_strength,
            layer_profiles=layer_profiles,
        )
        logger.debug(f"Gates ablated: {stats['gates_ablated']}")

        # Check routing health after gate ablation
        if not self.check_routing_health(bad_prompts[:20]):
            logger.warning("Routing collapsed after gate ablation")
            stats["routing_collapsed"] = 1
            return stats

        # Stage 2: Re-track activations with modified routing
        logger.info(
            "Two-stage MoE abliteration: Stage 2 - Re-tracking expert activations"
        )
        new_activations = self.track_expert_activations(bad_prompts)

        if new_activations is None:
            logger.warning("Could not re-track activations after gate ablation")
            return stats

        # Get newly targeted experts
        targeted_experts = self.get_moe_targeted_experts(
            new_activations,
            threshold=expert_threshold,
            top_k=self.settings.moe_top_k_experts,
        )
        logger.debug(
            "Targeted experts after re-tracking",
            n_layers_with_targets=len(targeted_experts),
            total_experts_targeted=sum(len(e) for e in targeted_experts.values()),
        )

        # Stage 3: Targeted expert abliteration
        logger.info(
            "Two-stage MoE abliteration: Stage 3 - Abliterating targeted experts"
        )
        expert_stats = self.abliterate_moe_targeted(
            refusal_directions,
            parameters,
            targeted_experts,
            layer_profiles=layer_profiles,
            use_mpoa=use_mpoa,
            mpoa_norm_mode=mpoa_norm_mode,
            mpoa_min_scale=mpoa_min_scale,
            mpoa_max_scale=mpoa_max_scale,
        )

        stats["experts_ablated"] = expert_stats.get("experts_ablated", 0)
        stats["shared_experts_ablated"] = expert_stats.get("shared_experts_ablated", 0)
        stats["attn_ablated"] = expert_stats.get("attn_ablated", 0)

        # Stage 4: (Optional) Bias manipulation
        if use_bias_manipulation:
            logger.info(
                "Two-stage MoE abliteration: Stage 4 - Computing compliance scores and modifying biases"
            )
            compliance_scores = self.compute_expert_compliance_scores(
                bad_prompts, evaluator
            )
            if compliance_scores:
                stats["biases_modified"] = self.modify_expert_biases(
                    compliance_scores, bias_delta
                )

        logger.info(
            "Two-stage MoE abliteration complete",
            gates_ablated=stats["gates_ablated"],
            experts_ablated=stats["experts_ablated"],
            shared_experts_ablated=stats["shared_experts_ablated"],
            biases_modified=stats["biases_modified"],
        )

        return stats

    def setup_moe_targeting(
        self,
        bad_prompts: list[str],
    ) -> dict[int, list[int]] | None:
        """Set up MoE targeting by tracking expert activations on bad prompts.

        This is a convenience method that:
        1. Checks if MoE targeting should be used
        2. Tracks expert activations on bad prompts
        3. Returns targeted experts per layer

        Args:
            bad_prompts: List of prompts that trigger refusals

        Returns:
            Dict mapping layer_idx -> expert indices to target,
            or None if MoE targeting is not applicable

        Note:
            Returns None (not raises) for non-fatal failures to allow
            fallback to standard abliteration.
        """
        if not self.should_use_moe_targeting():
            return None

        if not bad_prompts:
            record_suppressed_error(
                error=None,
                context="setup_moe_targeting",
                module="model",
                severity="warning",
                details={"reason": "Empty bad_prompts list"},
            )
            print(
                "  * [yellow]Cannot set up MoE targeting: no bad prompts provided[/yellow]"
            )
            return None

        print("* Setting up MoE router-aware expert targeting...")
        logger.info(
            "Setting up MoE targeting",
            n_bad_prompts=len(bad_prompts),
            threshold=self.settings.moe_expert_activation_threshold,
            top_k=self.settings.moe_top_k_experts,
        )

        # Track activations
        try:
            activations = self.track_expert_activations(bad_prompts)
        except Exception as e:
            record_suppressed_error(
                error=e,
                context="setup_moe_targeting",
                module="model",
                severity="error",
                details={"n_prompts": len(bad_prompts)},
                include_traceback=True,
            )
            print(f"  * [yellow]Expert activation tracking failed: {e}[/yellow]")
            return None

        if activations is None:
            record_suppressed_error(
                error=None,
                context="setup_moe_targeting",
                module="model",
                severity="warning",
                details={"reason": "track_expert_activations returned None"},
            )
            print("  * [yellow]Could not track expert activations[/yellow]")
            return None

        # Get targeted experts
        targeted = self.get_moe_targeted_experts(
            activations,
            threshold=self.settings.moe_expert_activation_threshold,
            top_k=self.settings.moe_top_k_experts,
        )

        # Print summary
        total_targeted = sum(len(experts) for experts in targeted.values())
        total_possible = activations.n_experts_per_layer * len(targeted)

        if total_targeted == 0:
            record_suppressed_error(
                error=None,
                context="setup_moe_targeting",
                module="model",
                severity="warning",
                details={
                    "reason": "No experts targeted",
                    "threshold": self.settings.moe_expert_activation_threshold,
                    "top_k": self.settings.moe_top_k_experts,
                    "n_layers": len(targeted),
                    "advice": "Lower moe_expert_activation_threshold or check expert tracking",
                },
            )
            print(
                f"[yellow]  * Warning: No experts met activation threshold ({self.settings.moe_expert_activation_threshold}). "
                f"Consider lowering threshold.[/yellow]"
            )

        if total_possible > 0:
            pct = (total_targeted / total_possible) * 100
            print(
                f"  * Targeting {total_targeted} experts across {len(targeted)} layers "
                f"({pct:.1f}% of {activations.n_experts_per_layer} experts per layer)"
            )
        else:
            print(
                f"  * Targeting {total_targeted} experts across {len(targeted)} layers"
            )

        logger.info(
            "MoE targeting setup complete",
            total_targeted=total_targeted,
            layers_with_targets=len(targeted),
            n_experts_per_layer=activations.n_experts_per_layer,
        )

        return targeted

    def get_abliteration_error_summary(self) -> str:
        """Get a summary of abliteration-related errors from the error tracker.

        Returns:
            Human-readable summary of abliteration errors, or empty string if none
        """
        from .error_tracker import error_tracker

        # Get all errors related to abliteration
        abliteration_contexts = [
            "abliterate",
            "abliterate_layer",
            "abliterate_moe",
            "abliterate_circuit",
            "get_residuals",
            "track_expert",
            "setup_moe",
            "get_layer_matrices",
        ]

        abliteration_errors = []
        for error in error_tracker.get_all():
            if any(ctx in error.context for ctx in abliteration_contexts):
                abliteration_errors.append(error)

        if not abliteration_errors:
            return ""

        lines = [
            f"Abliteration encountered {len(abliteration_errors)} issues:",
        ]

        # Group by severity
        by_severity: dict[str, list] = {}
        for error in abliteration_errors:
            sev = error.severity.value
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(error)

        for severity in ["error", "warning", "info"]:
            if severity in by_severity:
                lines.append(f"  {severity.upper()}: {len(by_severity[severity])}")

        # Show most recent errors
        lines.append("Most recent issues:")
        for error in abliteration_errors[-5:]:
            lines.append(
                f"  - [{error.severity.value}] {error.context}: {error.error_message}"
            )

        return "\n".join(lines)

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
