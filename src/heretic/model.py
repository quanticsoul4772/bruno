# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import copy
import math
from contextlib import suppress
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
    ConfigurationError,
    ModelLoadError,
    SacredDirectionError,
    SupervisedProbeError,
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
        clf = LogisticRegression(
            penalty="l2",
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
                settings.model
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
            if layer_index % CACHE_CLEAR_INTERVAL == (CACHE_CLEAR_INTERVAL - 1):
                empty_cache()

        # Clear the device projector cache to prevent memory leaks
        device_projectors_cache.clear()

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

    # Phase 2: Supervised Refusal Probing
    def get_refusal_directions_supervised(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        bad_prompts: list[str],
        evaluator: "Evaluator",
        min_probe_accuracy: float = 0.65,
    ) -> SupervisedExtractionResult | PCAExtractionResult:
        """Extract refusal directions using supervised probing.

        Unlike PCA which finds variance directions, this finds the actual
        decision boundary between refuse/comply by training linear classifiers.

        The method:
        1. Generates responses to bad prompts
        2. Labels each prompt as "triggered refusal" or "got compliance"
        3. Trains a logistic regression probe per layer
        4. Uses the learned weights as refusal directions

        Falls back to PCA extraction if:
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
            SupervisedExtractionResult if successful, or PCAExtractionResult as fallback
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
            print(
                f"[yellow]Warning: Imbalanced classes (refusals={n_refusals}, comply={n_comply}). "
                f"Falling back to PCA extraction.[/yellow]"
            )
            return self.get_refusal_directions_pca(
                good_residuals, bad_residuals, n_components=1
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

        # Prepare data for parallel training (convert to numpy once, outside the loop)
        # This avoids repeated tensor->numpy conversions and CUDA serialization issues
        refusal_np = refusal_residuals.cpu().numpy()
        comply_np = comply_residuals.cpu().numpy()

        # Train all probes in parallel using joblib
        from joblib import Parallel, delayed

        def train_probe_for_layer(
            layer_idx: int,
            refusal_data: np.ndarray,
            comply_data: np.ndarray,
        ) -> tuple[np.ndarray, float]:
            """Train a single probe for one layer (designed for parallel execution)."""
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            # Extract data for this layer
            X = np.concatenate(
                [refusal_data[:, layer_idx, :], comply_data[:, layer_idx, :]],
                axis=0,
            )
            y = np.concatenate([np.ones(len(refusal_data)), np.zeros(len(comply_data))])

            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )

            # Cross-validate to check probe quality
            if len(X) >= 10:
                n_splits = min(5, len(X) // 2)
                if n_splits >= 2:
                    cv_scores = cross_val_score(clf, X, y, cv=n_splits)
                    accuracy = float(cv_scores.mean())
                else:
                    accuracy = 0.5
            else:
                accuracy = 0.5

            # Fit on all data
            clf.fit(X, y)

            # Extract and normalize weights
            weights = clf.coef_[0]
            weights = weights / (np.linalg.norm(weights) + EPSILON)

            return weights, accuracy

        # Run parallel training across all layers
        # n_jobs=-1 uses all available CPU cores
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(train_probe_for_layer)(layer_idx, refusal_np, comply_np)
            for layer_idx in range(n_layers)
        )

        # Unpack results and convert weights back to torch tensors
        for weights_np, accuracy in results:
            directions.append(torch.from_numpy(weights_np).float().to(device))
            accuracies.append(accuracy)

        mean_acc = sum(accuracies) / len(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)

        print(
            f"  * Probe accuracies: min={min_acc:.2f}, max={max_acc:.2f}, mean={mean_acc:.2f}"
        )

        # Check accuracy threshold - fall back to PCA if too low
        if mean_acc < min_probe_accuracy:
            print(
                f"[yellow]Warning: Mean probe accuracy ({mean_acc:.2f}) below threshold "
                f"({min_probe_accuracy}). Falling back to PCA extraction.[/yellow]"
            )
            return self.get_refusal_directions_pca(
                good_residuals, bad_residuals, n_components=1
            )

        return SupervisedExtractionResult(
            directions=torch.stack(directions),
            accuracies=accuracies,
        )

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
    ) -> ConceptConeExtractionResult | PCAExtractionResult:
        """Extract refusal directions using concept cone clustering.

        Clusters harmful prompts by their residual patterns to identify
        different categories of harmful content (e.g., violence, fraud,
        self-harm), then extracts category-specific refusal directions.

        Falls back to standard PCA if:
        - Not enough samples for clustering
        - Silhouette score below threshold (poor clustering quality)

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
            ConceptConeExtractionResult if successful,
            or PCAExtractionResult if fallback to global directions
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
            print(
                f"[yellow]Warning: Silhouette score ({result.silhouette_score:.3f}) "
                f"below threshold ({min_silhouette_score}). "
                f"Falling back to global PCA directions.[/yellow]"
            )
            return self.get_refusal_directions_pca(
                good_residuals, bad_residuals, n_components=directions_per_cone
            )

        # Check if we have any valid cones
        if not result.cones:
            print(
                "[yellow]Warning: No valid concept cones extracted. "
                "Falling back to global PCA directions.[/yellow]"
            )
            return self.get_refusal_directions_pca(
                good_residuals, bad_residuals, n_components=directions_per_cone
            )

        return result

    def abliterate_concept_cones(
        self,
        cone_result: "ConceptConeExtractionResult",
        parameters: dict[str, "AbliterationParameters"],
        direction_weights: list[float] | None = None,
        layer_profiles: list["LayerRangeProfile"] | None = None,
    ) -> None:
        """Abliterate using concept cone-specific directions.

        Applies ablation for each concept cone's primary direction,
        weighted by the cone's size (number of prompts).

        Args:
            cone_result: Result from get_refusal_directions_concept_cones
            parameters: Abliteration parameters for each component
            direction_weights: Optional weights for each direction component.
                If None, uses first direction only for each cone.
            layer_profiles: Optional per-layer weight profiles
        """
        if not cone_result.cones:
            print("[yellow]Warning: No concept cones to abliterate.[/yellow]")
            return

        total_size = sum(cone.size for cone in cone_result.cones)

        print(f"  * Abliterating {len(cone_result.cones)} concept cones...")

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
            if direction_weights is None:
                # Use first direction only
                primary_direction = cone.directions[:, 0, :]
                self.abliterate(
                    primary_direction, None, scaled_parameters, layer_profiles
                )
            else:
                # Use multiple directions with weights
                self.abliterate_multi_direction(
                    cone.directions,
                    direction_weights,
                    scaled_parameters,
                    layer_profiles,
                )

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

        # Try supervised directions
        supervised_result = self.get_refusal_directions_supervised(
            good_residuals, bad_residuals, bad_prompts, evaluator, min_probe_accuracy
        )

        # Check if supervised probing succeeded or fell back to PCA
        if isinstance(supervised_result, PCAExtractionResult):
            raise SupervisedProbeError(
                "Ensemble probe+PCA extraction failed: supervised probing returned PCA fallback. "
                "This happens when there is class imbalance (< 10 samples per class) or "
                f"probe accuracy is below {min_probe_accuracy}. Check that your bad_prompts "
                "dataset contains prompts that trigger refusals."
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
    ) -> bool:
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

        Returns:
            True if CAA addition was applied, False if skipped due to high overlap
        """
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
        if refusal_directions.dim() == 3:
            # Multi-direction: use first component (primary direction)
            self.abliterate(
                refusal_directions[:, 0, :],
                None,  # Per-layer mode
                scaled_parameters,
                layer_profiles=layer_profiles,
            )
        else:
            # Single direction
            self.abliterate(
                refusal_directions,
                None,  # Per-layer mode
                scaled_parameters,
                layer_profiles=layer_profiles,
            )

        # Step 2: Add compliance direction (only if overlap is acceptable)
        if caa_applied:
            print(
                f"  * CAA: Adding compliance direction (strength={addition_strength:.2f})"
            )

            num_layers = len(self.get_layers())

            for layer_idx in range(num_layers):
                layer = self.get_layers()[layer_idx]

                # Get layer-specific compliance direction
                compliance_dir = compliance_direction[layer_idx]

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
                o_proj.data.add_(effective_strength * (device_projector @ o_proj.data))

                # Clean up periodically
                if layer_idx % CACHE_CLEAR_INTERVAL == (CACHE_CLEAR_INTERVAL - 1):
                    empty_cache()

        return caa_applied

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
            # o_proj[start_idx:end_idx, :] -= strength * (projector @ o_proj[start_idx:end_idx, :])
            head_weights = o_proj.data[start_idx:end_idx, :]
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
        """Save discovered circuits to a JSON cache file.

        Args:
            circuits: List of RefusalCircuit objects to save
            cache_path: Path to the JSON cache file
        """
        import json

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

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"  * Saved {len(circuits)} circuits to {cache_path}")

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
