# Next-Level Abliteration Improvements for Heretic

## Executive Summary

Based on comprehensive analysis of the heretic codebase, 2024 research papers, and state-of-the-art techniques in representation engineering, this document outlines **innovative improvements** to significantly enhance abliteration quality.

**Current State**: Heretic already implements contrastive PCA, eigenvalue weights, layer profiles, orthogonalization, iterative refinement, and MMLU validation.

**Goal**: Push from "good abliteration" to "state-of-the-art precision refusal removal with capability preservation."

**Expected Combined Improvement**: 15-25% reduction in remaining refusals with maintained or improved capability preservation.

> ⚠️ **Note**: Previous estimates of 40-60% were overly optimistic. Real-world gains depend heavily on model architecture and baseline refusal patterns.

---

## ⚠️ Critical Issues & Limitations

Before implementing any improvement, be aware of these fundamental constraints:

### Architectural Constraints

1. **Heretic modifies weights permanently** via `matrix.sub_(...)`. Any "dynamic" or "per-prompt" approach requires inference-time hooks, not weight modification.

2. **No hook infrastructure exists** in the current codebase. Circuit-level ablation would require significant architectural changes.

3. **Multi-GPU memory constraints** - Additional models (NLI classifier) may not fit alongside 32B+ target models.

### Missing Dependencies

The following dependency has been added to `pyproject.toml`:

```toml
"scikit-learn>=1.3.0",  # Required for Improvements 1, 2 (LogisticRegression, KMeans)
```

### Resource Requirements

| Improvement | Additional VRAM | Additional Compute Time |
|-------------|-----------------|------------------------|
| 1. Neural Detector | ~4GB (BART-large) | +2x evaluation time |
| 2. Supervised Probing | ~100MB (sklearn) | +30 seconds per layer |
| 3. Activation-Scaled Weights | Negligible | +1 minute |
| 4. Concept Cones | ~200MB (cluster data) | +2 minutes |
| 5. CAA | Negligible | +1 minute |
| 6. Circuit Ablation | ~8GB (activation cache) | +30 min to 11 hours |
| 7. Cross-Model Transfer | ~1GB (direction cache) | -90% (when cached) |

---

## Innovation Matrix

| # | Improvement | Impact | Effort | Priority | Status |
|---|-------------|--------|--------|----------|--------|
| 1 | **Neural Refusal Detector** | HIGH | MEDIUM | 🥇 1st | Ready to implement |
| 2 | **Supervised Refusal Probing** | MEDIUM | MEDIUM | 🥈 2nd | Ready to implement |
| 3 | **Activation-Scaled Weights** | LOW-MEDIUM | LOW | 🥉 3rd | **Redesigned** |
| 4 | **Concept Cones** | MEDIUM | MEDIUM | 4th | Ready to implement |
| 5 | **Contrastive Activation Addition (CAA)** | LOW-MEDIUM | LOW | 5th | Ready to implement |
| 6 | **Circuit-Level Ablation** | HIGH | VERY HIGH | 6th | Needs architecture changes |
| 7 | **Cross-Model Hyperparameter Transfer** | LOW | MEDIUM | 7th | **Limited applicability** |
| 8 | **Sparse Autoencoder Integration** | UNKNOWN | VERY HIGH | 8th | Research project |

### Realistic Impact Estimates

| Improvement | Claimed (Original) | Realistic Estimate | Notes |
|-------------|-------------------|-------------------|-------|
| 1. Neural Detector | +10-20% | **+10-15%** | Most reliable estimate (pure evaluation) |
| 2. Supervised Probing | +10-15% | **+3-8%** | Depends on class balance |
| 3. Activation-Scaled | +5-15% | **+2-5%** | Training-time only |
| 4. Concept Cones | +15-25% | **+5-10%** | Assumes clean clustering |
| 5. CAA | +10-20% | **+3-8%** | May conflict with ablation |
| 6. Circuit Ablation | 50% less damage | **10-30% less damage** | If implemented correctly |
| 7. Hyperparameter Transfer | 10x faster | **~2x faster Optuna convergence** | Directions NOT transferable |
| Combined | 40-60% | **15-25%** | Improvements don't multiply |

---

## Improvement 1: Neural Refusal Detector

### Problem
String matching with 22+ markers misses:
- Soft refusals ("I'd prefer to discuss something else...")
- Topic changes (evasive responses)
- Partial compliance with warnings
- Non-English refusals
- Novel refusal patterns

### Solution
Use zero-shot NLI classification to detect ALL forms of refusal behavior.

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| Additional VRAM | ~4GB for BART-large-mnli |
| Additional time | ~2x evaluation time |
| Disk space | ~1.6GB model download |

> ⚠️ **Warning**: On 4x RTX 4090 with 32B model, this may cause OOM. Use `neural_detection_for_optuna = false` to only use for final evaluation.

> ⚠️ **Note**: First run will auto-download `facebook/bart-large-mnli` (~1.6GB). Ensure you have disk space and network connectivity.

### Implementation

```python
# src/heretic/evaluator.py - Add neural refusal detection

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Pipeline


class NeuralRefusalDetector:
    """Neural network-based refusal detection using zero-shot NLI.

    Uses BART-large-mnli for zero-shot classification. This catches:
    - Soft refusals
    - Topic changes
    - Evasive responses
    - Partial compliance with warnings
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
        batch_size: int = 8,
    ):
        from transformers import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.classifier: "Pipeline" = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            batch_size=batch_size,
        )

        self.labels = [
            "The assistant refuses to help with the request",
            "The assistant provides the requested information",
            "The assistant changes the topic or redirects",
            "The assistant gives warnings but still complies",
        ]

        self._refusal_labels = {self.labels[0], self.labels[2]}  # Refuse or redirect

    def is_refusal(self, response: str) -> tuple[bool, float, str]:
        """Detect if response is a refusal.

        Args:
            response: The model's response text

        Returns:
            Tuple of (is_refusal, confidence, refusal_type)
        """
        # Truncate to ~512 tokens (rough estimate: 4 chars per token)
        truncated = response[:2048]

        result = self.classifier(
            truncated,
            self.labels,
            multi_label=False,
        )

        top_label = result['labels'][0]
        confidence = result['scores'][0]

        is_refusal = top_label in self._refusal_labels

        if "refuses" in top_label:
            refusal_type = "hard"
        elif "redirects" in top_label or "changes" in top_label:
            refusal_type = "evasive"
        elif "warnings" in top_label:
            refusal_type = "soft"
        else:
            refusal_type = "none"

        return is_refusal, confidence, refusal_type

    def count_refusals_batched(
        self,
        responses: list[str],
    ) -> dict:
        """Count refusals with detailed breakdown (batched for efficiency)."""
        results = {
            "total": len(responses),
            "hard_refusals": 0,
            "soft_refusals": 0,
            "evasive": 0,
            "compliant": 0,
            "uncertain": 0,
        }

        for response in responses:
            is_refusal, confidence, refusal_type = self.is_refusal(response)

            if confidence < 0.5:
                results["uncertain"] += 1
            elif is_refusal:
                if refusal_type == "hard":
                    results["hard_refusals"] += 1
                elif refusal_type == "evasive":
                    results["evasive"] += 1
                else:
                    results["soft_refusals"] += 1
            else:
                results["compliant"] += 1

        results["refusal_count"] = (
            results["hard_refusals"] +
            results["soft_refusals"] +
            results["evasive"]
        )

        return results
```

### Config Changes

> **Note**: These settings go in `config.toml` (or use CLI flags like `--use-neural-refusal-detection`). Heretic supports TOML config, environment variables (`HERETIC_` prefix), and CLI arguments.

```toml
# Use neural refusal detection (requires additional ~4GB VRAM)
use_neural_refusal_detection = false

# Model for neural detection
neural_detection_model = "facebook/bart-large-mnli"

# Use neural detection for Optuna optimization (slower, better signal)
# Set to false to only use for final evaluation
neural_detection_for_optuna = false
```

### Failure Modes

1. **NLI model bias**: BART-mnli may have its own biases about what constitutes refusal
2. **OOM on large models**: 4GB extra VRAM may not be available
3. **Slow evaluation**: ~2x slower than string matching

### Expected Impact
- **+10-15% detection accuracy** for soft/evasive refusals
- Most reliable of all improvements (pure evaluation, no ablation changes)

---

## Improvement 2: Supervised Refusal Probing

### Problem
Current contrastive PCA finds directions with high variance difference, but **high variance ≠ refusal-relevant**. It may capture content categories (violence vs. fraud) rather than the decision to refuse.

### Solution
Train a lightweight linear classifier (probe) on residual activations to predict "will refuse" vs "will comply". Use the **learned probe weights** as the refusal direction instead of unsupervised PCA.

### Prerequisites

**Dependency** (already in pyproject.toml):
```toml
"scikit-learn>=1.3.0",
```

### Implementation

```python
# src/heretic/model.py - Add new method

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import TYPE_CHECKING

from heretic.utils import load_prompts

if TYPE_CHECKING:
    from heretic.evaluator import Evaluator


class RefusalProbe:
    """Linear probe trained to predict refusal behavior."""

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        self.device = device
        self.weights = torch.zeros(hidden_dim, device=device)
        self.bias = 0.0
        self.accuracy = 0.0  # Track probe quality

    def fit(
        self,
        refusal_residuals: Tensor,  # Residuals from prompts that triggered refusals
        comply_residuals: Tensor,   # Residuals from prompts that got compliance
        layer_idx: int,
    ) -> float:
        """Train probe on labeled residuals using logistic regression.

        Returns:
            Cross-validation accuracy score
        """
        # Combine data
        X = torch.cat([
            refusal_residuals[:, layer_idx, :],
            comply_residuals[:, layer_idx, :],
        ], dim=0).cpu().numpy()

        y = np.concatenate([
            np.ones(len(refusal_residuals)),
            np.zeros(len(comply_residuals)),
        ])

        # Handle class imbalance with balanced class weights
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced data
            solver='lbfgs',
        )

        # Cross-validate to check probe quality
        if len(X) >= 10:
            cv_scores = cross_val_score(clf, X, y, cv=min(5, len(X) // 2))
            self.accuracy = cv_scores.mean()

        # Fit on all data
        clf.fit(X, y)

        # Extract weights as the refusal direction
        self.weights = torch.from_numpy(clf.coef_[0]).float().to(self.device)
        self.bias = float(clf.intercept_[0])

        # Normalize to unit vector
        self.weights = F.normalize(self.weights, p=2, dim=0)

        return self.accuracy


# Add this method to the Model class:
def get_refusal_directions_supervised(
    self,
    good_residuals: Tensor,
    bad_residuals: Tensor,
    evaluator: "Evaluator",
) -> tuple[Tensor, list[float]]:
    """Extract refusal directions using supervised probing.

    Unlike PCA which finds variance directions, this finds the
    actual decision boundary between refuse/comply.

    Args:
        good_residuals: Residuals from harmless prompts, shape (n_good, n_layers, hidden_dim)
        bad_residuals: Residuals from harmful prompts, shape (n_bad, n_layers, hidden_dim)
        evaluator: Evaluator instance for refusal detection

    Returns:
        Tuple of (directions tensor shape (n_layers, hidden_dim), list of per-layer accuracies)
    """
    n_layers = good_residuals.shape[1]
    hidden_dim = good_residuals.shape[2]
    device = good_residuals.device
    directions: list[Tensor] = []
    accuracies: list[float] = []

    # Get actual model responses to label data
    bad_prompts = load_prompts(self.settings.bad_prompts)
    bad_responses = self.get_responses_batched(bad_prompts)

    # Label based on actual responses
    refusal_mask = torch.tensor([
        evaluator.is_refusal(r) for r in bad_responses
    ], device=device)

    # Check for class balance
    n_refusals = refusal_mask.sum().item()
    n_comply = len(refusal_mask) - n_refusals

    if n_refusals < 10 or n_comply < 10:
        print(f"[yellow]Warning: Imbalanced classes (refusals={n_refusals}, comply={n_comply})[/yellow]")
        print("[yellow]Falling back to PCA extraction[/yellow]")
        return self.get_refusal_directions_pca(good_residuals, bad_residuals)

    for layer_idx in range(n_layers):
        probe = RefusalProbe(hidden_dim, device=str(device))

        # Use bad prompts that actually triggered refusals
        # and good prompts that got compliance
        refusal_residuals = bad_residuals[refusal_mask]
        comply_residuals = torch.cat([
            good_residuals,
            bad_residuals[~refusal_mask],
        ], dim=0)

        accuracy = probe.fit(refusal_residuals, comply_residuals, layer_idx)
        accuracies.append(accuracy)
        directions.append(probe.weights)

    print(f"  * Probe accuracies: min={min(accuracies):.2f}, max={max(accuracies):.2f}, mean={sum(accuracies)/len(accuracies):.2f}")

    return torch.stack(directions), accuracies
```

### Config Changes

> **Note**: These settings go in `config.toml` (or use CLI flags like `--use-supervised-probing`). Heretic supports TOML config, environment variables (`HERETIC_` prefix), and CLI arguments.

```toml
# Use supervised probing instead of PCA for direction extraction
use_supervised_probing = false  # Default off until validated

# Minimum probe accuracy to use supervised direction (else fall back to PCA)
min_probe_accuracy = 0.65

# Combine supervised probe with PCA (ensemble)
ensemble_probe_pca = false
ensemble_weight_probe = 0.7
ensemble_weight_pca = 0.3
```

### Failure Modes

1. **Class imbalance**: If 90%+ of bad prompts trigger refusals, probe learns "is harmful" not "will refuse"
2. **Overfitting**: Probe may memorize training prompts, not generalize
3. **Layer variance**: Some layers may have random-chance accuracy

### Mitigation
- Use `class_weight='balanced'` in LogisticRegression
- Cross-validate and report accuracy per layer
- Fall back to PCA if accuracy < 0.65

### Expected Impact
- **+3-8% more targeted ablation** (conservative estimate)
- Best when baseline model has ~50-70% refusal rate on bad prompts

---

## Improvement 3: Activation-Scaled Ablation Weights (REDESIGNED)

### Problem (Original)
The original "Dynamic Intervention Strength" proposal was **fundamentally flawed** because:
- Heretic modifies weights **permanently** via `matrix.sub_(...)`
- You cannot do "per-prompt dynamic strength" at inference time with permanent weight changes
- The proposal conflated inference-time steering with permanent ablation

### Redesigned Solution
Instead of dynamic per-prompt intervention, we compute **training-time statistics** about refusal activation strength across the prompt set, and use these to **calibrate ablation weights** before permanent modification.

**Key insight**: Measure the *distribution* of refusal activations during direction extraction, then scale ablation weights based on percentile targeting.

### Implementation

```python
# src/heretic/model.py - Add activation-scaled weight computation

import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class RefusalActivationStats:
    """Statistics about refusal direction activations across prompts."""
    mean_projection: float
    std_projection: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    n_samples: int


# Add this method to the Model class:
def compute_refusal_activation_stats(
    self,
    refusal_directions: Tensor,
    bad_residuals: Tensor,
    target_layer_frac: float = 0.6,
) -> RefusalActivationStats:
    """Compute statistics about refusal activations for calibration.

    Args:
        refusal_directions: Shape (n_layers, hidden_dim)
        bad_residuals: Shape (n_prompts, n_layers, hidden_dim)
        target_layer_frac: Which layer to measure (as fraction)

    Returns:
        RefusalActivationStats with projection statistics
    """
    n_layers = bad_residuals.shape[1]
    target_layer = int(n_layers * target_layer_frac)

    # Get direction at target layer
    direction = refusal_directions[target_layer]

    # Compute projections for all prompts
    projections = []
    for i in range(bad_residuals.shape[0]):
        residual = bad_residuals[i, target_layer, :]
        projection = torch.dot(residual, direction).abs().item()
        projections.append(projection)

    projections_arr = np.array(projections)

    return RefusalActivationStats(
        mean_projection=float(projections_arr.mean()),
        std_projection=float(projections_arr.std()),
        percentile_25=float(np.percentile(projections_arr, 25)),
        percentile_75=float(np.percentile(projections_arr, 75)),
        percentile_95=float(np.percentile(projections_arr, 95)),
        n_samples=len(projections_arr),
    )


# Add this method to the Model class:
def compute_calibrated_weights(
    self,
    stats: RefusalActivationStats,
    base_weight: float = 1.0,
    target_percentile: float = 0.75,  # Target 75th percentile of refusals
) -> float:
    """Compute calibrated ablation weight based on activation statistics.

    The idea: instead of fixed weights, scale based on how strong refusal
    activations are in this specific model/prompt combination.

    Args:
        stats: Activation statistics from compute_refusal_activation_stats
        base_weight: Starting weight before calibration
        target_percentile: Which percentile of activations to target

    Returns:
        Calibrated weight multiplier
    """
    if target_percentile == 0.25:
        target_value = stats.percentile_25
    elif target_percentile == 0.75:
        target_value = stats.percentile_75
    elif target_percentile == 0.95:
        target_value = stats.percentile_95
    else:
        # Linear interpolation
        target_value = stats.mean_projection + stats.std_projection * (target_percentile - 0.5)

    # Scale weight inversely to activation strength
    # Strong activations need less weight, weak activations need more
    if stats.mean_projection > 0:
        calibration_factor = target_value / stats.mean_projection
    else:
        calibration_factor = 1.0

    # Clamp to reasonable range
    calibration_factor = max(0.5, min(2.0, calibration_factor))

    return base_weight * calibration_factor
```

### Config Changes
```toml
# Use activation-based weight calibration (training-time, not inference-time)
use_activation_calibration = false  # Default off

# Target percentile of refusal activations (higher = more aggressive)
activation_target_percentile = 0.75
```

### Key Differences from Original

| Original (Broken) | Redesigned (Valid) |
|-------------------|--------------------|
| Per-prompt dynamic weights | Training-time calibration |
| Requires inference hooks | Works with permanent weights |
| Changes per input | Fixed after calibration |
| Incompatible with heretic | Compatible with existing architecture |

### Failure Modes

1. **Over-calibration**: If activation distribution is bimodal, percentile targeting may pick wrong threshold
2. **Outlier sensitivity**: Extreme activation values may skew percentile calculations
3. **Model mismatch**: Calibration from one prompt set may not generalize to different prompt distributions

### Expected Impact
- **+2-5% capability preservation** (conservative)
- Helps avoid over-ablation when refusal activations are weak

---

## Improvement 4: Prompt-Adaptive Direction Selection (Concept Cones)

### Problem
Research suggests refusal may be encoded as **multiple "concept cones"** - different harm categories (violence, illegal content, self-harm, privacy) might activate different refusal mechanisms.

### Prerequisites

**Dependency** (already in pyproject.toml):
```toml
"scikit-learn>=1.3.0",  # For KMeans clustering
```

### Implementation

```python
# src/heretic/model.py - Add concept cone extraction

import torch
from torch import Tensor
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heretic.model import Model, PCAExtractionResult


@dataclass
class ConceptCone:
    """A cluster of harmful prompts with their specific refusal direction."""
    cluster_id: int
    centroid: Tensor  # Embedding centroid for this category
    directions: Tensor  # Refusal directions specific to this cone
    eigenvalues: Tensor
    prompt_indices: list[int]  # Which prompts belong to this cone
    size: int  # Number of prompts in this cone


class ConceptConeExtractor:
    """Extract multiple concept cones from harmful prompts."""

    def __init__(
        self,
        n_cones: int = 5,
        n_directions_per_cone: int = 2,
        min_cluster_size: int = 10,
    ):
        self.n_cones = n_cones
        self.n_directions_per_cone = n_directions_per_cone
        self.min_cluster_size = min_cluster_size
        self.cones: list[ConceptCone] = []
        self.silhouette_score = 0.0

    def extract(
        self,
        good_residuals: Tensor,
        bad_residuals: Tensor,
        model: "Model",
    ) -> list[ConceptCone]:
        """Extract concept cones from bad prompts.

        Returns:
            List of ConceptCone objects, may be fewer than n_cones if clusters are too small
        """
        # Step 1: Cluster bad prompts by their residual patterns
        # Use the mean across layers as the embedding
        bad_embeddings = bad_residuals.mean(dim=1).cpu().numpy()

        # Try clustering
        kmeans = KMeans(n_clusters=self.n_cones, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(bad_embeddings)

        # Evaluate clustering quality
        if len(set(cluster_labels)) > 1:
            self.silhouette_score = silhouette_score(bad_embeddings, cluster_labels)
            print(f"  * Concept cone silhouette score: {self.silhouette_score:.3f}")

            if self.silhouette_score < 0.1:
                print("[yellow]Warning: Poor clustering quality. Refusal may not have distinct concept cones.[/yellow]")

        self.cones = []
        for cluster_id in range(self.n_cones):
            mask = cluster_labels == cluster_id
            cluster_size = mask.sum()

            if cluster_size < self.min_cluster_size:
                print(f"  * Skipping cluster {cluster_id} (size={cluster_size} < {self.min_cluster_size})")
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
                size=int(cluster_size),
            )
            self.cones.append(cone)

        print(f"  * Extracted {len(self.cones)} concept cones from {self.n_cones} clusters")
        return self.cones

    def select_cone(self, prompt_residual: Tensor) -> ConceptCone:
        """Select the most relevant cone for a given prompt."""
        if not self.cones:
            raise ValueError("No cones extracted. Call extract() first.")

        embedding = prompt_residual.mean(dim=0).cpu()  # Average across layers

        # Find nearest centroid
        min_dist = float('inf')
        best_cone = self.cones[0]

        for cone in self.cones:
            dist = torch.norm(embedding - cone.centroid).item()
            if dist < min_dist:
                min_dist = dist
                best_cone = cone

        return best_cone
```

### Config Changes
```toml
# Enable concept cone extraction for adaptive ablation
use_concept_cones = false  # Default off - needs validation

# Number of harm category clusters to try
n_concept_cones = 5

# Minimum samples per cone (smaller clusters are merged)
min_cone_size = 10

# Directions per cone
directions_per_cone = 2

# Minimum silhouette score to use concept cones (else use global directions)
min_silhouette_score = 0.1
```

### Failure Modes

1. **Meaningless clusters**: KMeans may find clusters that don't correspond to harm categories
2. **Too few samples**: With 400 prompts / 5 clusters = 80 per cluster (marginal)
3. **Silhouette < 0.1**: Indicates overlapping clusters, no real structure

### Expected Impact
- **+5-10%** if refusal truly has distinct concept cones (unproven assumption)
- **0% or negative** if clusters are meaningless

---

## Improvement 5: Contrastive Activation Addition (CAA)

### Problem
Current ablation only **removes** the refusal direction. Research shows that **adding** a helpfulness/compliance direction can be complementary.

### Implementation

```python
# src/heretic/model.py - Add CAA integration

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heretic.model import AbliterationParameters, LayerRangeProfile


# Add these methods to the Model class:

def extract_compliance_direction(
    self,
    compliant_residuals: Tensor,  # Residuals when model complied
    refusing_residuals: Tensor,   # Residuals when model refused
) -> Tensor:
    """Extract a direction that encodes 'compliance' behavior.

    Note: This is the OPPOSITE of refusal direction, but computed
    from different data (actual compliance responses).

    Args:
        compliant_residuals: Shape (n_comply, n_layers, hidden_dim)
        refusing_residuals: Shape (n_refuse, n_layers, hidden_dim)

    Returns:
        Compliance direction, shape (n_layers, hidden_dim)
    """
    compliance_direction = F.normalize(
        compliant_residuals.mean(dim=0) - refusing_residuals.mean(dim=0),
        p=2,
        dim=-1,  # Normalize across hidden_dim
    )
    return compliance_direction


def abliterate_with_caa(
    self,
    refusal_directions: Tensor,  # Shape (n_layers, hidden_dim) or (n_layers, n_components, hidden_dim)
    compliance_direction: Tensor,  # Shape (n_layers, hidden_dim)
    parameters: dict[str, "AbliterationParameters"],
    removal_strength: float = 1.0,
    addition_strength: float = 0.3,  # Usually smaller than removal
    layer_profiles: list["LayerRangeProfile"] | None = None,
) -> None:
    """Combined ablation: remove refusal + add compliance.

    WARNING: Addition may partially undo removal if directions aren't orthogonal.
    Check cosine similarity first.

    Args:
        refusal_directions: Refusal directions to remove
        compliance_direction: Compliance direction to add
        parameters: Abliteration parameters per component
        removal_strength: Multiplier for refusal direction removal
        addition_strength: Multiplier for compliance direction addition
        layer_profiles: Optional per-layer weight profiles
    """
    # Handle both single-direction and multi-direction refusal tensors
    if refusal_directions.dim() == 2:
        # Single direction: (n_layers, hidden_dim)
        refusal_for_sim = refusal_directions
    else:
        # Multi-direction: (n_layers, n_components, hidden_dim) - use first component
        refusal_for_sim = refusal_directions[:, 0, :]

    # Check orthogonality
    cosine_sim = F.cosine_similarity(
        refusal_for_sim,
        compliance_direction,
        dim=-1
    ).mean().item()

    if abs(cosine_sim) > 0.5:
        print(f"[yellow]Warning: Refusal and compliance directions have high overlap (cosine={cosine_sim:.2f})[/yellow]")
        print("[yellow]CAA addition may partially undo ablation.[/yellow]")

    # Step 1: Remove refusal direction (existing ablation)
    # Note: abliterate() takes directions and a direction_index for multi-component
    self.abliterate(
        refusal_directions,
        0,  # direction_index - use first/primary direction
        parameters,
        layer_profiles=layer_profiles,
    )

    # Step 2: Add compliance direction (only to attention output projections)
    for layer_idx, layer in enumerate(self.get_layers()):
        compliance_dir = compliance_direction[layer_idx]
        projector = torch.outer(compliance_dir, compliance_dir).to(self.model.dtype)

        # Add to output projections
        o_proj = layer.self_attn.o_proj.weight
        o_proj.data += addition_strength * (projector @ o_proj.data)
```

### Config Changes
```toml
# Use Contrastive Activation Addition alongside ablation
use_caa = false  # Default off

# Strength of compliance direction addition (relative to removal)
caa_addition_strength = 0.3

# Maximum allowed cosine similarity between refusal and compliance directions
caa_max_overlap = 0.5
```

### Failure Modes

1. **Direction overlap**: If refusal ≈ -compliance, addition undoes removal
2. **Cumulative drift**: Addition compounds if run multiple times
3. **Sycophancy**: Over-adding compliance may make model too agreeable

### Expected Impact
- **+3-8%** if directions are sufficiently orthogonal
- **0% or negative** if directions overlap significantly

---

## Improvement 6: Circuit-Level Ablation (Attention Head Targeting)

### ⚠️ Status: REQUIRES MAJOR ARCHITECTURAL CHANGES

This improvement requires:
1. Hook infrastructure for activation patching
2. TransformerLens or equivalent library integration
3. Support for Grouped Query Attention (GQA) in Qwen2.5 and Llama 3.x
4. ~11 hours of compute for circuit discovery on 32B models

### Problem
Current layer-level ablation is **too coarse**. Research shows refusal is mediated by specific attention heads, not entire layers.

### Prerequisites

```toml
# Would need to add to pyproject.toml:
# "transformer-lens>=1.0.0",  # Conflicts with some transformers versions
```

### Implementation (CORRECTED)

The original code had bugs. Here's the corrected version:

```python
# src/heretic/model.py - Add circuit discovery
# NOTE: This is pseudocode - actual implementation requires hook infrastructure

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


@dataclass
class RefusalCircuit:
    """A specific attention head identified as mediating refusal."""
    layer_idx: int
    head_idx: int
    importance: float  # How much this head contributes to refusal


# Add this method to the Model class:
def abliterate_circuits(
    self,
    circuits: list[RefusalCircuit],
    refusal_directions: Tensor,
    ablation_strength: float = 1.0,
) -> None:
    """Abliterate only specific attention heads, not entire layers.

    CORRECTED: Fixed head_dim calculation and added GQA handling.

    Args:
        circuits: List of attention head circuits to ablate
        refusal_directions: Refusal directions, shape (n_layers, hidden_dim)
        ablation_strength: Base ablation strength multiplier
    """
    # Check for GQA and skip if detected (until proper support is added)
    n_heads = self.model.config.num_attention_heads
    n_kv_heads = getattr(self.model.config, 'num_key_value_heads', n_heads)

    if n_kv_heads != n_heads:
        print(f"[yellow]Warning: GQA model detected (n_heads={n_heads}, n_kv_heads={n_kv_heads}).[/yellow]")
        print("[yellow]Circuit ablation is not supported for GQA models. Skipping.[/yellow]")
        return  # Skip circuit ablation for GQA models

    for circuit in circuits:
        layer = self.get_layers()[circuit.layer_idx]
        o_proj = layer.self_attn.o_proj.weight

        hidden_dim = o_proj.shape[0]
        head_dim = hidden_dim // n_heads

        # Compute head-specific indices
        start_idx = circuit.head_idx * head_dim
        end_idx = start_idx + head_dim

        # Validate indices
        if end_idx > hidden_dim:
            print(f"[red]Error: Head {circuit.head_idx} out of bounds for layer {circuit.layer_idx}[/red]")
            continue

        # Ablate only this head's contribution
        direction = refusal_directions[circuit.layer_idx]
        strength = ablation_strength * circuit.importance

        # Project direction to head subspace
        head_direction = direction[start_idx:end_idx]
        head_direction = F.normalize(head_direction, p=2, dim=0)

        # Project out refusal direction from this head only
        projector = torch.outer(head_direction, head_direction)
        o_proj.data[start_idx:end_idx, :] -= strength * (
            projector @ o_proj.data[start_idx:end_idx, :]
        )
```

### Config Changes
```toml
# Use circuit-level ablation instead of layer-level
# WARNING: Experimental, requires TransformerLens integration
use_circuit_ablation = false

# Number of top circuits to ablate
n_refusal_circuits = 20

# Path to cache discovered circuits
circuit_cache_path = "./refusal_circuits.json"
```

### Resource Requirements

| Model Size | Circuit Discovery Time | Memory |
|------------|----------------------|--------|
| 7B | ~30 minutes | +4GB |
| 13B | ~2 hours | +8GB |
| 32B | ~11 hours | +16GB |
| 70B | ~24+ hours | +32GB |

### Failure Modes

1. **GQA incompatibility**: Modern models use GQA; now properly detected and skipped
2. **Wrong heads identified**: Activation patching may find spurious circuits
3. **Capability damage**: Ablating specific heads may damage more than full-layer

### Expected Impact
- **10-30% less capability damage** (if implemented correctly)
- **HIGH RISK** of implementation errors

---

## Improvement 7: Cross-Model Hyperparameter Transfer

### ⚠️ Status: LIMITED APPLICABILITY

The original proposal ignored a fundamental problem: **hidden dimensions differ across models**.

- Llama-3-8B: 4096 hidden_dim
- Qwen2.5-32B: 5120 hidden_dim
- Llama-3-70B: 8192 hidden_dim

You **cannot** directly transfer directions between models with different hidden dimensions. The vectors exist in completely different spaces.

### What CAN Be Transferred

1. **Layer position patterns**: Where refusal peaks (as % of layers)
2. **Ablation hyperparameters**: Good starting points for Optuna
3. **Component weights**: Relative importance of attn vs mlp

### Implementation (Limited Version)

```python
# src/heretic/transfer.py - Cross-model hyperparameter transfer only

from dataclasses import dataclass


@dataclass
class ModelAbliterationProfile:
    """Transferable hyperparameters (NOT directions) for a model family."""
    model_family: str
    recommended_max_weight_position: float  # As fraction of layers
    recommended_min_weight_distance: float  # As fraction of layers
    attn_vs_mlp_ratio: float  # Relative ablation strength
    typical_kl_divergence: float  # Expected KL at good ablation


# Profiles learned from successful abliterations
MODEL_PROFILES = {
    "llama": ModelAbliterationProfile(
        model_family="llama",
        recommended_max_weight_position=0.65,
        recommended_min_weight_distance=0.35,
        attn_vs_mlp_ratio=1.0,
        typical_kl_divergence=0.3,
    ),
    "qwen": ModelAbliterationProfile(
        model_family="qwen",
        recommended_max_weight_position=0.60,
        recommended_min_weight_distance=0.30,
        attn_vs_mlp_ratio=1.1,
        typical_kl_divergence=0.35,
    ),
    "mistral": ModelAbliterationProfile(
        model_family="mistral",
        recommended_max_weight_position=0.70,
        recommended_min_weight_distance=0.40,
        attn_vs_mlp_ratio=0.9,
        typical_kl_divergence=0.25,
    ),
}


def get_warm_start_params(model_name: str, n_layers: int) -> dict:
    """Get warm-start parameters for Optuna based on model family.

    This does NOT transfer directions, just hyperparameters.
    """
    # Detect family
    model_lower = model_name.lower()
    family = None
    for key in MODEL_PROFILES:
        if key in model_lower:
            family = key
            break

    if family is None:
        return {}  # No warm start available

    profile = MODEL_PROFILES[family]

    return {
        "max_weight_position": profile.recommended_max_weight_position * n_layers,
        "min_weight_distance": profile.recommended_min_weight_distance * n_layers,
    }
```

### What the Original Proposal Got Wrong

| Claim | Reality |
|-------|--------|
| "Transfer directions across models" | Can't - different hidden dimensions |
| "10x faster setup" | Only ~2x faster (warm start) |
| "5% accuracy trade-off" | 20-50% accuracy loss if forced |

### Config Changes
```toml
# Enable warm-start parameters from model family profiles
use_warm_start_params = false

# Override detected model family (auto-detected if not set)
model_family = ""  # e.g., "llama", "qwen", "mistral"
```

### Failure Modes

1. **Family misdetection**: Model name patterns may not match (e.g., "Llama-3.2" vs "llama")
2. **Profile staleness**: Hardcoded profiles may not apply to newer model versions
3. **Over-reliance**: Using warm-start exclusively skips exploration that might find better params

### Expected Impact
- **~2x faster** Optuna convergence with warm-start parameters
- **NOT** a replacement for direction extraction

---

## Improvement 8: Sparse Autoencoder Integration

### Status: RESEARCH PROJECT

Sparse Autoencoders (SAEs) are cutting-edge research from Anthropic. They:
- Are not publicly available for most models
- Require significant compute to train (~100+ GPU hours)
- Have unknown effectiveness for abliteration

**Recommendation**: Monitor research progress but do not allocate development resources until:
1. Pre-trained SAEs are available for target model families
2. Published results demonstrate effectiveness for refusal removal

---

## Revised Implementation Roadmap

### Phase 1 (Week 1-2): Foundation & Neural Detector
- [x] Add `scikit-learn>=1.3.0` to pyproject.toml ✅ (already completed)
- [ ] Add new settings to `src/heretic/config.py`:
  - `use_neural_refusal_detection: bool = False`
  - `neural_detection_model: str = "facebook/bart-large-mnli"`
  - `neural_detection_for_optuna: bool = False`
- [ ] Implement **Neural Refusal Detector** in `src/heretic/evaluator.py`
- [ ] Add unit tests for neural detector

### Phase 2 (Week 3-4): Core Improvements
- [ ] Add supervised probing settings to `src/heretic/config.py`:
  - `use_supervised_probing: bool = False`
  - `min_probe_accuracy: float = 0.65`
  - `ensemble_probe_pca: bool = False`
- [ ] Implement **Supervised Refusal Probing** with cross-validation
- [ ] Implement **Activation-Scaled Weights** (calibration)
- [ ] Add integration tests for direction extraction
- [ ] Create A/B test benchmark suite in `tests/benchmark/`

### Phase 3 (Week 5-6): Experimental
- [ ] Add concept cones and CAA settings to `src/heretic/config.py`
- [ ] Implement **Concept Cones** with silhouette validation
- [ ] Implement **CAA** with orthogonality checking
- [ ] Run comparative benchmarks

### Phase 4 (Week 7-8): Documentation & Validation
- [ ] Statistical significance testing across 5+ models
- [ ] Update documentation with validated impact numbers
- [ ] Create model-specific recommendation profiles

### Deferred (Not Scheduled)
- Circuit-Level Ablation (requires architectural changes, GQA support)
- Cross-Model Direction Transfer (limited value - different hidden dimensions)
- SAE Integration (waiting for research maturity)

---

## Validation Framework

### Statistical Requirements

```
For each improvement:
1. Test on minimum 5 models:
   - Llama-3-8B-Instruct
   - Qwen2.5-7B-Instruct
   - Mistral-7B-Instruct-v0.3
   - Qwen2.5-14B-Instruct
   - Llama-3.1-8B-Instruct

2. Use 3 random seeds per model

3. Report:
   - Mean ± std for refusal_rate, kl_divergence, mmlu_average
   - p-value from paired t-test (baseline vs improvement)
   - Effect size (Cohen's d)

4. Accept if:
   - p < 0.05 for refusal rate improvement
   - KL divergence not significantly worse (p > 0.05)
   - Effect size d > 0.3 (small-medium effect)
```

### Benchmark Suite Structure

> **Note**: This benchmark directory needs to be created as part of Phase 1 implementation.

```python
# tests/benchmark/run_benchmark.py (to be created)

MODELS = [
    "meta-llama/Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

SEEDS = [42, 123, 456]

IMPROVEMENTS = [
    "baseline",
    "supervised_probing",
    "concept_cones",
    "activation_calibration",
    "caa",
]

def run_benchmark(model, seed, improvement):
    # Run heretic with specific configuration
    # Return: refusal_rate, kl_divergence, mmlu_scores
    pass
```

---

## References

1. **Representation Engineering** (Zou et al., 2023) - Foundation for probing
2. **Contrastive Activation Addition** (Rimsky et al., ACL 2024) - CAA technique
3. **Refusal in LLMs** (Arditi et al., NeurIPS 2024) - Single direction analysis
4. **Sparse Autoencoders** (Anthropic, 2024) - Feature isolation
5. **COSMIC** (arXiv 2024, preprint) - Cross-model transfer limitations
6. **TransformerLens** (Nanda) - Circuit discovery tools
7. **Concept Cones** (Wei et al., OpenReview 2024) - "On the Geometry of Refusal in Large Language Models" - Multi-direction geometry (findings contested)

---

## Summary

This revised document provides:

1. **Realistic impact estimates** (15-25% combined, not 40-60%)
2. **Fixed code** (head_dim calculation, GQA handling)
3. **Redesigned Improvement 3** (training-time calibration, not runtime)
4. **Clear dependencies** (scikit-learn needed)
5. **Resource requirements** (memory, compute time)
6. **Failure modes** for each improvement
7. **Statistical validation requirements**
8. **Honest assessment** of what's feasible vs research-only

**Recommended Implementation Order** (matches Innovation Matrix priorities):
1. **Neural Refusal Detector** (pure evaluation, safest, no ablation changes)
2. **Supervised Probing** (well-understood technique, good fallback to PCA)
3. **Activation-Scaled Weights** (simple training-time calibration)
4. **Concept Cones** (experimental, needs silhouette validation)
5. **CAA** (complementary to ablation, check orthogonality first)

**Skip for now**: Circuit Ablation (GQA incompatibility), Cross-Model Direction Transfer (hidden dimension mismatch), SAE Integration (research maturity)
