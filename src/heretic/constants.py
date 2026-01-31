# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Centralized constants for heretic.

This module contains all magic numbers and thresholds used throughout the codebase.
Centralizing these values provides:
- Single source of truth for numerical constants
- Easy tuning and experimentation
- Self-documenting code with meaningful names
- Reduced risk of inconsistent values

Usage:
    from heretic.constants import Thresholds, LayerPositions, Defaults
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalThresholds:
    """Small numerical thresholds for stability and validation."""

    # Near-zero detection threshold for directions
    NEAR_ZERO: float = 1e-6

    # Epsilon for numerical stability in normalization
    EPSILON: float = 1e-8

    # Tolerance for floating-point comparisons
    FLOAT_TOLERANCE: float = 1e-4

    # Minimum magnitude warning threshold after orthogonalization
    LOW_MAGNITUDE_WARNING: float = 0.5

    # Minimum probe accuracy for cross-validation
    MIN_PROBE_CV_ACCURACY: float = 0.5


@dataclass(frozen=True)
class LayerPositions:
    """Layer position fractions (0.0-1.0) for direction extraction and ablation.

    Research shows different layer ranges encode different aspects of refusal:
    - Early layers (0.0-0.4): Basic representations, light abliteration preserves capabilities
    - Middle layers (0.4-0.7): Semantic "what to refuse", primary abliteration target
    - Late layers (0.7-1.0): Behavioral "how to refuse", moderate abliteration
    """

    # Early layer range
    EARLY_START: float = 0.0
    EARLY_END: float = 0.4

    # Middle layer range (primary abliteration target)
    MIDDLE_START: float = 0.4
    MIDDLE_END: float = 0.7

    # Late layer range
    LATE_START: float = 0.7
    LATE_END: float = 1.0

    # Default layer profile multipliers
    EARLY_WEIGHT_MULTIPLIER: float = 0.5
    MIDDLE_WEIGHT_MULTIPLIER: float = 1.0
    LATE_WEIGHT_MULTIPLIER: float = 0.8

    # Direction extraction ranges
    DIRECTION_INDEX_MIN_FRAC: float = 0.4  # Minimum layer fraction for global direction
    DIRECTION_INDEX_MAX_FRAC: float = 0.9  # Maximum layer fraction for global direction

    # Ablation parameter ranges
    MAX_WEIGHT_POSITION_MIN_FRAC: float = 0.6  # Minimum position for max weight
    MIN_WEIGHT_DISTANCE_MIN_FRAC: float = 0.6  # Minimum distance for weight falloff

    # Activation calibration default layer
    ACTIVATION_CALIBRATION_LAYER_FRAC: float = 0.6


@dataclass(frozen=True)
class AblationDefaults:
    """Default values for ablation parameters."""

    # Weight range for ablation
    MAX_WEIGHT_MIN: float = 0.8
    MAX_WEIGHT_MAX: float = 1.5
    MIN_WEIGHT_FRAC_MIN: float = 0.0
    MIN_WEIGHT_FRAC_MAX: float = 1.0

    # Calibration factors
    CALIBRATION_MIN_FACTOR: float = 0.5
    CALIBRATION_MAX_FACTOR: float = 2.0

    # Sacred direction PCA alpha
    SACRED_PCA_ALPHA: float = 0.5

    # CAA defaults
    CAA_ADDITION_STRENGTH: float = 0.3
    CAA_MAX_OVERLAP: float = 0.5

    # Iterative ablation
    MIN_DIRECTION_MAGNITUDE: float = 0.1
    MAX_KL_PER_ROUND: float = 0.5


@dataclass(frozen=True)
class ProbeDefaults:
    """Default values for supervised probing."""

    # Minimum samples per class for probing
    MIN_SAMPLES_PER_CLASS: int = 10

    # Minimum cross-validation splits
    MIN_CV_SPLITS: int = 2
    MAX_CV_SPLITS: int = 5

    # Logistic regression parameters
    LOGISTIC_C: float = 1.0
    LOGISTIC_MAX_ITER: int = 1000

    # Ensemble weights
    ENSEMBLE_PROBE_WEIGHT: float = 0.7
    ENSEMBLE_PCA_WEIGHT: float = 0.3

    # Minimum probe accuracy threshold
    MIN_PROBE_ACCURACY: float = 0.65


@dataclass(frozen=True)
class ClusteringDefaults:
    """Default values for concept cone clustering."""

    # KMeans parameters
    N_INIT: int = 10
    RANDOM_STATE: int = 42

    # Minimum silhouette score for valid clustering
    MIN_SILHOUETTE_SCORE: float = 0.1

    # Minimum cluster size
    MIN_CLUSTER_SIZE: int = 10

    # Default number of concept cones
    N_CONCEPT_CONES: int = 5
    DIRECTIONS_PER_CONE: int = 2


@dataclass(frozen=True)
class CircuitDefaults:
    """Default values for circuit-level ablation."""

    # Number of refusal circuits to discover
    N_REFUSAL_CIRCUITS: int = 20

    # Minimum importance score for circuits
    IMPORTANCE_THRESHOLD: float = 0.1

    # Default ablation strength
    CIRCUIT_ABLATION_STRENGTH: float = 1.0


@dataclass(frozen=True)
class MemoryDefaults:
    """Default values for memory management."""

    # Clear cache every N layers during ablation
    CACHE_CLEAR_INTERVAL: int = 8

    # Maximum OOM retries before giving up
    MAX_OOM_RETRIES: int = 3

    # TorchDynamo cache size limit
    DYNAMO_CACHE_SIZE_LIMIT: int = 64


@dataclass(frozen=True)
class ValidationDefaults:
    """Default values for validation."""

    # MMLU parameters
    MMLU_SAMPLES_PER_CATEGORY: int = 20
    MMLU_FEW_SHOT: int = 3

    # KL divergence thresholds
    MAX_ACCEPTABLE_KL: float = 1.0
    MAX_ACCEPTABLE_MMLU_DROP: float = 0.05

    # Refusal detection token limit
    REFUSAL_CHECK_TOKENS: int = 30

    # Neural detection threshold
    NEURAL_DETECTION_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class DirectionWeights:
    """Default direction weights for multi-direction ablation."""

    # Fixed weights for PCA components (when eigenvalue weights disabled)
    DEFAULT_WEIGHTS: tuple[float, ...] = (1.0, 0.5, 0.25)

    # Eigenvalue weight temperature for softmax
    EIGENVALUE_TEMPERATURE: float = 1.0


@dataclass(frozen=True)
class RetryDefaults:
    """Default values for retry logic."""

    # Maximum delay between retries (seconds)
    MAX_RETRY_DELAY: float = 60.0

    # Base delay for exponential backoff
    BASE_RETRY_DELAY: float = 1.0

    # SSH connection timeout (seconds)
    SSH_CONNECT_TIMEOUT: int = 30

    # Command timeout (seconds)
    COMMAND_TIMEOUT: int = 30


# Singleton instances for easy import
Thresholds = NumericalThresholds()
LayerPos = LayerPositions()
Ablation = AblationDefaults()
Probe = ProbeDefaults()
Clustering = ClusteringDefaults()
Circuit = CircuitDefaults()
Memory = MemoryDefaults()
Validation = ValidationDefaults()
Weights = DirectionWeights()
Retry = RetryDefaults()


# For backward compatibility and direct imports
NEAR_ZERO = Thresholds.NEAR_ZERO
EPSILON = Thresholds.EPSILON
LOW_MAGNITUDE_WARNING = Thresholds.LOW_MAGNITUDE_WARNING
SACRED_PCA_ALPHA = Ablation.SACRED_PCA_ALPHA
MAX_OOM_RETRIES = Memory.MAX_OOM_RETRIES
CACHE_CLEAR_INTERVAL = Memory.CACHE_CLEAR_INTERVAL
