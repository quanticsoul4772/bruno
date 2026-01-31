"""Custom exception hierarchy for heretic.

This module defines a comprehensive exception hierarchy for all error types
that can occur during neural behavior modification operations.
"""


class HereticError(Exception):
    """Base exception for all heretic errors.

    All custom exceptions in the heretic package inherit from this base class.
    This allows catching all heretic-specific errors with a single except clause
    while letting system exceptions (KeyboardInterrupt, SystemExit) propagate.
    """

    pass


# ============================================================================
# Model Errors (3)
# ============================================================================


class ModelError(HereticError):
    """Base exception for model-related errors.

    Covers errors during model loading, initialization, and inference.
    """

    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails.

    Common causes:
    - Model not found on HuggingFace
    - Authentication required for gated models
    - Insufficient disk space
    - Unsupported model architecture
    - Network timeout during download
    """

    pass


class ModelInferenceError(ModelError):
    """Exception raised during model inference.

    Common causes:
    - GPU out of memory
    - Invalid input shape
    - Tokenization errors
    - Model forward pass failures
    """

    pass


# ============================================================================
# Dataset Errors (2)
# ============================================================================


class DatasetError(HereticError):
    """Base exception for dataset-related errors.

    Covers errors during dataset loading, processing, and validation.
    """

    pass


class DatasetConfigError(DatasetError):
    """Exception raised when dataset configuration is invalid.

    Common causes:
    - Missing required config parameter (e.g., C4 needs 'en')
    - Invalid split format (e.g., 'train[:abc]')
    - Column not found in dataset
    - Dataset variant not available
    """

    pass


# ============================================================================
# Network Errors (2)
# ============================================================================


class NetworkError(HereticError):
    """Base exception for network-related errors.

    Covers errors during remote operations and API calls.
    """

    pass


class NetworkTimeoutError(NetworkError):
    """Exception raised when network operation times out.

    Common causes:
    - HuggingFace Hub unreachable
    - Slow internet connection
    - Large file download interrupted
    - API rate limiting
    """

    pass


# ============================================================================
# File I/O Errors (2)
# ============================================================================


class FileOperationError(HereticError):
    """Base exception for file operation errors.

    Covers errors during reading, writing, and file system operations.
    """

    pass


class ValidationFileError(FileOperationError):
    """Exception raised when validation file operations fail.

    Common causes:
    - Corrupted JSON validation report
    - Missing required fields in validation data
    - Permission denied when reading/writing
    - Insufficient disk space
    """

    pass


# ============================================================================
# Cloud/SSH Errors (2)
# ============================================================================


class CloudError(HereticError):
    """Base exception for cloud provider errors.

    Covers errors during cloud instance operations (Vast.ai, etc.).
    """

    pass


class SSHError(CloudError):
    """Exception raised during SSH operations.

    Common causes:
    - SSH connection timeout
    - Authentication failure
    - Command execution timeout
    - Network interruption during remote operation
    """

    pass


# ============================================================================
# Configuration Errors (2)
# ============================================================================


class ConfigurationError(HereticError):
    """Base exception for configuration-related errors.

    Covers errors in settings, parameters, and user input validation.
    """

    pass


class PhaseConfigError(ConfigurationError):
    """Exception raised when Phase 1-7 configuration is invalid.

    Common causes:
    - Invalid parameter combination (e.g., CAA + circuit ablation)
    - Missing required parameters for enabled phase
    - Parameter value out of valid range
    - Incompatible model architecture for phase
    """

    pass


class WarmStartError(ConfigurationError):
    """Exception raised when warm-start parameter loading fails.

    Common causes:
    - No profile found for model family
    - Unknown model family
    - Invalid model name format
    """

    pass


# ============================================================================
# Abliteration Errors (2)
# ============================================================================


class AbliterationError(HereticError):
    """Base exception for abliteration operation errors.

    Covers errors during direction extraction and weight modification.
    """

    pass


class CircuitAblationError(AbliterationError):
    """Exception raised during circuit-level ablation.

    Common causes:
    - Model uses GQA (Grouped Query Attention) - not supported
    - No refusal circuits discovered
    - Attention head ablation failed
    - Incompatible model architecture
    """

    pass


# ============================================================================
# Resource Errors (2)
# ============================================================================


class ResourceError(HereticError):
    """Base exception for resource-related errors.

    Covers errors related to GPU memory, batch sizing, and system resources.
    """

    pass


class BatchSizeError(ResourceError):
    """Exception raised when batch size causes resource exhaustion.

    Common causes:
    - Batch size too large for available GPU memory
    - Repeated OOM errors during optimization
    - Model too large for available VRAM
    """

    pass


# ============================================================================
# Extraction Errors (2)
# ============================================================================


class ExtractionError(AbliterationError):
    """Base exception for direction extraction errors.

    Covers errors during refusal direction extraction methods.
    """

    pass


class SupervisedProbeError(ExtractionError):
    """Exception raised when supervised probing fails.

    Common causes:
    - Class imbalance (< 10 samples per class)
    - Probe accuracy below threshold
    - Model doesn't distinguish refusal vs compliance patterns
    """

    pass


class ConceptConeError(ExtractionError):
    """Exception raised when concept cone extraction fails.

    Common causes:
    - Silhouette score below threshold (poor clustering quality)
    - Harmful prompts don't have distinct categories
    - Insufficient samples for clustering
    """

    pass


class CAAExtractionError(ExtractionError):
    """Exception raised when CAA extraction fails.

    Common causes:
    - Insufficient refusal samples (< 10)
    - Insufficient compliance samples (< 10)
    - Model already complies or refuses everything
    """


class SacredDirectionError(ExtractionError):
    """Exception raised during sacred direction extraction or orthogonalization.

    Common causes:
    - Sacred direction extraction failed
    - Orthogonalization resulted in near-zero direction
    - Refusal direction lies within sacred direction span
    - Sacred prompts dataset could not be loaded
    """

    pass


# ============================================================================
# Exception Hierarchy Reference
# ============================================================================
"""
HereticError (base)
├── ModelError
│   ├── ModelLoadError
│   └── ModelInferenceError
├── DatasetError
│   └── DatasetConfigError
├── NetworkError
│   └── NetworkTimeoutError
├── FileOperationError
│   └── ValidationFileError
├── CloudError
│   └── SSHError
├── ConfigurationError
│   ├── PhaseConfigError
│   └── WarmStartError
├── ResourceError
│   └── BatchSizeError
└── AbliterationError
    ├── CircuitAblationError
    └── ExtractionError
        ├── SupervisedProbeError
        ├── ConceptConeError
        └── CAAExtractionError

Total: 22 custom exceptions (1 base + 21 specific)
"""
