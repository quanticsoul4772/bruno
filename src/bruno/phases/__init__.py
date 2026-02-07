# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Phase modules for the heretic abliteration pipeline.

This package contains modular components that break down the abliteration
workflow into discrete, testable phases:

- dataset_loading: Load and prepare prompt datasets
- direction_extraction: Extract refusal and sacred directions
- optimization: Run Optuna optimization loop
- model_saving: Save and upload abliterated models
- gguf_conversion: Convert models to GGUF and create quantizations

Usage:
    from bruno.phases import DatasetBundle, DirectionExtractionResult
    from bruno.phases.dataset_loading import load_datasets
    from bruno.phases.direction_extraction import extract_directions
    from bruno.phases.gguf_conversion import convert_and_quantize
"""

from .dataset_loading import DatasetBundle, load_datasets
from .direction_extraction import (
    DirectionExtractionResult,
    apply_helpfulness_orthogonalization,
    extract_refusal_directions,
    extract_sacred_directions,
)
from .gguf_conversion import convert_and_quantize, upload_gguf_files
from .model_saving import save_model_local, upload_model_huggingface
from .optimization import create_study, enqueue_warm_start_trials, run_optimization

__all__ = [
    # Dataset loading
    "DatasetBundle",
    "load_datasets",
    # Direction extraction
    "DirectionExtractionResult",
    "apply_helpfulness_orthogonalization",
    "extract_refusal_directions",
    "extract_sacred_directions",
    # Optimization
    "create_study",
    "enqueue_warm_start_trials",
    "run_optimization",
    # Model saving
    "save_model_local",
    "upload_model_huggingface",
    # GGUF conversion
    "convert_and_quantize",
    "upload_gguf_files",
]
