# Heretic Architecture

This document describes the modular architecture of heretic's abliteration pipeline.

## Overview

Heretic v1.2.0 refactored the monolithic `run()` function into discrete, testable phase modules. This improves:
- **Testability**: Each phase can be unit tested independently
- **Maintainability**: Changes to one phase don't affect others
- **Extensibility**: New phases can be added without modifying existing code
- **Reusability**: Phase functions can be used in custom pipelines

---

## Phase Module Structure

```
src/heretic/phases/
├── __init__.py              # Exports all phase functions and dataclasses
├── dataset_loading.py       # Phase 1: Dataset loading
├── direction_extraction.py  # Phase 2: Direction extraction
├── optimization.py          # Phase 3: Optuna optimization
└── model_saving.py          # Phase 4: Model persistence
```

---

## Phase 1: Dataset Loading

**Module**: `phases/dataset_loading.py`

**Purpose**: Load all prompt datasets needed for abliteration.

**Exports**:
- `DatasetBundle` - Dataclass containing all loaded datasets
- `load_datasets(settings)` - Load datasets based on settings

**DatasetBundle Fields**:
```python
@dataclass
class DatasetBundle:
    good_prompts: list[str]              # Harmless prompts
    bad_prompts: list[str]               # Harmful prompts (trigger refusals)
    helpful_prompts: list[str] | None    # For helpfulness orthogonalization
    unhelpful_prompts: list[str] | None  # For helpfulness contrast
    sacred_prompts: list[str] | None     # For capability preservation (MMLU)
    sacred_baseline_prompts: list[str] | None  # For sacred direction contrast
```

**Usage**:
```python
from heretic.phases import load_datasets, DatasetBundle

datasets = load_datasets(settings)
print(f"Loaded {len(datasets.good_prompts)} good prompts")
print(f"Loaded {len(datasets.bad_prompts)} bad prompts")
```

---

## Phase 2: Direction Extraction

**Module**: `phases/direction_extraction.py`

**Purpose**: Extract refusal directions and apply orthogonalizations.

**Exports**:
- `DirectionExtractionResult` - Dataclass with extracted directions
- `extract_refusal_directions(model, good_prompts, bad_prompts, settings)` - Main extraction
- `apply_helpfulness_orthogonalization(model, result, helpful_prompts, unhelpful_prompts)` - Helpfulness orthogonalization
- `extract_sacred_directions(model, result, sacred_prompts, sacred_baseline_prompts, settings)` - Sacred direction preservation

**DirectionExtractionResult Fields**:
```python
@dataclass
class DirectionExtractionResult:
    refusal_directions: Tensor           # Shape: (n_layers, hidden_dim)
    use_multi_direction: bool            # Whether to use multi-direction ablation
    refusal_directions_multi: Tensor | None  # Shape: (n_layers, n_components, hidden_dim)
    direction_weights: list[float] | None    # Weights for each direction component
    helpfulness_direction: Tensor | None     # Helpfulness direction (for reference)
    sacred_directions_result: SacredDirectionResult | None  # Sacred direction result
```

**Usage**:
```python
from heretic.phases import extract_refusal_directions, apply_helpfulness_orthogonalization

# Extract refusal directions
good_residuals, bad_residuals, result = extract_refusal_directions(
    model=model,
    good_prompts=datasets.good_prompts,
    bad_prompts=datasets.bad_prompts,
    settings=settings,
)

# Apply helpfulness orthogonalization
if datasets.helpful_prompts:
    result = apply_helpfulness_orthogonalization(
        model=model,
        result=result,
        helpful_prompts=datasets.helpful_prompts,
        unhelpful_prompts=datasets.unhelpful_prompts,
    )
```

---

## Phase 3: Optimization

**Module**: `phases/optimization.py`

**Purpose**: Create Optuna studies and run optimization loops.

**Exports**:
- `create_study(settings)` - Create or load Optuna study
- `enqueue_warm_start_trials(study, settings, model)` - Enqueue warm-start trials
- `run_optimization(study, objective, n_trials)` - Run optimization loop

**Usage**:
```python
from heretic.phases import create_study, enqueue_warm_start_trials, run_optimization

# Create study with resume support
study = create_study(settings)

# Enqueue warm-start trials (returns True if enqueued)
warm_start_enqueued = enqueue_warm_start_trials(study, settings, model)

# Run optimization
run_optimization(study, objective_function, settings.n_trials)
```

**Error Handling**:
- Raises `ConfigurationError` for database lock, disk space, or corruption issues
- Provides helpful error messages with actionable solutions

---

## Phase 4: Model Saving

**Module**: `phases/model_saving.py`

**Purpose**: Save abliterated models locally or to HuggingFace Hub.

**Exports**:
- `save_model_local(model, save_directory)` - Save to local disk
- `upload_model_huggingface(model, settings, trial, evaluator, repo_id, private, token)` - Upload to HF Hub

**Usage**:
```python
from heretic.phases import save_model_local, upload_model_huggingface

# Save locally
save_model_local(model, "./models/my-abliterated-model")

# Upload to HuggingFace
upload_model_huggingface(
    model=model,
    settings=settings,
    trial=best_trial,
    evaluator=evaluator,
    repo_id="username/model-heretic",
)
```

---

## Constants Module

**Module**: `src/heretic/constants.py`

**Purpose**: Centralize magic numbers and thresholds.

**Dataclasses**:
- `Thresholds` - Numerical thresholds (NEAR_ZERO, EPSILON, etc.)
- `LayerPos` - Layer position fractions for direction extraction
- `Ablation` - Ablation parameter defaults
- `Memory` - Memory management constants
- `Validation` - Validation thresholds

**Usage**:
```python
from heretic.constants import Thresholds, LayerPos, Memory

if magnitude < Thresholds.NEAR_ZERO:
    raise SacredDirectionError("Direction too small")

start_layer = int(LayerPos.MIDDLE_START * n_layers)
end_layer = int(LayerPos.MIDDLE_END * n_layers)
```

---

## Exception Hierarchy

**Module**: `src/heretic/exceptions.py`

Heretic uses a structured exception hierarchy for clear error handling:

```
HereticError (base)
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   ├── ModelInferenceError
│   └── AbliterationError
├── DatasetError
│   ├── DatasetLoadError
│   └── DatasetValidationError
├── ConfigurationError
│   └── BatchSizeError
├── SacredDirectionError      # NEW: Sacred direction extraction failures
├── SupervisedProbeError
├── ConceptConeError
├── CAAExtractionError
├── WarmStartError
└── ... (22+ exception types)
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     HERETIC PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐                                           │
│   │ 1. DATASET      │  load_datasets(settings)                  │
│   │    LOADING      │  → DatasetBundle                          │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │ 2. DIRECTION    │  extract_refusal_directions()             │
│   │    EXTRACTION   │  apply_helpfulness_orthogonalization()    │
│   │                 │  extract_sacred_directions()              │
│   │                 │  → DirectionExtractionResult              │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │ 3. OPTIMIZATION │  create_study()                           │
│   │                 │  enqueue_warm_start_trials()              │
│   │                 │  run_optimization()                       │
│   │                 │  → Pareto-optimal trials                  │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │ 4. MODEL        │  save_model_local()                       │
│   │    SAVING       │  upload_model_huggingface()               │
│   └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing

Phase modules have dedicated unit tests:

```bash
# Test phase modules
uv run pytest tests/unit/test_phases.py -v

# Test sacred directions
uv run pytest tests/unit/test_sacred_directions.py -v

# Run all unit tests
uv run pytest tests/unit/ -v
```

---

## Extending the Pipeline

To add a new phase:

1. Create `src/heretic/phases/your_phase.py`
2. Define dataclasses for inputs/outputs
3. Implement phase functions
4. Export from `src/heretic/phases/__init__.py`
5. Add unit tests in `tests/unit/test_phases.py`
6. Integrate in `src/heretic/main.py`

---

## Configuration

All phase behavior is controlled via `Settings` in `config.py`:

- **Dataset settings**: `good_prompts`, `bad_prompts`, `sacred_prompts`, etc.
- **Direction settings**: `use_pca_extraction`, `n_refusal_directions`, `use_sacred_directions`
- **Optimization settings**: `n_trials`, `use_warm_start_params`, `storage`
- **Saving settings**: `auto_select`, `auto_select_path`, `hf_upload`

Configuration priority: CLI args > env vars > config.toml > defaults
