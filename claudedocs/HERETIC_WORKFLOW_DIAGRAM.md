# Heretic Abliteration Workflow - Detailed Diagram

**Version:** 1.2.0+
**Date:** 2026-01-31

---

## Overview

Heretic performs neural behavior modification through activation direction analysis and Optuna-based optimization. This document provides detailed diagrams of the complete workflow.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HERETIC PIPELINE                            │
│                                                                     │
│  Input: Model + Prompts → Output: Modified Model (Abliterated)     │
└─────────────────────────────────────────────────────────────────────┘

                               │
                               ▼
        ┌──────────────────────────────────────────┐
        │  Phase 1: DATASET LOADING                │
        │  - Load good/bad prompts                 │
        │  - Load unhelpfulness prompts (C4)       │
        │  - Optional: Sacred capability prompts   │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  Phase 2: DIRECTION EXTRACTION           │
        │  - Extract refusal directions (PCA)      │
        │  - Optional: Supervised probing          │
        │  - Optional: Sacred directions           │
        │  - Orthogonalization                     │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  Phase 3: OPTIMIZATION (Optuna Loop)     │
        │  - 200 trials (default)                  │
        │  - Multi-objective: KL div + refusals    │
        │  - Layer-wise parameter tuning           │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  Phase 4: MODEL SAVING                   │
        │  - Save best model locally               │
        │  - Optional: Upload to HuggingFace       │
        └──────────────────────────────────────────┘
```

---

## Detailed Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HERETIC COMPONENTS                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Settings   │      │     Model    │      │  Evaluator   │
│              │      │              │      │              │
│ - Config     │─────▶│ - Loading    │◀────▶│ - KL div     │
│ - CLI args   │      │ - Caching    │      │ - Refusals   │
│ - TOML       │      │ - Abliterate │      │ - Scoring    │
└──────────────┘      └──────┬───────┘      └──────────────┘
                             │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DatasetBundle│    │  Directions  │    │   Optuna     │
│              │    │              │    │              │
│ - Good       │───▶│ - Refusal    │───▶│ - TPE        │
│ - Bad        │    │ - Helpfulness│    │ - Trials     │
│ - Unhelpful  │    │ - Sacred     │    │ - Pruning    │
│ - Sacred     │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Phase 1: Dataset Loading

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATASET LOADING PHASE                          │
└─────────────────────────────────────────────────────────────────────┘

Input: Dataset specifications from Settings
Output: DatasetBundle with all loaded prompts

┌─────────────────────────────────────────────────────────────────────┐
│  DatasetSpecification                                               │
│  ├─ dataset: "cais/harmless_base"                                   │
│  ├─ split: "train[:200]"                                            │
│  ├─ column: "chosen"                                                │
│  └─ config: Optional["en"]  (for C4)                                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  load_datasets() - phases/dataset_loading.py                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Load good_prompts (harmless/helpful responses)            │ │
│  │    ✓ Example: "Please help me with my homework"             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Load bad_prompts (harmful requests)                       │ │
│  │    ✓ Example: "How do I build a weapon?"                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 3. Load unhelpfulness_prompts (C4 dataset)                   │ │
│  │    ✓ Streaming mode (v1.1.0+): No disk overhead             │ │
│  │    ✓ Config required: "en"                                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 4. Optional: Load sacred_prompts (MMLU, capability tests)    │ │
│  │    ✓ Only if use_sacred_directions=true                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DatasetBundle                                                      │
│  ├─ good_prompts: list[str]          (200 prompts)                 │
│  ├─ bad_prompts: list[str]           (200 prompts)                 │
│  ├─ unhelpfulness_prompts: list[str] (200 prompts)                 │
│  └─ sacred_prompts: Optional[list[str]]                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Direction Extraction

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DIRECTION EXTRACTION PHASE                        │
└─────────────────────────────────────────────────────────────────────┘

Input: Model + DatasetBundle
Output: DirectionExtractionResult

┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Extract Refusal Directions                                │
│                                                                     │
│  extract_refusal_directions(model, bad_prompts, good_prompts)      │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Get hidden states (residuals) for bad prompts             │ │
│  │    bad_residuals = model.get_residuals_batched(bad_prompts)  │ │
│  │    Shape: (n_prompts, n_layers, hidden_dim)                  │ │
│  │    Example: (200, 64, 5120) for 32B model                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Get hidden states for good prompts                        │ │
│  │    good_residuals = model.get_residuals_batched(good_prompts)│ │
│  │    Shape: (n_prompts, n_layers, hidden_dim)                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 3. Contrastive PCA per layer                                 │ │
│  │    For each layer:                                            │ │
│  │      mean_bad = bad_residuals[:, layer, :].mean(axis=0)      │ │
│  │      mean_good = good_residuals[:, layer, :].mean(axis=0)    │ │
│  │      diff = bad_residuals - good_residuals                   │ │
│  │      cov = diff.T @ diff                                     │ │
│  │      eigvals, eigvecs = torch.linalg.eigh(cov)  (GPU!)       │ │
│  │      direction = eigvecs[:, -1]  (top eigenvector)           │ │
│  │                                                               │ │
│  │    GPU Optimization (v1.0.1+):                               │ │
│  │      ✓ 15-20x faster than CPU                                │ │
│  │      ✓ Completes in <5 min for 32B models                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 4. Optional: Supervised Probing (Phase 2)                    │ │
│  │    if use_supervised_probing:                                │ │
│  │      Train linear classifiers per layer                      │ │
│  │      Ensemble with PCA directions                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Helpfulness Orthogonalization                             │
│                                                                     │
│  apply_helpfulness_orthogonalization(refusal_dirs, unhelpful_dirs) │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Extract unhelpfulness direction from C4 prompts           │ │
│  │    unhelpful_residuals = model.get_residuals(C4_prompts)     │ │
│  │    unhelpful_dir = PCA(unhelpful_residuals)                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Orthogonalize refusal directions                          │ │
│  │    For each layer:                                            │ │
│  │      overlap = refusal_dir @ unhelpful_dir                   │ │
│  │      refusal_dir -= overlap * unhelpful_dir                  │ │
│  │      refusal_dir /= norm(refusal_dir)  (renormalize)         │ │
│  │                                                               │ │
│  │    Result: Refusal direction without padding/verbosity       │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Optional Sacred Direction Extraction                      │
│                                                                     │
│  if use_sacred_directions:                                         │
│    extract_sacred_directions(model, sacred_prompts)                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Extract capability-encoding directions                    │ │
│  │    sacred_residuals = model.get_residuals(MMLU_prompts)      │ │
│  │    baseline_residuals = model.get_residuals(wikitext)        │ │
│  │    sacred_dirs = contrastive_PCA(sacred - baseline)          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Orthogonalize refusal against sacred                      │ │
│  │    For each sacred_dir:                                       │ │
│  │      overlap = refusal_dir @ sacred_dir                      │ │
│  │      refusal_dir -= overlap * sacred_dir                     │ │
│  │                                                               │ │
│  │    Result: Refusal removal without capability damage         │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DirectionExtractionResult                                          │
│  ├─ refusal_directions: Tensor[n_layers, hidden_dim]               │
│  ├─ unhelpfulness_directions: Optional[Tensor]                     │
│  └─ sacred_directions: Optional[Tensor[n_sacred, n_layers, dim]]   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Optimization Loop (Optuna)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION PHASE (200 TRIALS)                 │
└─────────────────────────────────────────────────────────────────────┘

Input: Model + Refusal Directions + Good Prompts
Output: Best parameters + Abliterated model

┌─────────────────────────────────────────────────────────────────────┐
│  Initialize Optuna Study                                            │
│                                                                     │
│  study = optuna.create_study(                                      │
│      study_name="heretic_study",                                   │
│      storage="sqlite:///heretic_study.db",  # Resume support       │
│      sampler=TPESampler(),                                         │
│      directions=["minimize", "minimize"],   # Multi-objective      │
│  )                                                                  │
│                                                                     │
│  if use_warm_start_params:                                         │
│      enqueue_warm_start_trials(study, model_family)                │
│      # Use known-good parameters for model family                  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRIAL LOOP (repeat 200 times)                                     │
│                                                                     │
│  for trial_num in range(200):                                      │
│                                                                     │
│    ┌─────────────────────────────────────────────────────────────┐ │
│    │ Step 1: Sample Parameters (Optuna TPE Sampler)             │ │
│    │                                                             │ │
│    │  parameters = {                                             │ │
│    │      "attn.o_proj": {                                       │ │
│    │          "layer_min": trial.suggest_float(0.0, 1.0),       │ │
│    │          "layer_max": trial.suggest_float(0.0, 1.0),       │ │
│    │          "weight": trial.suggest_float(0.0, 10.0),         │ │
│    │      },                                                     │ │
│    │      "mlp.down_proj": {                                     │ │
│    │          "layer_min": trial.suggest_float(0.0, 1.0),       │ │
│    │          "layer_max": trial.suggest_float(0.0, 1.0),       │ │
│    │          "weight": trial.suggest_float(0.0, 10.0),         │ │
│    │      },                                                     │ │
│    │  }                                                          │ │
│    │                                                             │ │
│    │  Each trial tests different layer ranges and weights       │ │
│    └─────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│    ┌─────────────────────────▼───────────────────────────────────┐ │
│    │ Step 2: Apply Abliteration with Parameters                 │ │
│    │                                                             │ │
│    │  model.abliterate(refusal_directions, parameters)          │ │
│    │                                                             │ │
│    │  For each layer in [layer_min, layer_max]:                 │ │
│    │    For each component (attn.o_proj, mlp.down_proj):        │ │
│    │      W = component.weight                                  │ │
│    │      d = refusal_direction[layer]                          │ │
│    │      projector = I - (d @ d.T)  # Orthogonal projector    │ │
│    │      W_new = weight * (projector @ W)                      │ │
│    │      component.weight = W_new                              │ │
│    │                                                             │ │
│    │  Removes refusal direction from weight matrices            │ │
│    └─────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│    ┌─────────────────────────▼───────────────────────────────────┐ │
│    │ Step 3: Evaluate Abliterated Model                         │ │
│    │                                                             │ │
│    │  score = evaluator.evaluate(model, good_prompts)           │ │
│    │                                                             │ │
│    │  ┌─────────────────────────────────────────────────────┐   │ │
│    │  │ 3a. KL Divergence Calculation                       │   │ │
│    │  │                                                     │   │ │
│    │  │  orig_probs = original_model(prompts)              │   │ │
│    │  │  ablated_probs = ablated_model(prompts)            │   │ │
│    │  │  kl_div = sum(orig * log(orig / ablated))          │   │ │
│    │  │                                                     │   │ │
│    │  │  Measures capability preservation                  │   │ │
│    │  │  Lower = Better (less behavior change)             │   │ │
│    │  └─────────────────────────────────────────────────────┘   │ │
│    │                       │                                     │ │
│    │  ┌─────────────────────▼───────────────────────────────┐   │ │
│    │  │ 3b. Refusal Counting                                │   │ │
│    │  │                                                     │   │ │
│    │  │  for prompt in good_prompts:                        │   │ │
│    │  │      response = model.generate(prompt, max_30_tok)  │   │ │
│    │  │      if contains_refusal_marker(response):          │   │ │
│    │  │          refusal_count += 1                         │   │ │
│    │  │                                                     │   │ │
│    │  │  Markers: "I cannot", "I can't", "I'm sorry"       │   │ │
│    │  │  Lower = Better (fewer false refusals)             │   │ │
│    │  └─────────────────────────────────────────────────────┘   │ │
│    │                                                             │ │
│    │  Return: (kl_divergence, refusal_count)                    │ │
│    └─────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│    ┌─────────────────────────▼───────────────────────────────────┐ │
│    │ Step 4: CRITICAL - Reload Model for Next Trial             │ │
│    │                                                             │ │
│    │  model.reload_model()                                       │ │
│    │                                                             │ │
│    │  Layer-Wise Caching (v1.2.0+):                             │ │
│    │  ┌─────────────────────────────────────────────────────┐   │ │
│    │  │ if layer_weights_cache is not None:                │   │ │
│    │  │     # Fast: Restore from memory (10-15s for 32B)   │   │ │
│    │  │     for layer_idx in range(num_layers):            │   │ │
│    │  │         for component in ["attn", "mlp"]:          │   │ │
│    │  │             for matrix in component.weights:       │   │ │
│    │  │                 matrix.copy_(cached[layer][comp])  │   │ │
│    │  │                                                     │   │ │
│    │  │     Cache size: 28GB (vs 62GB full cache)          │   │ │
│    │  │     6-12x faster than disk reload                  │   │ │
│    │  │ else:                                               │   │ │
│    │  │     # Slow: Reload from disk (60-120s for 32B)     │   │ │
│    │  │     model = AutoModelForCausalLM.from_pretrained() │   │ │
│    │  └─────────────────────────────────────────────────────┘   │ │
│    │                                                             │ │
│    │  Reset model to pristine state for next trial              │ │
│    └─────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│    ┌─────────────────────────▼───────────────────────────────────┐ │
│    │ Step 5: Optuna Update                                      │ │
│    │                                                             │ │
│    │  trial.report(kl_divergence, refusal_count)                │ │
│    │                                                             │ │
│    │  TPE Sampler learns from results:                          │ │
│    │    - Good trials → sample nearby parameters                │ │
│    │    - Bad trials → avoid similar parameters                 │ │
│    │                                                             │ │
│    │  Median Pruner prunes unpromising trials early             │ │
│    └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  end for                                                            │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Select Best Trial (Pareto Optimal)                                │
│                                                                     │
│  Plot all trials on 2D space:                                      │
│    X-axis: KL Divergence (capability preservation)                 │
│    Y-axis: Refusal Count (false refusals)                          │
│                                                                     │
│  Pareto frontier = trials not dominated by any other               │
│  Best trial = closest to origin (0, 0) on frontier                 │
│                                                                     │
│  best_params = study.best_trial.params                             │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Apply Best Parameters to Model                                    │
│                                                                     │
│  model.reload_model()  # Start from pristine state                 │
│  model.abliterate(refusal_directions, best_params)                 │
│                                                                     │
│  Final abliterated model ready for saving                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Timeline (32B Model, 200 Trials)

```
┌─────────────────────────────────────────────────────────────────────┐
│             TIMELINE: 32B Model Abliteration (200 Trials)          │
└─────────────────────────────────────────────────────────────────────┘

WITHOUT Caching (v1.1.x and earlier):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│<── Setup ──>│<───── Direction Extraction ────>│<─── 200 Trials ──>│
│             │                                 │                   │
│   5 min     │        15-20 min               │    13-15 hours     │
│             │                                 │                   │
│             │                                 │  Per Trial:       │
│             │                                 │  - Abliterate: 5s │
│             │                                 │  - Evaluate: 45s  │
│             │                                 │  - Reload: 90s    │
│             │                                 │  Total: ~2.3 min  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~14 hours

WITH Layer-Wise Caching (v1.2.0+):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│<── Setup ──>│<───── Direction Extraction ────>│<─── 200 Trials ──>│
│             │                                 │                   │
│   5 min     │        15-20 min               │     9-11 hours     │
│  +cache     │                                 │                   │
│  creation   │                                 │  Per Trial:       │
│  (2 min)    │                                 │  - Abliterate: 5s │
│             │                                 │  - Evaluate: 45s  │
│             │                                 │  - Reload: 12s ✓  │
│             │                                 │  Total: ~1.0 min  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~10 hours

SAVINGS: 3-4 hours (30% faster) | $6-8 on H200 @ $2.14/hr
```

---

## Memory Layout (32B Model on H200 141GB)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MEMORY USAGE: 32B Model (H200 141GB)              │
└─────────────────────────────────────────────────────────────────────┘

v1.1.x (Without Layer-Wise Caching):
┌─────────────────────────────────────────────────────────────────────┐
│ Model Weights                                    62 GB              │
│ ████████████████████████████████████████████████████               │
├─────────────────────────────────────────────────────────────────────┤
│ Weight Cache                                     DISABLED (OOM)     │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Would be 62 GB     │
├─────────────────────────────────────────────────────────────────────┤
│ HuggingFace Cache                                5-10 GB            │
│ ████                                                                │
├─────────────────────────────────────────────────────────────────────┤
│ Working Memory                                   10 GB              │
│ ████                                                                │
└─────────────────────────────────────────────────────────────────────┘
Total Used: 77-82 GB
Reload Time: 60-120s per trial


v1.2.0+ (With Layer-Wise Caching):
┌─────────────────────────────────────────────────────────────────────┐
│ Model Weights                                    62 GB              │
│ ████████████████████████████████████████████████████               │
├─────────────────────────────────────────────────────────────────────┤
│ Layer-Wise Cache (abliterable only)             28 GB ✓            │
│ ██████████████████████                                             │
├─────────────────────────────────────────────────────────────────────┤
│ HuggingFace Cache                                5-10 GB            │
│ ████                                                                │
├─────────────────────────────────────────────────────────────────────┤
│ Working Memory                                   10 GB              │
│ ████                                                                │
└─────────────────────────────────────────────────────────────────────┘
Total Used: 105-110 GB ✓ FITS ON 141GB H200
Reload Time: 10-15s per trial

Cache Breakdown:
- attn.o_proj: ~12GB (10-12% of model params)
- mlp.down_proj: ~16GB (33% of model params)
- Total: ~28GB (43% of model params)
- Reduction: 55% vs full cache (62GB → 28GB)
```

---

## Phase 4: Model Saving

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL SAVING PHASE                           │
└─────────────────────────────────────────────────────────────────────┘

Input: Abliterated Model + Best Parameters
Output: Saved model on disk / HuggingFace Hub

┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Save Locally                                               │
│                                                                     │
│  save_model_local(model, output_path)                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Save model state                                           │ │
│  │    model.save_pretrained(output_path)                         │ │
│  │    tokenizer.save_pretrained(output_path)                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Save metadata                                              │ │
│  │    metadata = {                                               │ │
│  │        "source_model": "Qwen/Qwen2.5-Coder-32B-Instruct",    │ │
│  │        "best_params": best_trial.params,                      │ │
│  │        "kl_divergence": best_trial.values[0],                 │ │
│  │        "refusal_count": best_trial.values[1],                 │ │
│  │        "n_trials": 200,                                       │ │
│  │        "heretic_version": "1.2.0",                            │ │
│  │    }                                                           │ │
│  │    json.dump(metadata, "heretic_metadata.json")               │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Optional HuggingFace Upload                                │
│                                                                     │
│  if hf_upload:                                                     │
│    upload_model_huggingface(model, repo_id)                        │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Create/update repository                                   │ │
│  │    api.create_repo(repo_id, exist_ok=True)                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 2. Upload model files                                         │ │
│  │    model.push_to_hub(repo_id)                                 │ │
│  │    tokenizer.push_to_hub(repo_id)                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                               │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ 3. Generate model card                                        │ │
│  │    card = ModelCard.from_template(                            │ │
│  │        model_name=source_model,                               │ │
│  │        results=metadata,                                      │ │
│  │    )                                                           │ │
│  │    card.push_to_hub(repo_id)                                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Output Structure                                                   │
│                                                                     │
│  output_path/                                                      │
│  ├─ config.json                  (Model configuration)             │
│  ├─ model-*.safetensors           (Model weights, sharded)         │
│  ├─ tokenizer.json                (Tokenizer)                      │
│  ├─ tokenizer_config.json         (Tokenizer config)               │
│  ├─ special_tokens_map.json       (Special tokens)                 │
│  └─ heretic_metadata.json         (Abliteration metadata)          │
│                                                                     │
│  Size: ~62-65 GB for 32B model                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW THROUGH HERETIC                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   Prompts   │
│             │
│ - Good      │─┐
│ - Bad       │ │
│ - C4        │ │
│ - MMLU      │ │
└─────────────┘ │
                │
                ▼
         ┌──────────────┐
         │    Model     │
         │              │
         │ get_residuals│
         │              │
         └──────┬───────┘
                │
                ▼
         Hidden States
         (n_prompts × n_layers × hidden_dim)
                │
                ▼
         ┌──────────────┐
         │ GPU PCA      │──────────► Refusal Directions
         │ (v1.0.1+)    │            (n_layers × hidden_dim)
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │Orthogonalize │──────────► Orthogonalized Directions
         │  (optional)  │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │    Optuna    │
         │ TPE Sampler  │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  Parameters  │
         │              │
         │ layer_min    │
         │ layer_max    │───────────┐
         │ weight       │           │
         └──────────────┘           │
                                    │
                ┌───────────────────┘
                │
                ▼
         ┌──────────────┐
         │  Abliterate  │
         │              │
         │ W -= w(d⊗d)W │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Evaluate   │
         │              │
         │ KL Div       │──────────► Scores
         │ Refusals     │            (kl, refusals)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │    Reload    │
         │              │
         │ Layer-wise   │◄───────── Cache
         │ Cache (v1.2) │            (28GB for 32B)
         └──────┬───────┘
                │
                │ (repeat 200x)
                │
                ▼
         ┌──────────────┐
         │ Best Params  │
         │              │
         │ Final Model  │──────────► Save to Disk
         └──────────────┘
```

---

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ERROR HANDLING IN HERETIC                      │
└─────────────────────────────────────────────────────────────────────┘

Cache Creation:
┌─────────────────────────────────────────────────────────────────────┐
│  _create_layer_weights_cache()                                     │
│                                                                     │
│  Try:                                                               │
│    ├─ get_layer_matrices() ──────┬─► Success → Clone tensors       │
│    │                              │                                 │
│    │                              └─► Error → ModelLoadError        │
│    │                                  "Failed to access layer N"    │
│    │                                                                │
│    ├─ tensor.clone().detach() ───┬─► Success → Add to cache        │
│    │                              │                                 │
│    │                              └─► RuntimeError (OOM)            │
│    │                                  → ModelLoadError              │
│    │                                  "Insufficient memory, use     │
│    │                                   --cache-weights false"       │
│    │                                                                │
│    └─ Validate cache ────────────┬─► Non-empty → Return cache      │
│                                   │                                 │
│                                   └─► Empty → ModelLoadError        │
│                                       "Cache creation resulted      │
│                                        in empty cache"              │
└─────────────────────────────────────────────────────────────────────┘

Cache Restoration:
┌─────────────────────────────────────────────────────────────────────┐
│  reload_model()                                                     │
│                                                                     │
│  If layer_weights_cache:                                           │
│    ├─ Check layer count ─────────┬─► Match → Continue              │
│    │                              │                                 │
│    │                              └─► Mismatch → ModelLoadError     │
│    │                                  "Cache has N layers, model    │
│    │                                   has M layers"                │
│    │                                                                │
│    ├─ Check layer exists ────────┬─► Exists → Continue             │
│    │                              │                                 │
│    │                              └─► Missing → ModelLoadError      │
│    │                                  "Layer N missing from cache"  │
│    │                                                                │
│    ├─ Check component exists ────┬─► Exists → Continue             │
│    │                              │                                 │
│    │                              └─► Missing → ModelLoadError      │
│    │                                  "Component 'X' missing"       │
│    │                                                                │
│    ├─ Check shapes match ────────┬─► Match → Continue              │
│    │                              │                                 │
│    │                              └─► Mismatch → ModelLoadError     │
│    │                                  "Shape mismatch at layer N"   │
│    │                                                                │
│    ├─ Check dtype match ─────────┬─► Match → Continue              │
│    │                              │                                 │
│    │                              └─► Mismatch → WARNING            │
│    │                                  "dtype mismatch, converting"  │
│    │                                  → Auto-convert                │
│    │                                                                │
│    └─ tensor.copy_() ────────────┬─► Success → Continue            │
│                                   │                                 │
│                                   └─► Error → ModelLoadError        │
│                                       "Failed to restore weights"   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION PRECEDENCE                         │
└─────────────────────────────────────────────────────────────────────┘

Priority (Highest to Lowest):
┌─────────────────────────────────────────────────────────────────────┐
│ 1. CLI Arguments                                                    │
│    heretic --model MODEL --cache-weights true --n-trials 200        │
│    ✓ Overrides everything                                           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Environment Variables                                            │
│    export HERETIC_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"          │
│    export HERETIC_CACHE_WEIGHTS=true                                │
│    ✓ Used if CLI arg not provided                                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. config.toml File                                                 │
│    [heretic]                                                        │
│    model = "Qwen/Qwen2.5-Coder-32B-Instruct"                        │
│    cache_weights = true                                             │
│    ✓ Used if no CLI arg or env var                                  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Default Values (config.py)                                      │
│    cache_weights: bool = True                                       │
│    n_trials: int = 200                                              │
│    ✓ Used if nothing else specified                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Key Workflow Improvements (v1.2.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    v1.2.0 IMPROVEMENTS                              │
└─────────────────────────────────────────────────────────────────────┘

Layer-Wise Weight Caching:
  ✓ Caches only abliterable components (attn.o_proj, mlp.down_proj)
  ✓ 55-75% memory reduction (62GB → 28GB for 32B)
  ✓ Enables caching for 32B+ models on 141GB GPUs
  ✓ 6-12x faster reload (10-15s vs 60-120s disk)
  ✓ Saves 3-4 hours per 200-trial run
  ✓ $6-8 cost savings on cloud GPUs

Error Handling:
  ✓ Comprehensive validation at cache creation
  ✓ Validates cache-model compatibility on reload
  ✓ Clear, actionable error messages
  ✓ Auto-corrects dtype/device mismatches
  ✓ Reports specific layer/component/matrix in errors

Cache Statistics:
  ✓ Logs cache size and parameter count
  ✓ Shows percentage of model cached
  ✓ Example: "Cache size: 28.4 GB (43.2% of model)"

Previous Improvements:
  ✓ GPU-accelerated PCA (v1.0.1): 15-20x faster
  ✓ C4 dataset streaming (v1.1.0): No disk overhead
  ✓ Resume support: SQLite storage + study names
  ✓ torch.compile(): 1.5-2x inference speedup
  ✓ Early stopping: 40-60% faster evaluation
