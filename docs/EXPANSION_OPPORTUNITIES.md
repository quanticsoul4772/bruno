# Heretic Expansion Opportunities

Strategic analysis of next-level improvements for neural behavior modification.

**Date**: 2026-01-30
**Status**: Research Proposals

---

## Executive Summary

Heretic's core technique (contrastive activation analysis + orthogonalization) is validated and production-ready. This document explores expansion opportunities from mathematics, neuroscience, physics, machine learning research, and philosophical approaches.

**Key Insight**: Current implementation uses linear methods (PCA) and heuristic layer targeting. Opportunities exist in:
1. Non-linear direction discovery (manifold learning)
2. Causal validation of layer targeting (activation patching)
3. Information-theoretic optimization (mutual information)
4. Few-shot behavior extraction (meta-learning)
5. Compositional behavior engineering (behavior algebra)

---

## Tier 1: High Impact, Low-Medium Effort

### 1. Information-Theoretic Direction Selection

**Current**: PCA maximizes variance difference
**Problem**: High variance doesn't guarantee behavioral relevance
**Solution**: Maximize mutual information between direction projection and behavioral outcome

**Implementation**:
- Replace variance maximization with MI maximization
- Use `sklearn.feature_selection.mutual_info_classif`
- Random projection candidates ranked by MI with refusal labels

**Expected Impact**: +10-20% direction quality, more robust to distribution shift
**Effort**: 2-3 days
**Risk**: Low (drop-in replacement for PCA)

---

### 2. Hierarchical Behavior Profiles (Neuroscience-Inspired)

**Current**: Fixed layer multipliers (early/middle/late)
**Insight**: Brain processes hierarchically - features → objects → concepts → intentions
**Solution**: Model behaviors as hierarchical levels with level-specific abliteration

**Levels**:
- **Syntactic** (0.0-0.3): Token patterns ("I cannot", "I'm unable")
- **Semantic** (0.3-0.7): Meaning patterns (safety concerns, harm concepts)
- **Intentional** (0.7-1.0): Goal patterns (boundary-setting, policy enforcement)

**Implementation**:
```python
@dataclass
class HierarchicalBehaviorProfile:
    syntactic_weight: float = 0.3   # Light touch - preserve grammar
    semantic_weight: float = 1.0    # Primary target
    intentional_weight: float = 0.8  # Moderate - preserve assertiveness
```

**Expected Impact**: +15-25% capability preservation, enables partial abliteration
**Effort**: 3-4 days
**Risk**: Low (extends existing layer profiles)

---

### 3. Attention-Weighted Direction Extraction

**Current**: All tokens contribute equally to direction extraction
**Insight**: Refusal decisions concentrate in specific tokens (policy statements)
**Solution**: Weight direction extraction by attention scores to focus on decision-critical tokens

**Implementation**:
- Extract attention scores during residual computation
- Weight token contributions by attention to last token
- Run contrastive PCA on attention-weighted residuals

**Expected Impact**: +10-15% direction quality
**Effort**: 2-3 days
**Risk**: Medium (requires attention score extraction)

---

## Tier 2: High Impact, High Effort

### 4. Causal Mediation Analysis

**Current**: Layer targeting based on heuristics (0.4-0.7 for semantic)
**Problem**: No validation that these layers actually cause refusal
**Solution**: Activation patching to measure causal contribution of each layer

**Method**:
1. Run bad prompt normally → measure refusal
2. Patch layer N with activations from good prompt
3. Measure refusal reduction
4. High reduction → layer N causally contributes

**Implementation**:
- Use `register_forward_hook` for activation patching
- Run systematic layer-by-layer analysis
- Build causal contribution map
- Auto-generate optimal layer profiles

**Expected Impact**: +15-25% by targeting causally-relevant layers
**Effort**: 1-2 weeks
**Risk**: Medium (needs careful hook management)

---

### 5. Meta-Learning for Few-Shot Behaviors

**Current**: Extracting new behavior requires 200+ examples
**Problem**: Slow for exploratory research
**Solution**: Train meta-learner to extract behaviors from 10-20 examples

**Approach**:
- Train on multiple known behaviors (refusal, verbosity, hedging)
- Learn universal extraction network
- Given 10-20 examples → predict direction vector
- Validates against PCA on full dataset during training

**Expected Impact**: 10x faster exploration of new behaviors
**Effort**: 2-3 weeks
**Risk**: High (requires training dataset of diverse behaviors)

---

### 6. Non-Linear Manifold Learning

**Current**: PCA assumes behaviors live in linear subspaces
**Problem**: Neural activations may encode behaviors on curved manifolds
**Solution**: Use Isomap, LLE, or UMAP to discover non-linear structure

**Implementation**:
- Apply manifold learning to bad (refusal-triggering) samples
- Extract tangent space directions at manifold points
- Use for abliteration instead of linear PCA directions

**Expected Impact**: +5-15% for complex behaviors (sycophancy, meta-commentary)
**Effort**: 1-2 weeks
**Risk**: Medium (manifold methods can be unstable)

---

## Tier 3: Paradigm Shifts

### 7. Behavior Algebra - Compositional Modification

**Vision**: Treat behaviors as vectors you can compose algebraically

**Capability**:
```python
modified_model = (
    base_model
    - refusal * 1.0
    - verbosity * 0.7
    + directness * 0.3
)
```

**Requirements**:
- Behavior vector database with known directions
- Composition rules (orthogonalization to prevent conflicts)
- Automatic weight optimization
- Conflict detection and resolution

**Expected Impact**: 10x usability improvement
**Effort**: 1-2 months
**Risk**: High (complex system design)

---

### 8. Streaming Abliteration - Inference-Time Steering

**Vision**: Apply modifications during inference, not permanently

**Advantages**:
- Per-prompt behavior tuning
- User-specific modifications
- A/B testing without retraining
- Zero-cost experimentation

**Implementation**:
- Register forward hooks for activation steering
- Cache behavior directions
- Apply modifications on-the-fly during generation

**Use Cases**:
- Agents with task-specific behaviors
- Personalized AI assistants
- Dynamic safety tuning

**Expected Impact**: Enables new use cases (agents, personalization)
**Effort**: 2-3 weeks
**Risk**: Medium (hook management, performance overhead)

---

### 9. Behavioral Forensics - Discovery-Driven Research

**Vision**: Automatically discover what behaviors exist in a model

**Current Gap**: Hypothesis-driven ("extract refusal") vs discovery-driven ("what behaviors exist?")

**Method**:
1. Sample diverse prompts across domains
2. Cluster residuals to find natural behavior groups
3. Extract direction for each cluster
4. Use LLM to name discovered behaviors

**Implementation**:
- K-means clustering on activation space
- Silhouette analysis for optimal cluster count
- Automated prompt analysis for behavior naming
- Validation through targeted prompt generation

**Expected Impact**: Discovery of unexpected behaviors, research acceleration
**Effort**: 3-4 weeks
**Risk**: High (requires robust clustering and validation)

---

## Zen/Philosophical Approaches

### Sacred Directions - Preserving Critical Capabilities

**Philosophy**: Mastery is knowing what NOT to touch

**Implementation**:
```python
sacred_directions = extract_capability_directions(mmlu_prompts)
refusal_direction = orthogonalize_against_sacred(refusal_dir, sacred_directions)
```

**Workflow**:
1. Extract capability directions from MMLU/benchmark tasks
2. Orthogonalize refusal direction against ALL sacred directions
3. Abliterate with sacred-orthogonalized direction

**Expected Impact**: +20-30% reduction in capability damage
**Effort**: 3-4 days
**Risk**: Low (builds on existing orthogonalization)

---

### Minimum Description Length - Optimal Direction Count

**Philosophy**: Simplest explanation is best (Occam's Razor)

**Current**: `n_refusal_directions=3` is arbitrary

**Solution**: Use MDL to automatically find optimal count
- Model complexity: bits to encode n directions
- Reconstruction error: bits to encode residuals after projection
- Optimal n minimizes total MDL

**Expected Impact**: +5-10% by avoiding under/overfitting
**Effort**: 2 days
**Risk**: Low (straightforward metric)

---

## Cross-Domain Innovations

### Thermodynamic Abliteration

**Physics Insight**: Statistical mechanics temperature controls state exploration

**Innovation**: Temperature-scaled abliteration strength
- T → 0: Aggressive removal (deterministic)
- T = 1: Standard removal
- T → ∞: Soft removal (minimal)

**Adaptive mode**: Measure "refusal energy" per prompt, adjust temperature dynamically

**Use case**: Soft abliteration for safety-critical deployments where complete removal is risky

**Expected Impact**: +10-20% capability preservation on edge cases
**Effort**: 2-3 days

---

### Tensor Decomposition for Behavior Factorization

**Insight**: Behaviors may be composed of shared primitives

**Example**: Refusal = politeness + boundary-setting + topic-avoidance

**Innovation**: Tucker/CP decomposition to factorize behavior tensor into primitives

**Enables**:
- Compositional behavior engineering
- Shared primitive optimization
- 50% reduction in trials (optimize primitives instead of full behaviors)

**Expected Impact**: Foundational for behavior algebra system
**Effort**: 1-2 weeks

---

## Implementation Priority Recommendations

**Completed:**
1. ✅ Sacred direction preservation (3-4 days, +20-30%) - **IMPLEMENTED v1.2.0**
   - `model.extract_sacred_directions()`, `model.orthogonalize_against_sacred()`
   - Config: `use_sacred_directions`, `n_sacred_directions`, `sacred_prompts`

**Start Here (Next Sprint)**:
1. ⏳ Information-theoretic direction selection (2-3 days, +10-20%)
2. ⏳ Hierarchical behavior profiles (3-4 days, +15-25%)
3. ⏳ MDL-based direction count (2 days, +5-10%)

**Research Phase (Next Month)**:
4. Causal mediation analysis (1-2 weeks, +15-25%)
5. Attention-weighted extraction (2-3 days, +10-15%)
6. MDL-based direction count (2 days, +5-10%)

**Long-Term Vision (Next Quarter)**:
7. Meta-learning few-shot extraction (2-3 weeks, 10x faster)
8. Streaming abliteration engine (2-3 weeks, new use cases)
9. Behavioral forensics discovery (3-4 weeks, discovery-driven)

**Moonshot (6-12 Months)**:
10. Behavior algebra system (1-2 months, 10x usability)
11. Non-linear manifold learning (1-2 weeks, complex behaviors)
12. Tensor decomposition framework (1-2 weeks, compositional engineering)

---

## Research Validation Strategy

**For each innovation**:
1. Implement on small model (Qwen2.5-7B-Instruct)
2. Compare against baseline (current heretic)
3. Measure: refusal reduction, KL divergence, MMLU accuracy
4. Validate on 3+ behavioral targets (refusal, verbosity, hedging)
5. If validated: scale to 32B models

**Success criteria**:
- Improvement > 5% on ANY metric (refusal, KL, or MMLU)
- No regression > 5% on ANY metric
- Maintained or improved runtime

---

## Theoretical Foundations

**Current foundations**:
- Linear algebra (PCA, orthogonalization)
- Multi-objective optimization (Pareto frontiers)
- Information theory (KL divergence)

**Proposed additions**:
- **Causal inference** (activation patching, interventions)
- **Manifold learning** (non-linear structure)
- **Information theory** (mutual information, MDL)
- **Meta-learning** (learning to learn behaviors)
- **Statistical mechanics** (temperature, energy landscapes)

These foundations position heretic as a **rigorous scientific tool** for neural behavior modification, not just an engineering utility.

---

## Philosophical Perspective

**Current**: "Remove what you don't want"
**Evolution**: "Understand what exists, preserve what matters, shape what remains"

**Zen principles applicable**:
- **Negative space**: Define behaviors by what they're NOT (sacred directions)
- **Minimal intervention**: Remove only what's necessary (temperature-scaled)
- **Composition over creation**: Combine primitives instead of engineering behaviors from scratch
- **Discovery over hypothesis**: Find what exists before deciding what to change

**Result**: Transform from removal-focused to **behavior understanding and shaping** platform.
