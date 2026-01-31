# Heretic Roadmap

This document captures the vision, research directions, and development plans for heretic.

## Vision

**Heretic is not just an "uncensoring tool."** It's a **surgical instrument for neural behavior modification**.

Abliteration (refusal removal) was the first surgery - but the scalpel works on *any* behavioral direction encoded in model weights.

### The Core Technique Is General

```
1. FIND a direction    → Compare activations: "behaves X" vs "doesn't behave X"
2. PROJECT it out      → Orthogonalize weight matrices against that direction
3. OPTIMIZE intensity  → Find strength that modifies behavior without destroying capability
```

**Refusals are just ONE direction.** Models have directions for:
- Verbosity / conciseness
- Sycophancy / directness
- Hedging / confidence
- Formality / casualness
- Meta-commentary ("Let me think step by step...")
- Safety disclaimers
- Role resistance ("As an AI...")
- ...potentially thousands more

---

## Current State

| Component | Status | Description |
|-----------|--------|-------------|
| **Core CLI** (`heretic`) | ✅ Mature | Optuna optimization, HF upload, auto-select |
| **Phase Modules** (`phases/`) | ✅ New | Modular pipeline (dataset, extraction, optimization, saving) |
| **Sacred Directions** | ✅ New | Capability preservation via MMLU orthogonalization |
| **Constants Module** | ✅ New | Centralized magic numbers and thresholds |
| **Cloud CLI** (`heretic-vast`) | ✅ Mature | Rich dashboard, SSH management, Vast.ai automation |
| **Chat UI** (`chat_app.py`) | ✅ Basic | Gradio, streaming, model switching |
| **Deployment** | ✅ Ready | Docker image, RunPod/Vast.ai scripts |
| **Experiments** | 🔬 Active | Verbosity spike in `experiments/verbosity/` |

---

## Behavioral Directions Research

### Tier 1: High Impact + Easy to Extract (Start Here)

| Direction | What It Does | How to Detect | Agent Value |
|-----------|--------------|---------------|-------------|
| **Verbosity** | Excessive explanation, padding | Same Q → "Explain in detail" vs "One sentence answer" | Highest - saves tokens |
| **Hedging** | "I think...", "It's possible..." | Factual Q → confident vs hedged answers | Critical for decisiveness |
| **Meta-commentary** | "Let me think step by step..." | Problem → just answer vs narrated thinking | Tool agents |

### Tier 2: High Impact + Medium Difficulty

| Direction | What It Does | How to Detect | Agent Value |
|-----------|--------------|---------------|-------------|
| **Sycophancy** | "Great question!", unearned praise | User shares mediocre work → honest critique vs praise | Editor/critic agents |
| **Safety disclaimers** | "Consult a professional..." | Medical/legal Q → direct vs disclaimed | Research agents |
| **Role resistance** | "As an AI, I don't..." | Roleplay → commits to character vs breaks | Creative agents |
| **Explanation-seeking** | Asking clarifying questions | Ambiguous request → just does it vs asks | Autonomous agents |

### Tier 3: Style & Tone Modifiers

| Direction | What It Does | How to Detect |
|-----------|--------------|---------------|
| **Formality** | Corporate-speak, professional tone | Same content → casual vs formal |
| **Both-sidesing** | "On one hand... on the other..." | Opinion Q → decisive vs balanced |
| **List-ification** | Converting everything to bullets | Same request → prose vs list |
| **Instruction echoing** | Restating the question before answering | Any request → just answers vs restates |

### Research Questions

1. **Do directions transfer?** Does Llama's verbosity direction work on Qwen?
2. **Do they compose?** Can you stack 3-5 directions without destroying capability?
3. **What's entangled?** Does removing verbosity accidentally affect thoroughness?
4. **What's the minimal modification?** How few layers can you touch?

---

## Purpose Agents Vision

**Agent capabilities are bottlenecked by model behavior, not just intelligence.**

A brilliant model that hedges, refuses, or buries answers in verbosity makes a terrible agent. Purpose agents need purpose alignment.

### Example Purpose Agents

| Agent Type | Purpose Modifications | Result |
|------------|----------------------|--------|
| **Coder Agent** | -verbosity, -hedging, -explanations, +directness | Outputs code, not essays about code |
| **Research Agent** | -refusal, +curiosity, -self-censorship | Actually explores controversial topics |
| **Editor Agent** | -sycophancy, +criticism, +directness | Gives real feedback, not "great job!" |
| **Creative Agent** | +creativity, -safety, -predictability | Actually creative, not "safe creative" |
| **Reasoning Agent** | -verbosity, +step-by-step, -hedging | Commits to conclusions, shows work |
| **Tool Agent** | -verbosity, -explanation, +structured-output | Just calls tools, minimal commentary |

### Architecture Sketch

```
┌─────────────────────────────────────────────────────────────────┐
│                    PURPOSE AGENT SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Coder Agent │    │Research Agent│    │ Editor Agent│        │
│   │             │    │              │    │             │        │
│   │  qwen-72b   │    │  llama-70b   │    │  mistral-   │        │
│   │  -verbose   │    │  -refusal    │    │  22b        │        │
│   │  -hedge     │    │  +curious    │    │  -sycophant │        │
│   │  +direct    │    │  +thorough   │    │  +critical  │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                  │                  │                 │
│          └──────────────────┼──────────────────┘                │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │  Orchestrator   │                          │
│                    │  (routes tasks) │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Multi-Direction Architecture (Future)

### Principles

1. **Evidence first, framework second** - Run experiments before building abstractions
2. **Directions as artifacts** - Save/load extracted directions as reusable tensors
3. **Composable** - Stack multiple directions on the same model
4. **Pluggable evaluators** - Each direction type has its own success metric

### Proposed File Structure

```
src/heretic/
├── main.py                 # CLI entry (backward compatible)
├── config.py               # Settings (extended)
├── model.py                # Model + abliterate (generalized)
├── evaluator.py            # Keep for backward compat
├── directions/             # NEW MODULE (future)
│   ├── __init__.py
│   ├── base.py             # Direction dataclass, save/load
│   ├── extractor.py        # Extract directions from models
│   ├── evaluators.py       # Pluggable evaluators
│   └── registry.py         # Built-in direction specs
└── vast.py                 # Cloud CLI
```

### Core Abstractions (Planned)

```python
@dataclass
class Direction:
    """A reusable behavioral direction extracted from a model."""
    name: str                    # "refusal", "verbosity", "hedging"
    tensor: Tensor               # Shape: (num_layers+1, hidden_dim)
    source_model: str            # Model it was extracted from
    hidden_dim: int              # For compatibility checking
    num_layers: int
    metadata: dict               # Timestamps, extraction config, etc.

    def save(self, path: Path): ...

    @classmethod
    def load(cls, path: Path) -> "Direction": ...
```

---

## Current Experiments

### Verbosity Spike v1 (`experiments/verbosity/`) - COMPLETE

**Status:** ✅ Complete. Extracted a "padding direction" but not a full "elaboration direction."

**Results:**
- Factual questions: 6-30 words (success!)
- Open-ended questions: 195-203 words (unchanged)

**Lesson:** Our prompts conflated question complexity with verbosity. Open-ended questions SHOULD get longer answers.

### Verbosity v2 (`experiments/verbosity_v2/`) - READY

**Goal:** Isolate padding behavior from question complexity.

**Approach:** Use SAME questions with different instructions:
- Concise: "What is 2+2?"
- Verbose: "What is 2+2? Please explain in detail."

This targets the padding that gets added when asked for "detail."

### Hedging Experiment (`experiments/hedging/`) - READY

**Goal:** Extract the hedging direction ("I think", "perhaps", "might be").

**Approach:** Contrast factual vs opinion questions:
- Confident: "What is 2+2?" (factual → direct answer)
- Hedged: "Do you think AI will replace jobs?" (opinion → hedged answer)

**Agent Value:** Critical for decision-making agents that need confident recommendations.

See individual experiment READMEs for details.

---

## Development Phases

### Phase 0: Validate Hypotheses (Current)

- [x] Create verbosity prompt dataset
- [ ] Run verbosity spike experiment
- [ ] Test if verbosity direction transfers across models
- [ ] Test composition: refusal + verbosity together

### Phase 1: Direction Infrastructure

- [ ] Add `save_direction()` / `load_direction()` functions
- [ ] Direction file format (`.safetensors` or `.pt`)
- [ ] CLI: `heretic extract` command
- [ ] CLI: `heretic apply` command

### Phase 2: Multiple Directions

- [ ] Create datasets for hedging, sycophancy
- [ ] Test composition of 3+ directions
- [ ] Build direction library/registry
- [ ] Document optimal parameters per direction

### Phase 3: Purpose Agents (Vision)

- [ ] Define agent profiles (modifications per role)
- [ ] Integration with agent frameworks
- [ ] A/B testing UI for comparing behaviors
- [ ] Inference-time direction application (optional)

---

## Key Insights

### Why This Matters

1. **Behaviors trained into models are bugs for agents** - An agent that hedges wastes tokens. An agent that refuses breaks workflows.

2. **The technique is general** - Abliteration proved the concept. The same math applies to any behavioral tendency.

3. **This is mechanistic interpretability** - Understanding which layers encode which behaviors is genuine research.

4. **Build evidence, not frameworks** - Run experiments first, then design abstractions around learnings.

### What Makes Heretic Different

- **Weight modification** vs prompt engineering - Permanent, no token overhead
- **Optimization-based** vs manual tuning - Finds Pareto-optimal parameters
- **Composable directions** (goal) - Stack behaviors like filters
- **Personal research tool** - Not a product, a way to understand LLMs

---

## Resources

- [Original Paper: Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Representation Engineering Paper](https://arxiv.org/abs/2310.01405)
- [Abliterated Models Collection](https://huggingface.co/collections/p-e-w/the-bestiary)
