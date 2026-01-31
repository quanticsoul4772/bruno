# Streaming Opportunities Analysis

Analysis of additional streaming patterns to reduce memory and disk usage.

**Context**: C4 dataset streaming (v1.1.0) successfully reduced disk from 150-230GB to ~87GB for 32B models by eliminating 30-50GB download overhead.

**Goal**: Find other memory-intensive operations that could benefit from streaming.

---

## Current Streaming Implementation

### C4 Dataset Streaming (✅ Implemented v1.1.0)

**What it does**: Streams C4 samples on-demand instead of downloading entire shards
**Impact**: 30-50GB disk reduction (50% reduction for 32B models)
**Pattern**: HuggingFace `load_dataset(streaming=True)` with retry logic

**Key implementation**:
```python
def _load_streaming_dataset(specification, sample_count, base_split):
    dataset = load_dataset(
        specification.dataset,
        specification.config,
        split=base_split,
        streaming=True  # ← Key: stream instead of download
    )

    prompts = []
    for i, example in enumerate(dataset):
        if i >= sample_count:
            break
        prompts.append(example[specification.column])
    return prompts
```

---

## New Streaming Opportunities

### 1. Batch Processing Streaming (HIGH PRIORITY)

**Current Problem**: Accumulates all results in memory before returning

**Location**: `model.py` - Multiple methods
```python
def get_residuals_batched(self, prompts: list[str]) -> Tensor:
    residuals = []
    for batch in batchify(prompts, self.settings.batch_size):
        residuals.append(self.get_residuals(batch))  # ← Accumulates
    return torch.cat(residuals, dim=0)  # ← Single large allocation

def get_responses_batched(self, prompts: list[str]) -> list[str]:
    responses = []
    for batch in batchify(prompts, self.settings.batch_size):
        for response in self.get_responses(batch):
            responses.append(response)  # ← Accumulates
    return responses
```

**Memory Impact**:
- 500 prompts × 80 layers × 5120 dims × 4 bytes = **204 MB** peak for residuals
- 500 responses × 100 tokens = **10-50 MB** for responses

**Streaming Solution**:
```python
def get_residuals_streaming(self, prompts: list[str]):
    """Stream residuals batch by batch without accumulation."""
    for batch in batchify(prompts, self.settings.batch_size):
        yield self.get_residuals(batch)

def get_responses_streaming(self, prompts: list[str]):
    """Stream responses without accumulation."""
    for batch in batchify(prompts, self.settings.batch_size):
        for response in self.get_responses(batch):
            yield response
```

**Benefits**:
- 40-60% peak memory reduction
- Process results incrementally
- Enable larger batch processing

**Effort**: 1-2 days (refactor consumers to use generators)

---

### 2. MMLU Evaluation Streaming (MEDIUM PRIORITY)

**Current Problem**: Loads entire MMLU categories into memory and caches them

**Location**: `validation.py`
```python
def _load_category(self, category: str) -> list[dict]:
    if category in self._datasets:
        return self._datasets[category]  # ← Full cache

    dataset = load_dataset("cais/mmlu", category, split="test")
    examples = []
    for item in dataset:
        examples.append({...})  # ← Accumulates all

    self._datasets[category] = examples  # ← Stores everything
    return examples
```

**Memory Impact**:
- Each category: ~10-20 MB
- All MMLU categories cached: **500+ MB**
- For validation with multiple categories: **50-200 MB**

**Streaming Solution**:
```python
def _load_category_streaming(self, category: str):
    """Stream MMLU examples without full cache."""
    dataset = load_dataset("cais/mmlu", category, split="test", streaming=True)

    for item in dataset:
        yield {
            "question": self._format_question(item),
            "choices": item["choices"],
            "answer": item["answer"]
        }

def evaluate_category(self, category: str) -> MMLUResult:
    """Evaluate streaming - no full category load."""
    correct = 0
    total = 0

    for example in self._load_category_streaming(category):
        # Process immediately, no accumulation
        if total >= self.samples_per_category:
            break
        # Evaluate...
        total += 1

    return MMLUResult(category, correct, total, correct / total)
```

**Benefits**:
- 90% memory reduction for MMLU evaluation
- Enable larger `samples_per_category` without OOM
- Faster validation startup (no pre-loading)

**Effort**: 1 day

---

### 3. Model Upload Streaming (MEDIUM PRIORITY)

**Current Problem**: Uploads model sequentially with no progress or resume support

**Location**: `main.py`
```python
model.model.push_to_hub(
    settings.hf_upload,
    private=settings.hf_private,
    token=token
)  # ← Blocks 10-30 min, no resume
```

**Disk/Network Impact**:
- 32B model: **62 GB** upload
- Sequential: 10-30 minutes
- Network failure = restart from beginning

**Streaming Solution**:
```python
# Enable HuggingFace Hub's built-in streaming upload
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HUB_TRANSFER_CONCURRENCY'] = '4'  # Parallel uploads

# Automatic resume on failure
model.model.push_to_hub(
    settings.hf_upload,
    private=settings.hf_private,
    token=token,
    max_shard_size="2GB"  # Stream in 2GB chunks
)
```

**Benefits**:
- **2-4x faster** uploads via parallelization
- Automatic resume on network failure
- Progress visibility
- No code changes required (environment variables)

**Effort**: 0.5 days (just configuration + testing)

---

### 4. PCA Computation Streaming (RESEARCH NEEDED)

**Current Problem**: PCA requires all residuals in memory for covariance computation

**Location**: `model.py` - `get_refusal_directions_pca()`
```python
def get_refusal_directions_pca(self, good_residuals, bad_residuals, n_components=3):
    # Requires full residuals: (n_samples, n_layers, hidden_dim)
    good_means = good_residuals.mean(dim=0)  # ← Full dataset average
    bad_means = bad_residuals.mean(dim=0)

    # Batched covariance computation
    cov_contrastive = torch.einsum(...)  # ← Needs all data
```

**Memory Impact**:
- 500 samples: 204 MB (manageable)
- 2000 samples: **816 MB** (could be issue)

**Streaming Solution**: Incremental PCA using Welford's algorithm
```python
def get_refusal_directions_pca_streaming(self, good_prompts, bad_prompts, n_components=3):
    """Compute PCA incrementally without loading all residuals."""
    n_layers = len(self.get_layers())
    hidden_dim = self.model.config.hidden_size

    # Initialize accumulators
    good_mean = torch.zeros(n_layers, hidden_dim)
    bad_mean = torch.zeros(n_layers, hidden_dim)
    good_M2 = torch.zeros(n_layers, hidden_dim, hidden_dim)  # Sum of squared deviations
    bad_M2 = torch.zeros(n_layers, hidden_dim, hidden_dim)

    # Stream good residuals
    good_count = 0
    for batch in batchify(good_prompts, self.settings.batch_size):
        residuals = self.get_residuals(batch)
        for sample in residuals:
            good_count += 1
            delta = sample - good_mean
            good_mean += delta / good_count
            delta2 = sample - good_mean
            good_M2 += torch.einsum('li,lj->lij', delta, delta2)

    # Stream bad residuals (same pattern)
    # ...

    # Compute covariance from accumulated statistics
    good_cov = good_M2 / (good_count - 1)
    bad_cov = bad_M2 / (bad_count - 1)
    cov_contrastive = bad_cov - good_cov

    # Eigendecomposition (same as current)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_contrastive)
    # ...
```

**Benefits**:
- Support **10x larger** prompt sets (5000+ samples)
- Constant memory usage regardless of dataset size
- Enable more robust direction extraction

**Effort**: 2-3 days (requires careful numerical stability validation)

---

### 5. Optuna Trial Checkpointing Streaming (LOW PRIORITY)

**Current Problem**: All trials stored in SQLite, loaded on study resume

**Location**: `main.py`
```python
study = optuna.create_study(
    study_name=settings.study_name,
    storage=settings.storage,  # sqlite:///heretic_study.db
    load_if_exists=True  # ← Loads all historical trials
)
```

**Memory Impact**:
- 200 trials: ~16 KB (negligible)
- 1000+ trials: Could become issue for long-running studies
- Resuming loads all trial metadata

**Streaming Solution**:
```python
# Custom storage backend with lazy loading
class StreamingStorage(optuna.storages.RDBStorage):
    """Lazy-load trials on demand instead of all at once."""

    def get_all_trials(self, study_id: int, deepcopy: bool = True):
        # Only load recent trials (last 100)
        # Older trials accessed on-demand
        return self._get_recent_trials(study_id, limit=100)
```

**Benefits**:
- Support studies with 1000+ trials without memory bloat
- Faster study initialization
- Better for long-running cloud instances

**Effort**: 3-5 days (complex, Optuna internals)

---

### 6. Weight Cache Streaming (RESEARCH NEEDED)

**Current Problem**: Weight caching stores entire `state_dict()` in memory

**Location**: `model.py`
```python
if settings.cache_weights:
    print("* Caching original weights in memory...")
    self.cached_state_dict = copy.deepcopy(self.model.state_dict())
    # ↑ For 32B: ~62 GB in memory TWICE (model + cache)
```

**Memory Impact**:
- 7B model: ~14 GB cache (OK)
- 32B model: **~62 GB cache (causes OOM)**
- 70B model: ~140 GB cache (impossible)

**Current workaround**: `--cache-weights false` for 32B+

**Streaming Solution**: Layer-wise weight caching
```python
def cache_weights_streaming(self):
    """Cache weights layer-by-layer instead of full model."""
    self.layer_weight_cache = {}

    for layer_idx, layer in enumerate(self.get_layers()):
        # Cache only abliterable components for this layer
        layer_cache = {}
        for component, matrices in self.get_layer_matrices(layer_idx).items():
            layer_cache[component] = [m.clone() for m in matrices]

        self.layer_weight_cache[layer_idx] = layer_cache

    # Result: Cache only what's needed, not full state_dict

def reload_layer(self, layer_idx: int):
    """Reload single layer from cache (streaming)."""
    cached = self.layer_weight_cache[layer_idx]
    for component, matrices in self.get_layer_matrices(layer_idx).items():
        for i, matrix in enumerate(matrices):
            matrix.copy_(cached[component][i])
```

**Benefits**:
- **Enable weight caching for 32B models** (currently disabled)
- 5-10x faster trial reset (currently 32B uses disk reload)
- 50% memory reduction (cache only abliterable matrices, not full state)

**Effort**: 2-3 days

---

### 7. Evaluation Response Streaming (QUICK WIN)

**Current Problem**: Evaluator generates all responses, then processes

**Location**: `evaluator.py`
```python
def count_refusals(self, use_neural: bool | None = None) -> int:
    # Generate ALL responses first
    responses = self.model.get_responses_batched(self.bad_prompts, max_tokens=...)

    # THEN count refusals
    if use_neural:
        return self._count_refusals_hybrid_batched(responses)
    else:
        return self._count_refusals_string_batched(responses)
```

**Memory Impact**: 500 responses × 100 tokens = 10-50 MB

**Streaming Solution**:
```python
def count_refusals_streaming(self) -> int:
    """Count refusals while generating, no accumulation."""
    count = 0

    for response in self.model.get_responses_streaming(self.bad_prompts):
        # Process immediately
        if self._is_refusal(response):
            count += 1

    return count
```

**Benefits**:
- Reduce evaluation memory by 10-20 MB
- Start counting while generating (slight latency improvement)
- Enable early stopping (stop generating if threshold reached)

**Effort**: 0.5 days

---

### 8. Validation Report Streaming (LOW PRIORITY)

**Current Problem**: Validation accumulates all results before saving

**Location**: `validation.py`
```python
def establish_baseline(self):
    """Run all evaluations, then save."""
    kl_div = self.evaluator.get_kl_divergence()
    refusals = self.evaluator.count_refusals()
    mmlu = self._run_mmlu() if self.settings.run_mmlu else None

    # All in memory, then save
    self.baseline = ValidationMetrics(...)
    # Save atomically (good)
```

**Streaming Solution**: Progressive validation with checkpoint saves
```python
def establish_baseline_streaming(self):
    """Stream validation with incremental checkpoints."""
    # Save after each metric
    report = ValidationReport(...)

    report.kl_divergence = self.evaluator.get_kl_divergence()
    report.save_checkpoint()

    report.refusals = self.evaluator.count_refusals()
    report.save_checkpoint()

    if self.settings.run_mmlu:
        report.mmlu = self._run_mmlu()
        report.save_checkpoint()
```

**Benefits**:
- Resume validation if interrupted
- Progress visibility during long MMLU runs
- Fault tolerance

**Effort**: 1 day

---

## Priority Recommendations

### Implement Now (High ROI, Low Effort)

**1. Evaluation Response Streaming** (0.5 days)
- Quick win, immediate memory benefit
- Enables early stopping for evaluation

**2. Model Upload with HF Transfer** (0.5 days)
- Just environment variables, no code changes
- 2-4x faster uploads

### Implement Soon (High Impact, Medium Effort)

**3. Batch Processing Streaming** (1-2 days)
- Significant memory reduction (40-60%)
- Enables larger prompt sets
- Foundational for other improvements

**4. Layer-wise Weight Caching** (2-3 days)
- **Enables weight caching for 32B models** (currently disabled due to OOM)
- 5-10x faster trial reset for 32B models
- Major performance improvement

### Research/Future (High Effort, Unclear Benefit)

**5. Incremental PCA Streaming** (2-3 days)
- Complex numerical validation required
- Only beneficial for 2000+ samples (rare)

**6. Optuna Storage Streaming** (3-5 days)
- Only beneficial for 1000+ trial studies (rare)
- Complex Optuna internals

**7. MMLU Dataset Streaming** (1 day)
- Moderate benefit (~50 MB savings)
- Useful for resource-constrained GPUs

---

## Implementation Plan

### Phase 1: Quick Wins (1 day)
1. ✅ Enable HF Transfer for uploads (environment variables)
2. ✅ Implement evaluation response streaming

### Phase 2: Core Memory Optimization (2-3 days)
3. ✅ Implement batch processing streaming (residuals + responses)
4. ✅ Refactor direction extraction to consume streams

### Phase 3: Major Performance Unlock (2-3 days)
5. ✅ Implement layer-wise weight caching
6. ✅ Enable weight caching for 32B models

### Phase 4: Polish (Optional, 1-2 days)
7. MMLU dataset streaming
8. Validation checkpointing

---

## Key Insight

**The biggest opportunity is layer-wise weight caching**, not more streaming per se.

Current problem:
- 32B models **cannot use weight caching** due to OOM (62 GB × 2 = 124 GB)
- Forced to use disk reload (20-30x slower than memory cache)

Solution:
- Cache only **abliterable matrices** per layer (not full state_dict)
- Estimated memory: ~20-30 GB (50% of current)
- **Enables 5-10x speedup for 32B models**

This is more impactful than streaming because it unlocks performance that's currently blocked.

---

## Summary Table

| Opportunity | Memory Save | Complexity | Effort | Priority |
|-------------|-------------|------------|--------|----------|
| Batch processing streaming | 40-60% | Medium | 1-2 days | HIGH |
| Evaluation response streaming | 10-20 MB | Easy | 0.5 days | HIGH |
| Layer-wise weight cache | 50% cache | Medium | 2-3 days | **CRITICAL** |
| HF upload streaming | 0% (4x faster) | Easy | 0.5 days | MEDIUM |
| MMLU dataset streaming | 50-200 MB | Easy | 1 day | MEDIUM |
| Incremental PCA | Enables 10x data | High | 2-3 days | LOW |
| Validation checkpointing | Fault tolerance | Medium | 1 day | LOW |
| Optuna storage streaming | Rare benefit | High | 3-5 days | LOW |

**Total high-priority effort**: 3-4 days for transformative improvements
