# PCA Extraction Optimization Plan
**Date:** 2026-01-30
**Problem:** Contrastive PCA takes 4-6 hours on 32B models (64 layers × 5120 hidden dim)
**Goal:** Reduce PCA time from 4-6 hours to <30 minutes

---

## Root Cause Analysis

### Current Implementation (model.py:690-769)

```python
def get_refusal_directions_pca(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    for layer_idx in range(n_layers):  # 64 iterations
        good_layer = good_residuals[:, layer_idx, :].cpu().numpy()  # Move to CPU!
        bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

        # Compute covariance matrices
        cov_good = (good_centered.T @ good_centered) / max(n_good - 1, 1)
        cov_bad = (bad_centered.T @ bad_centered) / max(n_bad - 1, 1)
        cov_contrastive = cov_bad - alpha * cov_good

        # Eigendecomposition (CPU-bound bottleneck)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_contrastive)  # O(d³) where d=5120
```

### Performance Bottleneck

**Per-layer computation:**
- Data: 400 × 5120 matrix (good/bad)
- Covariance: 5120 × 5120 matrix
- Eigendecomposition: **O(5120³) = ~134 billion FLOPs**
- On CPU: ~4-5 minutes per layer
- **Total: 64 layers × 4 min = 256 minutes (4+ hours)**

**Why So Slow:**
1. **CPU-only:** Moves tensors from GPU → CPU → numpy (line 719-720)
2. **Single-threaded:** numpy.linalg.eigh is single-core by default
3. **Sequential:** Processes layers one at a time (no parallelization)
4. **Full eigendecomposition:** Computes all 5120 eigenvalues when we only need top 3

---

## Optimization Strategies

### Strategy 1: Use GPU for Eigendecomposition (Highest Impact)

**Change:** Use `torch.linalg.eigh` on GPU instead of `np.linalg.eigh` on CPU

**Implementation:**
```python
def get_refusal_directions_pca(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    n_layers = good_residuals.shape[1]
    directions = []
    all_eigenvalues = []

    for layer_idx in range(n_layers):
        # KEEP ON GPU (don't move to CPU/numpy)
        good_layer = good_residuals[:, layer_idx, :]  # Still on GPU
        bad_layer = bad_residuals[:, layer_idx, :]

        # Center the data (GPU operations)
        good_centered = good_layer - good_layer.mean(dim=0)
        bad_centered = bad_layer - bad_layer.mean(dim=0)

        # Compute covariance matrices (GPU matmul)
        n_good = good_centered.shape[0]
        n_bad = bad_centered.shape[0]

        cov_good = (good_centered.T @ good_centered) / max(n_good - 1, 1)
        cov_bad = (bad_centered.T @ bad_centered) / max(n_bad - 1, 1)
        cov_contrastive = cov_bad - alpha * cov_good

        # GPU eigendecomposition (MUCH faster than CPU)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_contrastive)
        except RuntimeError:
            # Fallback to mean difference
            diff = (bad_layer.mean(dim=0) - good_layer.mean(dim=0))
            diff = F.normalize(diff.unsqueeze(0), p=2, dim=1)
            layer_directions = diff.expand(n_components, -1)
            directions.append(layer_directions)
            all_eigenvalues.append(torch.ones(n_components, device=good_layer.device))
            continue

        # Take top n_components
        idx = torch.argsort(eigenvalues, descending=True)[:n_components]
        top_eigenvalues = eigenvalues[idx]
        top_directions = eigenvectors[:, idx].T

        # Normalize
        layer_directions = F.normalize(top_directions, p=2, dim=1)
        directions.append(layer_directions)
        all_eigenvalues.append(top_eigenvalues)

    return PCAExtractionResult(
        directions=torch.stack(directions),
        eigenvalues=torch.stack(all_eigenvalues),
    )
```

**Expected speedup:** 10-20x faster (GPU eigendecomposition vs CPU)
- GPU eigendecomposition: ~10-20 seconds per layer
- **Total: 64 layers × 15s = 16 minutes** (vs 4+ hours)

---

### Strategy 2: Parallelize Across Layers (Medium Impact)

**Change:** Process multiple layers simultaneously using GPU parallelization

**Implementation:**
```python
def get_refusal_directions_pca_parallel(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    """Parallel PCA extraction across all layers (vectorized)."""
    # good_residuals: (n_good, n_layers, hidden_dim)
    # bad_residuals: (n_bad, n_layers, hidden_dim)

    # Center data across all layers at once
    good_centered = good_residuals - good_residuals.mean(dim=0, keepdim=True)  # (n_good, n_layers, d)
    bad_centered = bad_residuals - bad_residuals.mean(dim=0, keepdim=True)

    # Compute per-layer covariances using batched operations
    # This is more complex - need to handle batch matrix multiplication
    n_layers = good_residuals.shape[1]

    # Process in chunks of 8 layers to manage memory
    chunk_size = 8
    all_directions = []
    all_eigenvals = []

    for chunk_start in range(0, n_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_layers)

        # Process 8 layers simultaneously
        good_chunk = good_centered[:, chunk_start:chunk_end, :]  # (n_good, 8, d)
        bad_chunk = bad_centered[:, chunk_start:chunk_end, :]

        # Batched covariance computation
        for layer_idx in range(chunk_end - chunk_start):
            # (Keep single-layer loop but GPU-accelerated)
            # ...existing logic...
```

**Expected speedup:** 2-3x faster when combined with GPU eigendecomposition
- Process 8 layers concurrently
- Better GPU utilization
- **Total: ~5-8 minutes**

---

### Strategy 3: Approximate PCA (Highest Impact for Large Hidden Dims)

**Change:** Use randomized/approximate PCA instead of full eigendecomposition

**Implementation:**
```python
def get_refusal_directions_pca_fast(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    """Fast approximate PCA using randomized SVD."""
    from sklearn.decomposition import PCA

    n_layers = good_residuals.shape[1]
    directions = []
    all_eigenvalues = []

    for layer_idx in range(n_layers):
        good_layer = good_residuals[:, layer_idx, :].cpu().numpy()
        bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

        good_centered = good_layer - good_layer.mean(axis=0)
        bad_centered = bad_layer - bad_layer.mean(axis=0)

        # Contrastive data: scale bad up, good down
        contrastive_data = np.vstack([
            np.sqrt(alpha) * bad_centered,
            -good_centered
        ])

        # Randomized PCA (much faster for large matrices)
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        pca.fit(contrastive_data)

        # Use principal components as directions
        top_directions = torch.from_numpy(pca.components_).float()
        layer_directions = F.normalize(top_directions, p=2, dim=1)
        directions.append(layer_directions)

        # Use explained variance as proxy for eigenvalues
        eigenvalues = torch.from_numpy(pca.explained_variance_).float()
        all_eigenvalues.append(eigenvalues)

    return PCAExtractionResult(
        directions=torch.stack(directions),
        eigenvalues=torch.stack(all_eigenvalues),
    )
```

**Expected speedup:** 5-10x faster than full eigendecomposition
- Randomized SVD: O(d² × k) where k=3, not O(d³)
- **Total: ~30-60 minutes** (still CPU but much faster algorithm)

---

### Strategy 4: Hybrid Approach (RECOMMENDED)

**Combine GPU operations with approximate PCA:**

```python
def get_refusal_directions_pca_hybrid(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    """Hybrid GPU/approximate PCA for optimal speed."""
    n_layers = good_residuals.shape[1]
    directions = []
    all_eigenvalues = []

    # Option 1: Use full GPU eigendecomposition for small hidden dims (<2048)
    # Option 2: Use randomized for large hidden dims (>=2048)
    hidden_dim = good_residuals.shape[2]
    use_gpu_full = hidden_dim < 2048

    for layer_idx in range(n_layers):
        if use_gpu_full:
            # Strategy 1: GPU full eigendecomposition (fast for small dims)
            good_layer = good_residuals[:, layer_idx, :]
            bad_layer = bad_residuals[:, layer_idx, :]

            good_centered = good_layer - good_layer.mean(dim=0)
            bad_centered = bad_layer - bad_layer.mean(dim=0)

            cov_good = (good_centered.T @ good_centered) / max(good_layer.shape[0] - 1, 1)
            cov_bad = (bad_centered.T @ bad_centered) / max(bad_layer.shape[0] - 1, 1)
            cov_contrastive = cov_bad - alpha * cov_good

            eigenvalues, eigenvectors = torch.linalg.eigh(cov_contrastive)

            idx = torch.argsort(eigenvalues, descending=True)[:n_components]
            top_eigenvalues = eigenvalues[idx]
            top_directions = eigenvectors[:, idx].T

            layer_directions = F.normalize(top_directions, p=2, dim=1)
            directions.append(layer_directions)
            all_eigenvalues.append(top_eigenvalues)
        else:
            # Strategy 3: Randomized PCA for large dims (5120 for Qwen 32B)
            good_layer = good_residuals[:, layer_idx, :].cpu().numpy()
            bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

            good_centered = good_layer - good_layer.mean(axis=0)
            bad_centered = bad_layer - bad_layer.mean(axis=0)

            contrastive_data = np.vstack([
                np.sqrt(alpha) * bad_centered,
                -good_centered
            ])

            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
            pca.fit(contrastive_data)

            top_directions = torch.from_numpy(pca.components_).float()
            layer_directions = F.normalize(top_directions, p=2, dim=1)
            directions.append(layer_directions)

            eigenvalues = torch.from_numpy(pca.explained_variance_).float()
            all_eigenvalues.append(eigenvalues)

    return PCAExtractionResult(
        directions=torch.stack(directions),
        eigenvalues=torch.stack(all_eigenvalues),
    )
```

**Expected speedup:** 15-20x faster
- Small models (<2048 dim): GPU full eigendecomposition (~2-3 min total)
- Large models (>=2048 dim): Randomized PCA (~20-30 min total)
- **For 32B: ~25-30 minutes** (vs 4-6 hours)

---

### Strategy 5: Cache PCA Results (Future Enhancement)

**Change:** Save PCA results to disk, reuse across experiments

```python
def get_refusal_directions_pca(self, good_residuals, bad_residuals, n_components=3, alpha=1.0):
    # Generate cache key from residuals hash
    cache_key = hashlib.sha256(
        str((good_residuals.shape, bad_residuals.shape, n_components, alpha)).encode()
    ).hexdigest()

    cache_file = Path(f".cache/pca_{cache_key}.pt")

    if cache_file.exists():
        print("* Loading cached PCA results...")
        return torch.load(cache_file)

    # ... compute PCA ...

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    torch.save(result, cache_file)

    return result
```

**Benefit:** Instant PCA for repeat experiments with same datasets

---

## Implementation Roadmap

### Phase 1: Quick Fix - GPU Eigendecomposition (2-4 hours)

**File:** `src/heretic/model.py`
**Lines:** 690-769

**Changes:**
1. Remove `.cpu().numpy()` conversions (line 719-720)
2. Replace `np.linalg.eigh()` with `torch.linalg.eigh()` (line 739)
3. Remove numpy array operations, use torch operations
4. Keep tensor on GPU throughout

**Testing:**
- Test on small model (7B) first: Should take <5 minutes for PCA
- Test on 32B: Should take ~15-20 minutes (vs 4-6 hours)
- Verify eigenvalues match (within numerical precision)

**Risk:** Low (torch.linalg.eigh is stable and well-tested)

---

### Phase 2: Add Approximate PCA for Large Models (4-6 hours)

**File:** `src/heretic/model.py`

**Changes:**
1. Add hidden_dim threshold check (line 714)
2. Implement randomized PCA path for large dims
3. Add config option: `pca_method: str = "auto" | "full" | "randomized"`

**Testing:**
- Compare quality: full vs randomized PCA on 7B model
- Verify speedup: 32B model should take ~25-30 min
- Check abliteration quality (KL divergence + refusals)

**Risk:** Medium (need to validate randomized PCA gives similar quality)

---

### Phase 3: Parallelize Across Layers (8-12 hours)

**File:** `src/heretic/model.py`

**Changes:**
1. Vectorize covariance computation
2. Use batched eigendecomposition
3. Process 8-16 layers concurrently

**Testing:**
- Verify correctness vs sequential
- Test memory usage doesn't explode
- Benchmark speedup

**Risk:** High (complex tensor reshaping, memory management)

---

### Phase 4: Add PCA Result Caching (4-6 hours)

**File:** `src/heretic/model.py`, `src/heretic/config.py`

**Changes:**
1. Add cache directory config
2. Implement cache key generation
3. Save/load PCA results
4. Add cache invalidation logic

**Testing:**
- Verify cache hits work correctly
- Test cache invalidation when params change
- Measure cache size

**Risk:** Low (mostly infrastructure)

---

## Recommended Implementation Order

### Immediate (Before Next 32B Run)

**Implement Strategy 1: GPU Eigendecomposition**
- **Effort:** 2-4 hours
- **Impact:** 15-20x speedup (4-6h → 15-20 min)
- **Risk:** Low
- **Code changes:** ~30 lines in model.py

### Short-term (Next Sprint)

**Implement Strategy 2: Approximate PCA**
- **Effort:** 4-6 hours
- **Impact:** Additional 2-3x on top of Strategy 1 (15-20 min → 5-10 min)
- **Risk:** Medium (need quality validation)

### Long-term (Nice to Have)

**Implement Strategy 3 + 4: Parallelization + Caching**
- **Effort:** 12-18 hours
- **Impact:** Marginal improvement on top of 1+2
- **Risk:** High (complex)

---

## Code Changes for Strategy 1 (GPU Eigendecomposition)

### File: src/heretic/model.py

**Current code (lines 718-764):**
```python
for layer_idx in range(n_layers):
    good_layer = good_residuals[:, layer_idx, :].cpu().numpy()  # ← BAD: Move to CPU
    bad_layer = bad_residuals[:, layer_idx, :].cpu().numpy()

    good_centered = good_layer - good_layer.mean(axis=0)  # ← numpy
    bad_centered = bad_layer - bad_layer.mean(axis=0)

    cov_good = (good_centered.T @ good_centered) / max(n_good - 1, 1)  # ← numpy matmul
    cov_bad = (bad_centered.T @ bad_centered) / max(n_bad - 1, 1)
    cov_contrastive = cov_bad - alpha * cov_good

    eigenvalues, eigenvectors = np.linalg.eigh(cov_contrastive)  # ← SLOW CPU
```

**Optimized code:**
```python
for layer_idx in range(n_layers):
    # Keep on GPU
    good_layer = good_residuals[:, layer_idx, :]  # ← GOOD: Stay on GPU
    bad_layer = bad_residuals[:, layer_idx, :]

    # Upcast to float32 for numerical stability in eigendecomposition
    good_layer = good_layer.float()
    bad_layer = bad_layer.float()

    # Center (GPU operations)
    good_centered = good_layer - good_layer.mean(dim=0)  # ← torch
    bad_centered = bad_layer - bad_layer.mean(dim=0)

    n_good = good_centered.shape[0]
    n_bad = bad_centered.shape[0]

    # Compute covariance matrices (GPU matmul - FAST)
    cov_good = (good_centered.T @ good_centered) / max(n_good - 1, 1)
    cov_bad = (bad_centered.T @ bad_centered) / max(n_bad - 1, 1)
    cov_contrastive = cov_bad - alpha * cov_good

    # GPU eigendecomposition - FAST
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_contrastive)
    except RuntimeError as e:
        # Fallback to mean difference if eigendecomposition fails
        print(f"  * Layer {layer_idx}: Eigendecomposition failed ({e}), using mean difference")
        diff = bad_layer.mean(dim=0) - good_layer.mean(dim=0)
        diff = F.normalize(diff.unsqueeze(0), p=2, dim=1)
        layer_directions = diff.expand(n_components, -1)
        directions.append(layer_directions.cpu())  # Move to CPU for stacking
        all_eigenvalues.append(torch.ones(n_components))
        continue

    # Sort by eigenvalue descending, take top n_components
    idx = torch.argsort(eigenvalues, descending=True)[:n_components]
    top_eigenvalues = eigenvalues[idx]
    top_directions = eigenvectors[:, idx].T  # (n_components, hidden_dim)

    # Normalize
    layer_directions = F.normalize(top_directions, p=2, dim=1)

    # Move to CPU for final stacking (saves GPU memory)
    directions.append(layer_directions.cpu())
    all_eigenvalues.append(top_eigenvalues.cpu())

# Stack results
return PCAExtractionResult(
    directions=torch.stack(directions),  # (n_layers, n_components, hidden_dim)
    eigenvalues=torch.stack(all_eigenvalues),  # (n_layers, n_components)
)
```

**Changes summary:**
- Line 719-720: Remove `.cpu().numpy()`
- Line 723-724: Use `torch` operations instead of `numpy`
- Line 731-732: Use `torch` matmul
- Line 739: Replace `np.linalg.eigh()` → `torch.linalg.eigh()`
- Line 753-763: Use `torch` operations for sorting and indexing
- Add float32 upcast for numerical stability
- Add fallback error handling

---

## Testing Strategy

### Test 1: Verify Correctness
```python
# Compare old CPU vs new GPU implementation
good_residuals_test = torch.randn(400, 64, 5120)
bad_residuals_test = torch.randn(400, 64, 5120)

# Old method
result_cpu = get_refusal_directions_pca_cpu(good_residuals_test, bad_residuals_test)

# New method
result_gpu = get_refusal_directions_pca_gpu(good_residuals_test, bad_residuals_test)

# Verify directions are similar (may differ in sign)
cosine_sim = F.cosine_similarity(
    result_cpu.directions.flatten(),
    result_gpu.directions.flatten(),
    dim=0
)
assert abs(cosine_sim) > 0.99, f"Directions differ: {cosine_sim}"
```

### Test 2: Benchmark Performance
```python
import time

# Small model
good_residuals_7b = torch.randn(400, 32, 2048, device='cuda')
bad_residuals_7b = torch.randn(400, 32, 2048, device='cuda')

start = time.time()
result = model.get_refusal_directions_pca(good_residuals_7b, bad_residuals_7b)
elapsed = time.time() - start

print(f"7B model PCA: {elapsed:.1f} seconds (expect: <60s)")

# Large model
good_residuals_32b = torch.randn(400, 64, 5120, device='cuda')
bad_residuals_32b = torch.randn(400, 64, 5120, device='cuda')

start = time.time()
result = model.get_refusal_directions_pca(good_residuals_32b, bad_residuals_32b)
elapsed = time.time() - start

print(f"32B model PCA: {elapsed:.1f} seconds (expect: <1200s / 20 min)")
```

### Test 3: Quality Validation
```python
# Run abliteration with GPU PCA
# Compare final model quality vs CPU PCA version
# Metrics: KL divergence, refusal count, MMLU scores
```

---

## Expected Impact

| Model Size | Hidden Dim | Current (CPU) | Strategy 1 (GPU) | Strategy 2 (Approx) | Strategy 4 (Hybrid) |
|------------|------------|---------------|------------------|---------------------|---------------------|
| 7B | 2048 | ~60 min | ~3 min | ~10 min | ~3 min |
| 14B | 4096 | ~120 min | ~8 min | ~15 min | ~8 min |
| 32B | 5120 | ~256 min | ~16 min | ~30 min | ~20 min |
| 70B | 8192 | ~600 min | ~45 min | ~60 min | ~45 min |

**For your 32B model:**
- Current: 4-6 hours PCA + 10-12 hours trials = **14-18 hours total**
- With GPU PCA: 20 minutes PCA + 10-12 hours trials = **10.5-12.5 hours total**
- **Saves: 3.5-5.5 hours and $7-11**

---

## Implementation Task

**Priority:** High (saves hours on every large model run)

**File to modify:** `src/heretic/model.py` (single file)
**Lines to change:** ~30 lines in `get_refusal_directions_pca` method
**Testing:** 2-3 hours (correctness + benchmarking)
**Total effort:** 4-6 hours

**Acceptance criteria:**
- [ ] PCA completes in <20 minutes for 32B models
- [ ] Eigenvalues/eigenvectors match CPU version (within 1e-5)
- [ ] GPU memory usage reasonable (<10GB for PCA)
- [ ] Fallback to mean difference if eigendecomposition fails
- [ ] Tests pass for 7B, 14B, 32B models

---

## Next Steps

1. **Now:** Instance terminated, billing stopped
2. **Implement:** GPU eigendecomposition (Strategy 1)
3. **Test:** Validate on 7B model locally
4. **Deploy:** Upload to H200 when PCA is optimized
5. **Run:** Full 200 trials with fast PCA (~10-12 hours total)

Should I implement Strategy 1 (GPU eigendecomposition) now?
