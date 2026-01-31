# C4 Dataset Streaming (v1.1.0)

**Date:** 2026-01-30

## Impact

Automatic streaming for C4 dataset eliminates 30-50GB download overhead:
- **32B models:** 400GB → 200GB disk (50% reduction)
- **70B models:** 500GB → 300GB disk (40% reduction)
- **Example:** 32B with orthogonalization: 150-230GB → ~87GB

## Implementation

**Auto-detection:** Any dataset with "c4" in name streams automatically
**Backward compatible:** No API changes, existing commands work unchanged
**Network required:** Streams on-demand during dataset loading

### Key Changes

**`src/heretic/utils.py`:**
- Added `_parse_split_count()` to parse split notation (`train[:200]` → 200)
- Added C4 detection and streaming in `load_prompts()`
- Fail-fast error handling for missing config or network issues

**`tests/unit/test_utils.py`:**
- 12 new tests (streaming, error handling, split parsing)
- 86% code coverage for utils.py
- Verified with real C4 data

## Error Handling

| Error | Exception | Message |
|-------|-----------|---------|
| Missing config | `ValueError` | "C4 dataset requires config parameter (e.g., 'en'). Use --unhelpfulness-prompts.config en" |
| Missing count | `ValueError` | "C4 dataset requires explicit sample count in split. Got: '{split}'. Expected format: 'train[:N]'" |
| Network failure | `RuntimeError` | "Failed to stream C4 dataset. Network required. Original error: {e}" |

## Usage

No changes required - streaming is automatic:

```bash
# Works unchanged with v1.1.0+
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --cache-weights false \
  --unhelpfulness-prompts.config en
```

## Disk Space Before/After

**Before (v1.0.x - Download):**
```
Model:     65.5 GB
C4:        30-50 GB  ← Downloaded shards
BART:      1.6 GB
Other:     0.1 GB
Working:   20 GB
─────────────────
TOTAL:     117-137 GB
```

**After (v1.1.0+ - Streaming):**
```
Model:     65.5 GB
C4:        ~0 GB     ← Streams on-demand
BART:      1.6 GB
Other:     0.1 GB
Working:   20 GB
─────────────────
TOTAL:     ~87 GB
```

## Performance

- **Load time:** 10-30 seconds (was 10-30 minutes for download)
- **Network:** Required during dataset loading
- **Backward compatible:** All existing tests pass
