# BRUNO Migration Plan - Complete Renaming Strategy

**From:** Heretic (abliteration-workflow)
**To:** Bruno (or Bruno Forge/Bruno Vector/Bruno Core)

---

## Step 1: Choose Final Name Variant

**Options:**

1. **BRUNO** (single name, simplest)
   - Package: `bruno` or `bruno-ai`
   - CLI: `bruno`
   - Repo: `bruno` or `bruno-ai`

2. **BRUNO FORGE** (compound, creative)
   - Package: `bruno-forge`
   - CLI: `bruno` or `bforge`
   - Repo: `bruno-forge`
   - Tagline: "Forge Infinite Behavioral Worlds"

3. **BRUNO VECTOR** (compound, technical)
   - Package: `bruno-vector`
   - CLI: `bruno` or `bvector`
   - Repo: `bruno-vector`
   - Tagline: "Infinite Behavioral Vectors"

4. **BRUNO CORE** (compound, framework)
   - Package: `bruno-core`
   - CLI: `bruno`
   - Repo: `bruno-core`
   - Tagline: "Core Neural Behavior Framework"

**Recommendation:** Start with **BRUNO FORGE** (honors compound preference, most evocative)

---

## Step 2: Repository Renaming

### GitHub Repository

**Current:** `quanticsoul4772/abliteration-workflow`
**Target:** `quanticsoul4772/bruno-forge`

**Actions:**
1. GitHub Settings → Rename repository
2. Update all git remotes locally
3. Update README badges

**Commands:**
```bash
# After GitHub rename
git remote set-url fork https://github.com/quanticsoul4772/bruno-forge.git
git remote -v  # Verify
```

**Important:** GitHub auto-redirects old URLs, but best to update everywhere

---

## Step 3: Python Package Renaming

### Directory Structure Change

**Current:**
```
src/
  heretic/
    __init__.py
    main.py
    model.py
    config.py
    ...
```

**Target:**
```
src/
  bruno/
    __init__.py
    main.py
    model.py
    config.py
    ...
```

**Action:** Rename directory
```bash
git mv src/heretic src/bruno
```

### Update pyproject.toml

**Changes needed:**

```toml
[project]
name = "bruno-forge"  # was: heretic-llm
version = "1.2.0"
description = "Neural behavior engineering through activation direction analysis"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]

[project.scripts]
bruno = "bruno.main:main"  # was: heretic = "heretic.main:main"
bruno-vast = "bruno.vast:cli"  # was: heretic-vast = "heretic.vast:cli"

[tool.uv.sources]
bruno = { workspace = true }  # was: heretic

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.coverage.run]
source = ["bruno"]  # was: ["heretic"]

[tool.ruff]
src = ["src/bruno"]  # was: ["src/heretic"]
```

### Update All Python Imports

**Files to update:**
- All files in `src/bruno/` (previously `src/heretic/`)
- All files in `tests/`
- All example scripts

**Pattern:**
```python
# OLD
from heretic.config import Settings
from heretic.model import Model
import heretic

# NEW
from bruno.config import Settings
from bruno.model import Model
import bruno
```

**Automated approach:**
```bash
# Find all Python files with heretic imports
grep -r "from heretic" src/ tests/ examples/
grep -r "import heretic" src/ tests/ examples/

# Use sed or manual replacement
find src tests examples -name "*.py" -exec sed -i 's/from heretic/from bruno/g' {} +
find src tests examples -name "*.py" -exec sed -i 's/import heretic/import bruno/g' {} +
```

---

## Step 4: CLI Command Renaming

### Entry Points

**Current commands:**
- `heretic` - Main CLI
- `heretic-vast` - Vast.ai CLI

**Target commands:**
- `bruno` - Main CLI
- `bruno-vast` - Vast.ai CLI

**Updated in pyproject.toml:**
```toml
[project.scripts]
bruno = "bruno.main:main"
bruno-vast = "bruno.vast:cli"
```

### Update All Documentation Examples

**Files to update:**
- README.md
- CLAUDE.md
- WORKFLOW.md
- All files in `docs/`
- All files in `claudedocs/`

**Pattern:**
```bash
# OLD
heretic --model Qwen2.5-Coder-32B-Instruct

# NEW
bruno --model Qwen2.5-Coder-32B-Instruct
```

---

## Step 5: Configuration and Data Files

### Environment Variables

**Current:** `HERETIC_*` prefix
**Target:** `BRUNO_*` prefix

**Examples:**
```bash
# OLD
export HERETIC_MODEL="..."
export HERETIC_CACHE_WEIGHTS=true

# NEW
export BRUNO_MODEL="..."
export BRUNO_CACHE_WEIGHTS=true
```

**Code changes in `src/bruno/config.py`:**
```python
# OLD
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HERETIC_",
        ...
    )

# NEW
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRUNO_",
        ...
    )
```

### Configuration Files

**Current:**
- `config.toml`
- `config.default.toml`

**Content changes:**
```toml
# File headers/comments
# OLD: Heretic Configuration
# NEW: Bruno Configuration

[bruno]  # was: [heretic]
model = "..."
cache_weights = true
...
```

### Database Files

**Current:**
- Default: `heretic_study.db`
- Study name: `heretic_study`

**Target:**
- Default: `bruno_study.db`
- Study name: `bruno_study`

**In config.py:**
```python
storage: str = "sqlite:///bruno_study.db"  # was: heretic_study.db
study_name: str = "bruno_study"  # was: heretic_study
```

---

## Step 6: Documentation Updates

### Major Documentation Files

**Files requiring full review and update:**

1. **README.md**
   - Title: "Bruno Forge" (not "Heretic")
   - Description: Update to new name
   - Installation: `pip install bruno-forge`
   - Usage: All `heretic` commands → `bruno`
   - Repository links

2. **CLAUDE.md**
   - Project overview section
   - All command examples
   - File paths (`src/heretic` → `src/bruno`)
   - CLI flags reference

3. **WORKFLOW.md**
   - All heretic-vast → bruno-vast
   - All command examples
   - Study names and file paths

4. **ROADMAP.md**
   - Vision statement
   - Update project name throughout

5. **LESSONS_LEARNED.md**
   - Update references

6. **QUICK_REFERENCE.md**
   - All command examples

7. **All files in `claudedocs/`**
   - 6+ documentation files
   - All references to heretic

8. **All files in `docs/`**
   - Implementation plans
   - Opportunity analyses

### Minor Files

- `LICENSE` - Update if contains project name
- `Dockerfile` - Update labels, package installation
- `pyproject.toml` - Already covered above
- `.github/workflows/*.yml` - Update job names, descriptions

---

## Step 7: GitHub Actions and CI/CD

### Workflow Files to Update

**`.github/workflows/ci.yml`**
```yaml
name: CI  # Keep or change to "Bruno CI"

jobs:
  checks:
    name: Check and build Bruno  # was: Check and build

    steps:
      - name: Install dependencies
        run: uv sync --all-extras --dev

      # All steps reference new package name
```

**`.github/workflows/test.yml`**
```yaml
name: Tests

jobs:
  test:
    steps:
      # Update any heretic-specific references
```

**`.github/workflows/lint.yml`**
```yaml
# Update paths if needed
run: uv run ruff check src/bruno/  # was: src/heretic/
```

---

## Step 8: Docker and Container Images

### Dockerfile

**Current:**
```dockerfile
# Labels and metadata
LABEL org.opencontainers.image.title="heretic"
LABEL org.opencontainers.image.description="LLM abliteration tool"

# Installation
RUN pip install heretic-llm

# Entrypoint
ENTRYPOINT ["heretic"]
```

**Target:**
```dockerfile
LABEL org.opencontainers.image.title="bruno-forge"
LABEL org.opencontainers.image.description="Neural behavior engineering framework"

RUN pip install bruno-forge

ENTRYPOINT ["bruno"]
```

### Docker Hub

**Current:** `quanticsoul4772/heretic`
**Target:** `quanticsoul4772/bruno-forge`

**Actions:**
1. Build new image with new name
2. Push to Docker Hub
3. Update documentation references

---

## Step 9: PyPI Package Publication

### Package Name Change

**Current PyPI:** `heretic-llm`
**Target PyPI:** `bruno-forge`

**Important:** This is a NEW package, not an update

**Steps:**

1. **Deprecate old package:**
   - Publish final version of `heretic-llm` (v1.1.9)
   - Add deprecation notice in description
   - Point to new package in README

2. **Publish new package:**
   - Build with new name: `uv build`
   - Test upload: `uv publish --test`
   - Production upload: `uv publish`

3. **Installation:**
   ```bash
   # OLD
   pip install heretic-llm

   # NEW
   pip install bruno-forge
   ```

### Version Strategy

**Options:**

A. **Start at v2.0.0** (signals major change)
   - Bruno Forge v2.0.0 (evolved from Heretic v1.2.0)

B. **Continue versioning** (v1.2.0 → v1.3.0)
   - Shows continuity
   - Bruno Forge v1.3.0 (renamed from Heretic v1.2.0)

C. **Reset to v1.0.0** (fresh start)
   - Bruno Forge v1.0.0 (new project)

**Recommendation:** **v2.0.0** (signals evolution, major breaking change in naming)

---

## Step 10: Logging and Internal Strings

### Logger Names

**In logging.py and throughout code:**
```python
# OLD
logger = get_logger("heretic")
logger = get_logger("heretic.model")
logger = get_logger("heretic.vast")

# NEW
logger = get_logger("bruno")
logger = get_logger("bruno.model")
logger = get_logger("bruno.vast")
```

### User-Facing Messages

**Search for user-facing strings:**
```bash
grep -r "heretic" src/ --include="*.py" | grep -i "print\|error\|warning\|info"
```

**Examples to update:**
```python
# OLD
print("Heretic: Neural Behavior Modification")
raise ModelLoadError("Heretic failed to load model")

# NEW
print("Bruno: Neural Behavior Engineering")
raise ModelLoadError("Bruno failed to load model")
```

---

## Step 11: Tests and Test Data

### Test Files

**All test files in `tests/`:**
- Update imports: `from heretic` → `from bruno`
- Update test data paths if they reference "heretic"
- Update test descriptions/docstrings

### Test Fixtures

**Check for:**
- Hardcoded paths with "heretic"
- Test database names: `test_heretic.db` → `test_bruno.db`
- Mock objects with "heretic" attributes

---

## Step 12: External References

### Git Remotes

**Current:**
```
fork: quanticsoul4772/abliteration-workflow
heretic-fork: quanticsoul4772/heretic
origin: p-e-w/heretic
```

**After rename:**
```
fork: quanticsoul4772/bruno-forge
upstream: p-e-w/heretic (keep for reference)
```

**Update:**
```bash
git remote set-url fork https://github.com/quanticsoul4772/bruno-forge.git
git remote rename heretic-fork heretic-upstream
```

### External Documentation

**Update references in:**
- Personal notes
- Blog posts (if any)
- Presentations
- External links

---

## Step 13: Metadata and Attribution

### README Attribution Section

**Add prominent section:**
```markdown
## Origins

Bruno Forge evolved from [Heretic](https://github.com/p-e-w/heretic) by
Philipp Emanuel Weidmann. Named after Giordano Bruno (1548-1600), the
philosopher burned at stake for proposing an infinite universe with
infinite worlds.

Like Bruno revealed infinite cosmic possibilities against imposed doctrine,
this framework reveals infinite behavioral possibilities in neural networks.

Core abliteration technique credit: Philipp Emanuel Weidmann
Advanced features and optimizations: [Your attribution]
```

### License Preservation

**Keep AGPL-3.0 license:**
- Maintain attribution to original author
- Add your contributions to COPYRIGHT
- Update file headers if needed

---

## Complete File Change Checklist

### Python Code Files
- [ ] `src/heretic/` → `src/bruno/` (directory rename)
- [ ] All `*.py` files: Update imports `from heretic` → `from bruno`
- [ ] All `*.py` files: Update logger names
- [ ] All `*.py` files: Update user-facing strings
- [ ] `pyproject.toml`: Update package name, scripts, paths
- [ ] `tests/**/*.py`: Update all imports and references

### Configuration Files
- [ ] `config.toml`: Update section headers
- [ ] `config.default.toml`: Update section headers
- [ ] Environment variable prefix: `HERETIC_` → `BRUNO_`

### Documentation
- [ ] `README.md`: Complete rewrite for Bruno
- [ ] `CLAUDE.md`: Update all examples and references
- [ ] `WORKFLOW.md`: Update all commands
- [ ] `ROADMAP.md`: Update project name
- [ ] `LESSONS_LEARNED.md`: Update references
- [ ] `QUICK_REFERENCE.md`: Update commands
- [ ] `docs/*.md`: Update all references
- [ ] `claudedocs/*.md`: Update all references

### CI/CD and Deployment
- [ ] `.github/workflows/ci.yml`: Update job names
- [ ] `.github/workflows/lint.yml`: Update paths
- [ ] `.github/workflows/test.yml`: Update references
- [ ] `Dockerfile`: Update labels, package, entrypoint
- [ ] Docker Compose (if exists): Update service names

### Scripts and Tools
- [ ] `scripts/*.sh`: Update command references
- [ ] `scripts/*.ps1`: Update command references

### Experiments
- [ ] `experiments/*/README.md`: Update command examples
- [ ] Experiment configs: Update package imports

### Examples
- [ ] `examples/chat_app.py`: Update imports
- [ ] Any example configs

---

## Migration Execution Order

### Phase 1: Preparation (No Breaking Changes)
1. Create migration plan document (this file)
2. Create feature branch: `git checkout -b migrate-to-bruno`
3. Document current state (git status, package list)

### Phase 2: Code Changes (Breaking Changes)
1. Rename directory: `git mv src/heretic src/bruno`
2. Update all Python imports (automated with sed/grep)
3. Update pyproject.toml
4. Update config.py (env prefix)
5. Run tests to verify: `uv run pytest`
6. Fix any import errors

### Phase 3: Documentation Updates
1. Update README.md (major rewrite)
2. Update CLAUDE.md (all examples)
3. Update WORKFLOW.md (all commands)
4. Update all other docs
5. Add attribution section

### Phase 4: CI/CD Updates
1. Update GitHub Actions workflows
2. Update Dockerfile
3. Test local build: `uv build`
4. Verify dist/ artifacts

### Phase 5: Repository and Package
1. Commit all changes
2. Push to fork
3. Rename GitHub repository
4. Update git remotes
5. Publish to PyPI as bruno-forge
6. Update Docker Hub

### Phase 6: Verification
1. Fresh install test: `pip install bruno-forge`
2. CLI test: `bruno --help`
3. Integration test: Run on small model
4. Documentation review
5. Announcement/changelog

---

## Risk Assessment and Mitigation

### High Risk Areas

**1. Import Chain Breaks**
- **Risk:** Missed import causing runtime errors
- **Mitigation:** Comprehensive grep, run full test suite
- **Test:** `uv run pytest tests/ -v`

**2. PyPI Package Name Conflicts**
- **Risk:** bruno-forge already taken on PyPI
- **Mitigation:** Check availability first: `pip search bruno-forge`
- **Backup:** bruno-ai, bruno-framework, bruno-engine

**3. Docker Image References**
- **Risk:** External scripts reference old image
- **Mitigation:** Keep old image with deprecation notice
- **Redirect:** Update Docker Hub description

**4. Git History Confusion**
- **Risk:** Lost context on why files moved
- **Mitigation:** Clear commit message, keep git history
- **Command:** `git mv` preserves history

### Medium Risk Areas

**5. Environment Variable Confusion**
- **Risk:** Users have HERETIC_* vars set
- **Mitigation:** Support both for one version
- **Deprecation:** Warn when old vars detected

**6. Database File Names**
- **Risk:** Existing heretic_study.db files
- **Mitigation:** Accept both old and new names
- **Migration:** Auto-detect and use existing

**7. External Links**
- **Risk:** Documentation links to old URLs
- **Mitigation:** GitHub redirects, but update anyway

---

## Automated Migration Script

### migration_helper.sh

```bash
#!/bin/bash
# Bruno Migration Helper Script

echo "=== Bruno Migration Helper ==="
echo ""

# 1. Check for heretic references in Python files
echo "[1/5] Checking Python imports..."
PYTHON_REFS=$(grep -r "from heretic\|import heretic" src/ tests/ examples/ 2>/dev/null | wc -l)
echo "  Found $PYTHON_REFS Python references to update"

# 2. Check documentation references
echo "[2/5] Checking documentation..."
DOC_REFS=$(grep -r "heretic" *.md docs/ claudedocs/ 2>/dev/null | wc -l)
echo "  Found $DOC_REFS documentation references to update"

# 3. Check configuration files
echo "[3/5] Checking configuration files..."
CONFIG_REFS=$(grep -r "heretic" *.toml 2>/dev/null | wc -l)
echo "  Found $CONFIG_REFS configuration references to update"

# 4. Check GitHub Actions
echo "[4/5] Checking GitHub Actions..."
WORKFLOW_REFS=$(grep -r "heretic" .github/workflows/ 2>/dev/null | wc -l)
echo "  Found $WORKFLOW_REFS workflow references to update"

# 5. Check scripts
echo "[5/5] Checking scripts..."
SCRIPT_REFS=$(grep -r "heretic" scripts/ 2>/dev/null | wc -l)
echo "  Found $SCRIPT_REFS script references to update"

echo ""
echo "=== Total References to Update: $((PYTHON_REFS + DOC_REFS + CONFIG_REFS + WORKFLOW_REFS + SCRIPT_REFS)) ==="
echo ""
echo "Ready to migrate? Run migration steps manually or use sed automation."
```

---

## Estimated Effort

**Total time:** 4-6 hours

| Phase | Estimated Time |
|-------|----------------|
| Preparation | 30 min |
| Code changes | 1-2 hours |
| Documentation updates | 2-3 hours |
| CI/CD updates | 30 min |
| Repository/Package | 30 min |
| Testing and verification | 1 hour |

---

## Breaking Changes Summary

**For users, this is a MAJOR version change:**

```markdown
# Upgrading from Heretic v1.2.0 to Bruno Forge v2.0.0

## Breaking Changes

1. Package name: `heretic-llm` → `bruno-forge`
2. CLI command: `heretic` → `bruno`
3. Vast CLI: `heretic-vast` → `bruno-vast`
4. Python imports: `from heretic` → `from bruno`
5. Environment variables: `HERETIC_*` → `BRUNO_*`
6. Config sections: `[heretic]` → `[bruno]`

## Migration

### Uninstall old package
pip uninstall heretic-llm

### Install new package
pip install bruno-forge

### Update your scripts
sed -i 's/heretic/bruno/g' your_script.py

### Update environment variables
# OLD: export HERETIC_MODEL="..."
# NEW: export BRUNO_MODEL="..."

### Update config files
# In config.toml, change [heretic] to [bruno]
```

---

## Next Steps

**Immediate actions:**

1. **Choose final name variant**
   - Bruno (single)
   - Bruno Forge (recommended)
   - Bruno Vector
   - Bruno Core

2. **Check PyPI availability**
   ```bash
   pip search bruno-forge
   pip search bruno-ai
   pip search bruno-vector
   ```

3. **Create migration branch**
   ```bash
   git checkout -b migrate-to-bruno
   ```

4. **Start with directory rename**
   ```bash
   git mv src/heretic src/bruno
   ```

5. **Run automated checks**
   ```bash
   grep -r "from heretic" src/ tests/
   grep -r "import heretic" src/ tests/
   ```

**Ready to proceed with the migration?**
