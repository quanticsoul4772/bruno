# Heretic Documentation Organization

This directory contains planning documents, implementation records, and archived documentation.

## Active Documents

### Planning & Implementation
- **IMPLEMENTATION_PLAN.md** - Primary technical roadmap (testing, security, CI/CD)
- **NEXT_LEVEL_IMPROVEMENTS.md** - Enhancement proposals with realistic impact estimates
- **RUNPOD_32B_PLAN.md** - A100 80GB setup plan for 32B models

## Archive

Historical documents moved to `archive/` for reference:
- **IMPROVEMENT_PLAN.md** - Superseded by NEXT_LEVEL_IMPROVEMENTS
- **INNOVATIVE_IMPROVEMENTS.md** - Superseded by NEXT_LEVEL_IMPROVEMENTS
- **DEPLOYMENT_GUIDE.md** - Consolidated into WORKFLOW.md
- **GPU_PCA_IMPLEMENTATION.md** - GPU PCA optimization completed
- **PCA_OPTIMIZATION_PLAN.md** - PCA optimization planning

## Organization Principles

**Root level** (`/`): User-facing documentation only
- README.md - Quick start and overview
- ROADMAP.md - Vision and future direction
- WORKFLOW.md - Cloud GPU comprehensive guide
- CLAUDE.md - AI assistant guide
- LESSONS_LEARNED.md - Troubleshooting and gotchas

**docs/** directory: Planning and implementation tracking
- Active planning documents
- Implementation records
- Archive of superseded documents

**claudedocs/** directory: Claude Code specific documentation
- Technical analyses
- Implementation notes
- Claude-generated reports
