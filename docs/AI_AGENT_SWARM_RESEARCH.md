# AI Agent Swarm Research Report

**Goal:** Build a CLI-based team of AI developers using abliterated coding models

**Research Date:** February 4, 2026

---

## Executive Summary

**Recommended Architecture:**
- **Hybrid: 1× 14B orchestrator + 6× 3B specialists**
- **Hardware:** A6000 48GB or A100 80GB
- **Framework:** CrewAI or Codebuff SDK
- **Models:** Qwen2.5-Coder (best for code, 88.4% HumanEval)
- **Total Memory:** 32GB with 4-bit quantization

**Key Insight:** Small 3B models retain 96% of specialized performance through model merging and abliteration, enabling swarms of 6-10 agents on a single GPU.

---

## 1. Top Coding Models (2026)

| Model | Size | HumanEval | VRAM (4-bit) | Best For |
|-------|------|-----------|--------------|----------|
| **Qwen2.5-Coder** | 0.5B | ~50% | 0.5GB | Embedded agents |
| **Qwen2.5-Coder** | 1.5B | 43.8% | 1GB | Simple tasks |
| **Qwen2.5-Coder** | 3B | ~65% | 2GB | **Specialized agents** ⭐ |
| **Qwen2.5-Coder** | 7B | **88.4%** | 4GB | General purpose |
| **Qwen2.5-Coder** | 14B | 90%+ | 8GB | **Orchestrator** ⭐ |
| **Qwen2.5-Coder** | 32B | 92%+ | 16GB | Complex reasoning |
| DeepSeek-V3 | 236B (37B active) | 65.2% | 20GB | Advanced reasoning |

**Winner:** Qwen2.5-Coder-7B beats GPT-4 (87.1%) and approaches GPT-4o (90.2%)

---

## 2. Multi-Agent Architectures

### Option A: Multiple Specialized 3B Models ⭐ RECOMMENDED

**Configuration:**
- 4-8× Qwen2.5-Coder-3B (abliterated per-role)
- Each agent specializes: frontend, backend, test, security, docs, devops
- Total VRAM: 8-16GB (4-bit quantization)

**Pros:**
- True parallelism (all agents run simultaneously)
- Fits on RTX 4090 24GB
- Can abliterate different behaviors per agent
- Fast inference (<100ms/token)

**Cons:**
- More complex orchestration
- Lower individual capability than 7B

**Hardware:**
- Local: RTX 4070 Ti 16GB (6 agents), RTX 4090 24GB (10 agents)
- Cloud: A6000 48GB (20 agents)

---

### Option B: Single Large Model

**Configuration:**
- 1× Qwen2.5-Coder-32B with role prompting

**Pros:**
- Highest capability (92% HumanEval)
- Simpler architecture

**Cons:**
- Sequential only (no parallelism)
- Requires 48-80GB GPU
- Can't abliterate per-agent

---

### Option C: Hybrid (Orchestrator + Specialists) ⭐ BEST FOR COMPLEX PROJECTS

**Configuration:**
- 1× 14B orchestrator (planning, architecture)
- 6× 3B specialists (execution)
- Total: 32GB (4-bit)

**Recommended Team:**
```yaml
Orchestrator: Qwen2.5-Coder-14B
  - Project planning
  - Architecture decisions
  - Task delegation

Specialists (Qwen2.5-Coder-3B each):
  - Frontend: React/Vue (abliterated: verbosity)
  - Backend: FastAPI/Django (abliterated: premature_optimization)
  - Testing: pytest/unittest (abliterated: test_reluctance)
  - Security: Auditing (amplified: security_paranoia)
  - Docs: Technical writing (abliterated: jargon_overload)
  - DevOps: Docker/K8s (abliterated: overengineering)
```

**Hardware:** A6000 48GB ($0.50-1.50/h on Vast.ai)

---

## 3. Framework Comparison

| Framework | CLI Native | Custom Models | Agent Specialization | Resource Efficiency | Bruno Integration |
|-----------|------------|---------------|---------------------|-------------------|-------------------|
| **Codebuff** | ✅ Yes | ✅ Yes | ✅ Multi-agent SDK | ✅ Excellent | ✅ **Perfect** |
| **CrewAI** | ❌ Python API | ✅ Yes | ✅ Role-based | ⚠️ Medium | ✅ Yes |
| **AutoGen** | ❌ Python API | ✅ Yes | ✅ Conversation-based | ⚠️ Medium | ✅ Yes |
| **Aider** | ✅ Yes | ✅ Yes | ❌ Single agent | ✅ Excellent | ✅ Yes |
| Claude Code | ✅ Yes | ❌ Claude only | ⚠️ Limited | ⚠️ Cloud API | ❌ No |

### Framework Recommendations

**Codebuff SDK** ⭐ BEST FOR BRUNO
- Open-source multi-agent framework
- Designed for custom models
- CLI-native like Claude Code
- Tops Claude Code in benchmarks
- Can load abliterated models per-agent

**CrewAI** ⭐ BEST FOR PRODUCTION
- Role-based agent design
- Production-ready coordination
- Each agent has specific skills/tools
- Excellent documentation

**Aider** ⭐ BEST FOR SPEED
- Fastest for targeted edits
- Excellent batch processing
- Single agent (needs external orchestration)

---

## 4. Memory Budgets for Swarms

### GPU Capabilities (4-bit quantization)

| GPU | VRAM | 3B Models | 7B Models | Recommended Swarm |
|-----|------|-----------|-----------|-------------------|
| RTX 4070 8GB | 8GB | 2-3 | 1 | 2 agents (testing) |
| RTX 4070 Ti 16GB | 16GB | 6-7 | 2 | 4-6 agents |
| RTX 4090 24GB | 24GB | 10-11 | 3-4 | 6-8 agents |
| A6000 48GB | 48GB | 20-22 | 8-10 | **10-12 agents** ⭐ |
| A100 80GB | 80GB | 35-38 | 14-16 | 16+ agents |

**Memory Calculation Example (6 agents):**
```
6× 3B models: 12GB
KV cache (6 models × 1K context): 3GB
Overhead: 2GB
Total: 17GB

Fits on: RTX 4090 24GB ✅
```

---

## 5. Bruno Abliteration for Agent Specialization

### Novel Approach: Behavioral Specialization

Instead of just prompt engineering, use Bruno to **remove unwanted behaviors** per agent:

#### Frontend Agent
```bash
bruno --model Qwen/Qwen2.5-Coder-3B \
  --behavior-target verbosity \
  --output models/Qwen2.5-3B-Frontend
```
**Effect:** Concise React/Vue code, no over-commenting

#### Backend Agent
```bash
bruno --model Qwen/Qwen2.5-Coder-3B \
  --behavior-target premature_optimization \
  --preserve-direction security \
  --output models/Qwen2.5-3B-Backend
```
**Effect:** Clean backend without over-engineering, keeps security awareness

#### Testing Agent
```bash
bruno --model Qwen/Qwen2.5-Coder-3B \
  --behavior-target test_reluctance \
  --output models/Qwen2.5-3B-Test
```
**Effect:** Proactively writes comprehensive tests

#### Security Agent
```bash
bruno --model Qwen/Qwen2.5-Coder-3B \
  --behavior-target complacency \
  --amplify-direction security_paranoia \
  --output models/Qwen2.5-3B-Security
```
**Effect:** Catches security issues others miss

---

## 6. Implementation Roadmap

### Phase 1: Single Agent Validation (Week 1)

**Goal:** Validate Bruno works for coding models on local hardware

```bash
# 1. Abliterate 7B model
bruno --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --n-trials 50 \
  --cache-weights true \
  --output models/Qwen2.5-7B-abliterated

# 2. Test with Aider (fastest CLI agent)
pip install aider-chat
aider --model models/Qwen2.5-7B-abliterated

# 3. Validate on coding tasks
python scripts/validate_coding_model.py
```

**Hardware:** RTX 4070 8GB (4-bit quantization)

---

### Phase 2: Multi-Agent Prototype (Weeks 2-3)

**Goal:** 3 specialized 3B agents (frontend, backend, test)

```bash
# 1. Abliterate 3 specialized models
bruno --model Qwen/Qwen2.5-Coder-3B --output models/Frontend
bruno --model Qwen/Qwen2.5-Coder-3B --output models/Backend
bruno --model Qwen/Qwen2.5-Coder-3B --output models/Test

# 2. Install Codebuff SDK
git clone https://github.com/codebuff/codebuff-sdk
pip install codebuff-sdk

# 3. Configure swarm
cat > swarm_config.yaml <<EOF
agents:
  - name: frontend
    model: ./models/Frontend
    role: "Frontend React developer"
  - name: backend
    model: ./models/Backend
    role: "Backend FastAPI developer"
  - name: test
    model: ./models/Test
    role: "QA engineer writing tests"
EOF

# 4. Run test task
codebuff swarm --config swarm_config.yaml \
  --task "Build authentication API with React UI"
```

**Hardware:** RTX 4090 24GB or cloud A6000 48GB

---

### Phase 3: Production Swarm (Weeks 4-6)

**Goal:** 6-agent production team with orchestrator

**Team Structure:**
```
swarm/
├── orchestrator/      # 14B - Project management
├── frontend/          # 3B - React/Vue specialist
├── backend/           # 3B - API/database specialist
├── testing/           # 3B - Test generation specialist
├── security/          # 3B - Security auditing specialist
├── docs/              # 3B - Documentation specialist
└── devops/            # 3B - Deployment specialist
```

**Framework:** CrewAI (production-ready)

```python
# swarm.py
from crewai import Agent, Task, Crew
from langchain.llms import Ollama

# Load abliterated models
agents = {
    "orchestrator": Agent(
        role="Project Manager",
        llm=Ollama(model="models/Qwen2.5-14B-Orchestrator"),
        goal="Plan and coordinate development"
    ),
    "frontend": Agent(
        role="Frontend Developer",
        llm=Ollama(model="models/Qwen2.5-3B-Frontend"),
        tools=[ReactAnalyzer(), CSSValidator()]
    ),
    # ... other agents
}

crew = Crew(agents=list(agents.values()), verbose=True)
```

**Hardware:** A6000 48GB (Vast.ai ~$1/hour)

---

### Phase 4: CLI Interface (Weeks 7-8)

**Goal:** Production CLI like Claude Code/Codebuff

```python
# bruno-swarm CLI
import click
from swarm import crew

@click.command()
@click.option('--task', help='Development task')
@click.option('--agents', default='all', help='frontend,backend,test')
@click.option('--interactive', is_flag=True)
def swarm(task, agents, interactive):
    """Run the Bruno AI developer swarm"""
    if interactive:
        # Interactive session like Claude Code
        run_interactive_mode(crew)
    else:
        # One-shot task execution
        result = crew.kickoff(inputs={'task': task})
        print(result)

if __name__ == '__main__':
    swarm()
```

**Usage:**
```bash
# One-shot task
bruno-swarm --task "Add user authentication to the app"

# Interactive mode (like Claude Code)
bruno-swarm --interactive

# Specific agents only
bruno-swarm --task "Fix security vulnerabilities" --agents security,backend
```

---

## 7. Cost Analysis

### Local Development (One-time GPU purchase)

| GPU | Cost | Agent Capacity | Use Case |
|-----|------|----------------|----------|
| RTX 4070 Ti 16GB | $799 | 6× 3B agents | Development/testing |
| RTX 4090 24GB | $1,599 | 10× 3B agents | Full swarm locally |

---

### Cloud Development (Per-hour costs on Vast.ai)

| GPU | VRAM | Cost/hour | Agent Capacity | Use Case |
|-----|------|-----------|----------------|----------|
| A6000 48GB | 48GB | $0.50-1.50 | 20× 3B or 8× 7B | Production swarm |
| A100 80GB | 80GB | $1.50-2.50 | 35× 3B or 14× 7B | Large scale |

**Monthly cost estimate (8 hours/day):**
- A6000: $120-360/month
- A100: $360-600/month

---

## 8. Quick Start Guide

### Option 1: Start Simple (RTX 4070 8GB)

**Goal:** Single abliterated 7B agent

```bash
# 1. Abliterate Qwen2.5-Coder-7B
bruno --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --n-trials 50 \
  --output models/Qwen2.5-7B-Coder

# 2. Use with Aider
pip install aider-chat
aider --model models/Qwen2.5-7B-Coder
```

**Cost:** Free (local GPU)

---

### Option 2: Multi-Agent (RTX 4090 24GB or A6000 48GB)

**Goal:** 4-agent dev team

```bash
# 1. Abliterate 4× 3B models (one per role)
for role in frontend backend test docs; do
  bruno --model Qwen/Qwen2.5-Coder-3B \
    --config configs/${role}.toml \
    --output models/${role}
done

# 2. Install CrewAI
pip install crewai

# 3. Configure and run swarm
python swarm.py --task "Build todo app with React + FastAPI"
```

**Cost:**
- Local: $1,599 (RTX 4090)
- Cloud: $1-2/hour (A6000)

---

### Option 3: Production Swarm (A6000 48GB or A100 80GB)

**Goal:** 6-8 agent production team

**Team:**
- Orchestrator (14B)
- 6× Specialists (3B each)

**Total VRAM:** 32GB (comfortable on 48GB GPU)

**Cost:** $1-2.50/hour cloud

---

## 9. Recommended Next Steps

### Immediate (This Week)

1. **Test single 7B agent**
   ```bash
   bruno --model Qwen/Qwen2.5-Coder-7B-Instruct \
     --n-trials 20 \
     --output models/test-agent

   aider --model models/test-agent
   ```

2. **Research Codebuff SDK**
   - Check if it supports local models
   - Test multi-agent capabilities
   - Evaluate vs CrewAI

### Short-term (Next 2 Weeks)

3. **Create 3 specialized 3B agents**
   - Frontend (concise code)
   - Backend (no over-engineering)
   - Testing (comprehensive coverage)

4. **Test CrewAI framework**
   ```bash
   pip install crewai
   # Build simple 3-agent crew
   ```

### Medium-term (Month 1-2)

5. **Rent A6000 48GB for full swarm**
   ```bash
   bruno-vast create A6000 1 --disk 400
   ```

6. **Build 6-agent production team**
   - Add orchestrator (14B)
   - Add security + docs + devops agents
   - Create CLI interface

7. **Integrate with Bruno CLI**
   ```bash
   bruno-swarm --task "Refactor authentication module"
   ```

---

## 10. Key Insights

### Small Models Are Viable

**Research shows:** 3B models achieve **96% of specialized model performance** through:
- Model merging techniques
- Task-specific fine-tuning
- Behavioral abliteration (Bruno's strength!)

**Implication:** You can run **6-10 specialized 3B agents** instead of 1 large model.

### Abliteration Enables True Specialization

**Beyond prompt engineering:**
- Remove verbosity from frontend agent
- Remove test avoidance from QA agent
- Amplify security paranoia in security agent
- Each agent has **different weights**, not just different prompts

### Hardware Sweet Spot

**For serious development:**
- **RTX 4090 24GB** ($1,599): 6-8 agents locally
- **A6000 48GB cloud** ($1/hour): 10-12 agents, production-ready

**ROI calculation:**
- A6000: 160 hours at $1/hour = $160/month
- RTX 4090: Pays for itself in 10 months
- **Recommendation:** Buy RTX 4090 if you'll use it >10 months

---

## 11. Framework-Specific Implementation

### Using CrewAI with Abliterated Models

```python
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Define abliterated models
frontend_llm = Ollama(model="qwen2.5-3b-frontend-abliterated")
backend_llm = Ollama(model="qwen2.5-3b-backend-abliterated")
test_llm = Ollama(model="qwen2.5-3b-test-abliterated")

# Create agents
frontend_dev = Agent(
    role='Frontend Developer',
    goal='Build responsive React components',
    backstory='Expert in React, TypeScript, Tailwind CSS',
    llm=frontend_llm,
    verbose=True
)

backend_dev = Agent(
    role='Backend Developer',
    goal='Create scalable FastAPI endpoints',
    backstory='Expert in FastAPI, PostgreSQL, async patterns',
    llm=backend_llm,
    verbose=True
)

qa_engineer = Agent(
    role='QA Engineer',
    goal='Write comprehensive test suites',
    backstory='Expert in pytest, coverage analysis, edge cases',
    llm=test_llm,
    verbose=True
)

# Define tasks
task1 = Task(
    description='Design authentication API endpoints',
    agent=backend_dev,
    expected_output='FastAPI router with auth endpoints'
)

task2 = Task(
    description='Create login form component',
    agent=frontend_dev,
    expected_output='React component with validation'
)

task3 = Task(
    description='Write tests for authentication flow',
    agent=qa_engineer,
    expected_output='pytest test suite'
)

# Create crew
crew = Crew(
    agents=[backend_dev, frontend_dev, qa_engineer],
    tasks=[task1, task2, task3],
    process=Process.sequential  # Or Process.hierarchical
)

# Execute
result = crew.kickoff()
print(result)
```

---

### Using Codebuff SDK

```python
from codebuff import Agent, Swarm

# Create swarm with abliterated models
swarm = Swarm(
    agents=[
        Agent(name="frontend", model="./models/Frontend", role="React developer"),
        Agent(name="backend", model="./models/Backend", role="FastAPI developer"),
        Agent(name="test", model="./models/Test", role="QA engineer"),
    ]
)

# Execute task
result = swarm.execute("Build user authentication system")
```

---

## 12. Recommended Starting Point

### **If you have RTX 4070 8GB (current setup):**

**Step 1:** Test single 7B agent
```bash
bruno --model Qwen/Qwen2.5-Coder-7B \
  --n-trials 20 \
  --output models/dev-agent

aider --model models/dev-agent
```

**Step 2:** If successful, rent A6000 48GB for multi-agent testing
```bash
bruno-vast create A6000 1 --disk 400
```

### **If you upgrade to RTX 4090 24GB:**

**Build 6-agent local swarm**
- No cloud costs
- Full development team
- 3B models for all agents

---

## Sources

- [10 Best Open-Source LLM Models (2025 Updated): Llama 4, Qwen 3 and DeepSeek R1](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Open Source AI vs Paid AI for Coding: The Ultimate 2026 Comparison Guide](https://aarambhdevhub.medium.com/open-source-ai-vs-paid-ai-for-coding-the-ultimate-2026-comparison-guide-ab2ba6813c1d)
- [Best Open Source LLM 2026 | Top Free AI Models](https://whatllm.org/blog/best-open-source-models-january-2026)
- [5 Open-Source Coding LLMs You Can Run Locally in 2025](https://www.labellerr.com/blog/best-coding-llms/)
- [The Best Open Source LLMs for Coding in 2026](https://www.siliconflow.com/articles/en/best-open-source-LLMs-for-coding)
- [Multi-task Code LLMs: Data Mix or Model Merge?](https://arxiv.org/html/2601.21115)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186)
- [Best AI Agent Frameworks in 2026: CrewAI vs. AutoGen vs. LangGraph](https://medium.com/@kia556867/best-ai-agent-frameworks-in-2026-crewai-vs-autogen-vs-langgraph-06d1fba2c220)
- [Codebuff goes Open Source, beats Claude Code, launches SDK](https://news.codebuff.com/p/codebuff-goes-open-source-beats-claude)
- [Top 5 CLI Coding Agents in 2026](https://dev.to/lightningdev123/top-5-cli-coding-agents-in-2026-3pia)
- [Ollama VRAM Requirements: Complete 2026 Guide to GPU Memory](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [Run 70B LLMs on a 4GB GPU: Complete Guide to Layer-Wise Inference](https://www.blog.brightcoding.dev/2026/01/13/run-70b-llms-on-a-4gb-gpu-the-complete-guide-to-layer-wise-inference-memory-optimization/)
