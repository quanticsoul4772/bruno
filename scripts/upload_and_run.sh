#!/bin/bash
# Upload all 7 agent models to a Vast.ai instance and configure CrewAI swarm
# Usage: bash scripts/upload_and_run.sh <SSH_HOST> <SSH_PORT> [--flat]
# Example: bash scripts/upload_and_run.sh ssh7.vast.ai 11280
# Example: bash scripts/upload_and_run.sh ssh7.vast.ai 11280 --flat

HOST=${1:-ssh7.vast.ai}
PORT=${2:-11280}
MODE=${3:---hierarchical}
PROJECT_DIR="C:/Development/Projects/heretic"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"

echo "==========================================="
echo "BRUNO AI DEVELOPER SWARM DEPLOYMENT"
echo "Instance: $HOST:$PORT"
echo "Mode: $MODE"
echo "==========================================="

# Step 1: Remote setup
echo "=== Step 1: Remote setup ==="
$SCP "$PROJECT_DIR/scripts/remote_setup.sh" root@$HOST:/workspace/
$SSH "bash /workspace/remote_setup.sh"

# Step 2: Upload 6 specialist models (3B each, ~5.8GB)
# Upload sequentially -- parallel scp saturates bandwidth
MODELS=("Frontend-3B" "Backend-3B" "Test-3B" "Security-3B" "Docs-3B" "DevOps-3B")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    STEP=$((i + 2))
    echo "=== Step $STEP: Uploading $MODEL (5.8GB) ==="
    $SCP -r "$PROJECT_DIR/models/$MODEL" root@$HOST:/workspace/
done

# Step 3: Upload orchestrator if not flat mode
if [ "$MODE" != "--flat" ]; then
    echo "=== Step 8: Uploading Orchestrator-14B ==="
    $SCP -r "$PROJECT_DIR/models/Orchestrator-14B" root@$HOST:/workspace/
fi

# Step 4: Create Ollama Modelfiles and load all agents
echo "=== Creating Ollama models ==="

# Specialist Modelfiles with CrewAI-tuned settings
$SSH 'cat > /workspace/Modelfile.frontend << "EOF"
FROM /workspace/Frontend-3B
SYSTEM "You are a Frontend Developer specializing in React, TypeScript, and Tailwind CSS. Write clean, concise code without over-engineering."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

$SSH 'cat > /workspace/Modelfile.backend << "EOF"
FROM /workspace/Backend-3B
SYSTEM "You are a Backend Developer specializing in FastAPI, PostgreSQL, and async patterns. Focus on clean architecture without premature optimization."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

$SSH 'cat > /workspace/Modelfile.test << "EOF"
FROM /workspace/Test-3B
SYSTEM "You are a QA Engineer specializing in pytest, coverage analysis, and edge case testing. Proactively write comprehensive tests for all code."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

$SSH 'cat > /workspace/Modelfile.security << "EOF"
FROM /workspace/Security-3B
SYSTEM "You are a Security Engineer specializing in vulnerability assessment, OWASP Top 10, and secure coding patterns. Identify security issues aggressively."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

$SSH 'cat > /workspace/Modelfile.docs << "EOF"
FROM /workspace/Docs-3B
SYSTEM "You are a Technical Writer specializing in API documentation, README files, and developer guides. Write clear, concise documentation."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

$SSH 'cat > /workspace/Modelfile.devops << "EOF"
FROM /workspace/DevOps-3B
SYSTEM "You are a DevOps Engineer specializing in Docker, CI/CD pipelines, and infrastructure as code. Write practical configs without overengineering."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'

# Load all specialist models
for agent in frontend backend test security docs devops; do
    echo "=== Loading ${agent}-agent into Ollama ==="
    $SSH "ollama create ${agent}-agent -f /workspace/Modelfile.${agent}"
done

# Load orchestrator if not flat mode
if [ "$MODE" != "--flat" ]; then
    $SSH 'cat > /workspace/Modelfile.orchestrator << "EOF"
FROM /workspace/Orchestrator-14B
SYSTEM "You are a Senior Software Architect. Plan development tasks, design system architecture, delegate work to specialists, and review code quality."
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
EOF'
    echo "=== Loading orchestrator-agent into Ollama ==="
    $SSH "ollama create orchestrator-agent -f /workspace/Modelfile.orchestrator"
fi

# Step 5: Upload and run swarm
echo "=== Uploading swarm script ==="
$SCP "$PROJECT_DIR/examples/seven_agent_swarm.py" root@$HOST:/workspace/
$SCP "$PROJECT_DIR/.env.crewai" root@$HOST:/workspace/

echo "=== Verifying Ollama models ==="
$SSH "ollama list"

echo "==========================================="
echo "DEPLOYMENT COMPLETE"
echo ""
echo "To run the swarm:"
echo "  $SSH \"cd /workspace && python seven_agent_swarm.py --task 'Build user auth system'\""
if [ "$MODE" == "--flat" ]; then
    echo "  (Running in flat mode -- no orchestrator)"
else
    echo "  $SSH \"cd /workspace && python seven_agent_swarm.py --task 'Build user auth system' --flat\""
fi
echo "==========================================="
