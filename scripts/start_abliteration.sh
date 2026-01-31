#!/bin/bash
# Run this after SSHing into the Vast.ai instance:
# ssh -p 12406 root@ssh1.vast.ai

# 1. Create the config file
cat > /workspace/config.toml << 'EOF'
n_trials = 100
batch_size = 8
max_batch_size = 8
max_response_length = 150

[good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:500]"
column = "text"

[bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:300]"
column = "text"

[good_evaluation_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[500:700]"
column = "text"

[bad_evaluation_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[300:450]"
column = "text"
EOF

echo "Config created:"
cat /workspace/config.toml

# 2. Start heretic abliteration on Qwen2.5-Coder-32B-Instruct
echo ""
echo "Starting abliteration on Qwen2.5-Coder-32B-Instruct..."
cd /workspace && heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models
