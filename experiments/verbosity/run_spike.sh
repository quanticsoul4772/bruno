#!/bin/bash
# Verbosity Direction Spike Experiment
#
# This script runs the complete spike experiment to test if a
# "verbosity direction" can be extracted from LLMs.
#
# Usage: ./run_spike.sh <model-name>
# Example: ./run_spike.sh meta-llama/Llama-3.1-8B-Instruct

set -e

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
MODEL_NAME=$(basename "$MODEL")
OUTPUT_DIR="./${MODEL_NAME}-concise"

echo "========================================"
echo "Verbosity Direction Spike Experiment"
echo "========================================"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 0: Create datasets
echo "[Step 0] Creating local datasets..."
python experiments/verbosity/load_local_dataset.py
echo ""

# Step 1: Measure baseline verbosity
echo "[Step 1] Measuring baseline verbosity..."
python experiments/verbosity/eval_verbosity.py \
    --model "$MODEL" \
    --num-prompts 30 \
    --output "experiments/verbosity/baseline_${MODEL_NAME}.json"
echo ""

# Step 2: Run heretic with verbosity config
echo "[Step 2] Extracting verbosity direction..."
echo "This will run heretic with the verbosity config."
echo "When finished, save the model to: $OUTPUT_DIR"
echo ""
echo "Press Enter to start heretic, or Ctrl+C to abort."
read -r

# Copy config
cp experiments/verbosity/config.verbosity.toml config.toml

# Run heretic
heretic --model "$MODEL" --auto-select --auto-select-path "$OUTPUT_DIR"

echo ""
echo "[Step 3] Measuring modified model verbosity..."
python experiments/verbosity/eval_verbosity.py \
    --original "$MODEL" \
    --modified "$OUTPUT_DIR" \
    --num-prompts 30 \
    --output "experiments/verbosity/comparison_${MODEL_NAME}.json"

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Results saved to:"
echo "  - experiments/verbosity/baseline_${MODEL_NAME}.json"
echo "  - experiments/verbosity/comparison_${MODEL_NAME}.json"
echo ""
echo "Modified model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the comparison metrics"
echo "  2. Chat with the modified model to test capability"
echo "  3. Try composing with refusal direction"
