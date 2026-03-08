#!/bin/bash
# Load CUDA driver for vLLM compilation if on SOL node
module load cuda-12.4.1-gcc-12.1.0 2>/dev/null || true

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate phase3_env

# Navigate to project directory
cd ~/AGENTIC_AI/TLDR_AGENTIC_AI/cse598_project/phase3 || exit

echo "Starting Agent Model (Qwen3-32B) on Port 8000..."
vllm serve "Qwen/Qwen3-32B" --port 8000 --host 0.0.0.0
