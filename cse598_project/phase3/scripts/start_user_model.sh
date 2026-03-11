#!/bin/bash
MODEL="Qwen/Qwen3-32B"
PORT=8001

# Load CUDA compiler if on SOL
module load cuda-12.4.1-gcc-12.1.0 2>/dev/null || true

# Activate Python venv
source ~/agents/TLDR_AGENTIC_AI/phase3_env/bin/activate

# Force cache to scratch to prevent home directory quota limits
export HF_HOME=/scratch/vavaghad/huggingface_cache
export XDG_CACHE_HOME=/scratch/vavaghad/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Check if port is in use and kill it
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

echo "Starting vLLM server for User Simulator ($MODEL) on port $PORT..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name "User-Qwen3-32B" \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 30000 \
    --max-num-batched-tokens 30000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
