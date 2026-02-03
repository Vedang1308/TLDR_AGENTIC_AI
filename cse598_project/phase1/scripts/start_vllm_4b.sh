#!/bin/bash
MODEL="Qwen/Qwen3-4B"
PORT=8000

# Force cache to scratch
export HF_HOME=/scratch/vgaduput/huggingface_cache
export XDG_CACHE_HOME=/scratch/vgaduput/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Allow long model length for vLLM
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

# Use venv python relative to this script
PYTHON_EXEC="python3"

echo "Starting vLLM server for Agent ($MODEL) on port $PORT..."
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 65536 \
    --hf-overrides '{"rope_scaling": {"type": "linear", "factor": 2.0}}' \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.40
