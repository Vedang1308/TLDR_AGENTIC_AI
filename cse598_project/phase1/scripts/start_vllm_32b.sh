#!/bin/bash
MODEL="Qwen/Qwen3-32B"
PORT=8000

# Force cache to scratch
export HF_HOME=/scratch/vgaduput/huggingface_cache
export XDG_CACHE_HOME=/scratch/vgaduput/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

# Use venv python relative to this script
PYTHON_EXEC="python3"

echo "Starting vLLM server for Agent ($MODEL) on port $PORT..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --port $PORT \
    --dtype bfloat16 \
    --served-model-name "gpt-4-32k" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --trust-remote-code \
    --max-num-seqs 16


# CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --port $PORT \
#     --dtype bfloat16 \
#     --served-model-name "gpt-4-32k" \
#     --max-model-len 32768 \
#     --max-num-seqs 128 \
#     --max-num-batched-tokens 32768 \
#     --gpu-memory-utilization 0.95 \
#     --enable-prefix-caching \
#     --async-scheduling \
#     --disable-log-requests