#!/bin/bash
MODEL="Qwen/Qwen3-4B"
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

# echo "Starting vLLM server for Agent ($MODEL) on port $PORT..."
# $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --trust-remote-code \
#     --port $PORT \
#     --dtype bfloat16 \
#     --max-model-len 16384 \
#     --max-num-batched-tokens 35248 \
#     --enable-prefix-caching \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.30 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --max-num-seqs 5 


# Optimized Qwen3-4B Settings
CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-4B" \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --trust-remote-code