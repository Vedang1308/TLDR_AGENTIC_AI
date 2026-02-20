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
# Recommended changes for vllm_32b script:
# CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --trust-remote-code \
#     --port $PORT \
#     --dtype bfloat16 \
#     --served-model-name "gpt-4-32k" \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.9 \
#     --enable-prefix-caching \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --max-num-seqs 32


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

# Optimized for high-concurrency throughput
# CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --port $PORT \
#     --dtype bfloat16 \
#     --max-model-len 16384 \
#     --max-num-seqs 32 \
#     --served-model-name "gpt-4-32k" \
#     --max-num-batched-tokens 4096 \
#     --enable-chunked-prefill \
#     --gpu-memory-utilization 0.9 \
#     --enable-prefix-caching \
#     --disable-log-requests \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \


CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-32B" \
    --port 8000 \
    --dtype bfloat16 \
    --quantization fp8 \
    --served-model-name "gpt-4-32k" \
    --max-model-len 32768 \
    --max-num-seqs 5 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes 