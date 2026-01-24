#!/bin/bash
MODEL="Qwen/Qwen3-8B-Instruct"
PORT=8000

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

echo "Starting vLLM server for $MODEL on port $PORT..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --confirm-runner-role-alias-check \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95
