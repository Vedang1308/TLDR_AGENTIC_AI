#!/bin/bash
MODEL="Qwen/Qwen3-14B"
PORT=8000

# Load CUDA compiler if on SOL
module load cuda-12.4.1-gcc-12.1.0 2>/dev/null || true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate phase3_env

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

echo "Starting vLLM server for Agent ($MODEL) on port $PORT..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 30000 \
    --max-num-batched-tokens 30000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
