#!/bin/bash
MODEL="Qwen/Qwen3-32B"
PORT=8001

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

# ── GPU AUTO-DETECTION ───────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "Detected $GPU_COUNT GPUs → DUAL-GPU mode: user model on GPU 1 (dedicated)"
    export CUDA_VISIBLE_DEVICES=1
    GPU_MEM_UTIL=0.90
else
    echo "Detected $GPU_COUNT GPU → SINGLE-GPU mode: user model shares GPU 0 with agent"
    GPU_MEM_UTIL=0.50
fi
# ────────────────────────────────────────────────────────────────────────────

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
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --quantization bitsandbytes \
    --load-format bitsandbytes
