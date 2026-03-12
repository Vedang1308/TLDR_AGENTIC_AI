#!/bin/bash
MODEL="Qwen/Qwen3-32B"
PORT=8001

# Load CUDA compiler if on SOL
module load cuda-12.4.1-gcc-12.1.0 2>/dev/null || true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate phase3_gaudi_env

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

# ── DEVICE AUTO-DETECTION ───────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    DEVICE_TYPE="cuda"
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
elif command -v hl-smi &> /dev/null; then
    DEVICE_TYPE="hpu"
    GPU_COUNT=$(hl-smi -q | grep -c "HL-225")
else
    DEVICE_TYPE="cpu"
    GPU_COUNT=0
fi

if [ "$DEVICE_TYPE" = "cuda" ]; then
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "Detected $GPU_COUNT GPUs → DUAL-GPU mode: user model on GPU 1 (dedicated)"
        export CUDA_VISIBLE_DEVICES=1
        GPU_MEM_UTIL=0.90
    else
        echo "Detected $GPU_COUNT GPU → SINGLE-GPU mode: user model shares GPU 0 with agent"
        GPU_MEM_UTIL=0.50
    fi
    DTYPE="float16"
    EXTRA_VLLM_ARGS="--quantization bitsandbytes --load-format bitsandbytes"
elif [ "$DEVICE_TYPE" = "hpu" ]; then
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "Detected $GPU_COUNT HPUs → DUAL-HPU mode: user model on HPU 1 (dedicated)"
        export HABANA_VISIBLE_DEVICES=1
        GPU_MEM_UTIL=0.90
    else
        echo "Detected $GPU_COUNT HPU → SINGLE-HPU mode: user model shares HPU 0 with agent"
        GPU_MEM_UTIL=0.50
    fi
    DTYPE="bfloat16"
    EXTRA_VLLM_ARGS="--device hpu"
    # Load Habana modules if on SOL
    module load habana-torch 2>/dev/null || true
else
    echo "No accelerator detected, falling back to CPU (unsupported for high performance)"
    GPU_MEM_UTIL=0.10
    DTYPE="float32"
    EXTRA_VLLM_ARGS=""
fi
# ────────────────────────────────────────────────────────────────────────────

echo "Starting vLLM server for User Simulator ($MODEL) on port $PORT (Device: $DEVICE_TYPE)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name "User-Qwen3-32B" \
    --trust-remote-code \
    --port $PORT \
    --dtype $DTYPE \
    --max-model-len 30000 \
    --max-num-batched-tokens 30000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    $EXTRA_VLLM_ARGS
