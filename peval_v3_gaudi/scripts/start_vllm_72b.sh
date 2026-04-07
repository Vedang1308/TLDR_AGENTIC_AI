#!/bin/bash
MODEL="Qwen/Qwen2.5-72B-Instruct"
PORT=8224

# Activate environment
eval "$(conda shell.bash hook)"
conda activate vllm_gaudi || conda activate vllm

# Force cache to /data (Consistent and safer than home)
export HF_HOME=/data/huggingface_cache
export XDG_CACHE_HOME=/data/xdg_cache
export HF_HUB_OFFLINE=1
mkdir -p $HF_HOME $XDG_CACHE_HOME

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

# ── DEVICE AUTO-DETECTION ───────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    DEVICE_TYPE="cuda"
    ACCEL_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
elif command -v hl-smi &> /dev/null; then
    DEVICE_TYPE="hpu"
    ACCEL_COUNT=$(hl-smi -q | grep -c "HL-225")
else
    DEVICE_TYPE="cpu"
    ACCEL_COUNT=0
fi

if [ "$DEVICE_TYPE" = "cuda" ]; then
    echo "Detected $ACCEL_COUNT GPUs (A100/H100 Cluster)"
    DTYPE="bfloat16"
    # 72B on A100-80GB needs TP-4 or TP-8
    # Using GPUs 0,1,2,3 for Agent
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    TP_SIZE=4
    EXTRA_ARGS=""
    MEM_UTIL=0.90
elif [ "$DEVICE_TYPE" = "hpu" ]; then
    echo "Detected $ACCEL_COUNT HPUs (Intel Gaudi HL-225 Cluster)"
    # PARTITION 1: Agent Server on HPUs 0,1,2,3 (TP-4)
    export HABANA_VISIBLE_DEVICES=0,1,2,3
    DTYPE="bfloat16"
    TP_SIZE=4
    EXTRA_ARGS="--device hpu --enable-auto-tool-choice --tool-call-parser hermes"
    MEM_UTIL=0.90
else
    echo "No accelerator detected, falling back to CPU (Extreme Latency!)"
    DTYPE="float32"
    TP_SIZE=1
    EXTRA_ARGS=""
    MEM_UTIL=0.10
fi
# ────────────────────────────────────────────────────────────────────────────

echo "Starting vLLM agent server ($MODEL) on port $PORT (Device: $DEVICE_TYPE, TP: $TP_SIZE)..."

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --port $PORT \
    --dtype $DTYPE \
    --max-model-len 32768 \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $MEM_UTIL \
    $EXTRA_ARGS
