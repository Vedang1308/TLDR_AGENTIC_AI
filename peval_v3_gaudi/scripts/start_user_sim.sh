#!/bin/bash
MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
PORT=8225

# Environment Redirects
export DATA_DIR="/data"
export HF_HOME="$DATA_DIR/huggingface_cache"
export HF_HUB_OFFLINE=1
mkdir -p $HF_HOME

# Check if port is in use
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

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
    # If single GPU, share with agent. If multiple, use GPU 1.
    if [ "$ACCEL_COUNT" -gt 1 ]; then
        echo "Using dedicated GPU 1 for User Sim"
        export CUDA_VISIBLE_DEVICES=1
        MEM_UTIL=0.90
        TP_SIZE=1 
    else
        echo "Sharing GPU 0 with Agent (Limited Memory)"
        MEM_UTIL=0.45
        TP_SIZE=1
    fi
    EXTRA_ARGS=""
elif [ "$DEVICE_TYPE" = "hpu" ]; then
    echo "Detected $ACCEL_COUNT HPUs (Intel Gaudi Cluster)"
    # For User Sim on Gaudi, we use HPU 0 (sharing with agent if TP matches, or dedicated if count permits)
    # Simulator typically runs with TP=2 or TP=4 depending on available HPUs
    if [ "$ACCEL_COUNT" -ge 8 ]; then TP_SIZE=2; else TP_SIZE=2; fi
    DTYPE="bfloat16"
    EXTRA_ARGS="--device hpu"
    MEM_UTIL=0.90
else
    echo "No accelerator detected."
    DTYPE="float32"
    TP_SIZE=1
    MEM_UTIL=0.10
    EXTRA_ARGS=""
fi
# ────────────────────────────────────────────────────────────────────────────

echo "Starting vLLM User Simulator ($MODEL_PATH) on port $PORT (Device: $DEVICE_TYPE)..."

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-72b-simulator \
    --host 127.0.0.1 \
    --port $PORT \
    --dtype $DTYPE \
    --max-model-len 8192 \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $MEM_UTIL \
    --trust-remote-code \
    --download-dir $HF_HOME \
    $EXTRA_ARGS
