#!/bin/bash
# Script to launch the PEVAL User Simulator Model on NVIDIA A100 GPU(s)
# 
# Usage:
#   For Multi A100 (72B): ./start_a100_user.sh Qwen/Qwen2.5-72B-Instruct 4

MODEL_PATH=${1:-"Qwen/Qwen2.5-72B-Instruct"}
NUM_GPUS=${2:-4} # 72B realistically needs 4x A100s 
PORT=8223

echo "--- Starting PEVAL User Simulator on $NUM_GPUS x A100 GPUs ---"
echo "Model: $MODEL_PATH"
echo "Serving on Port: $PORT"

# --- [CLEANUP] Kill any existing process on this port to prevent 'Address already in use' ---
echo "--- Cleaning up any existing process on Port $PORT... ---"
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2  # Allow CUDA context to fully release

# Memory Management for Single vs Multi GPU
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "!!! Detected Single-GPU mode: Enabling BitsAndBytes 8-bit Quantization !!!"
    GPU_MEM_UTIL=0.40
    MAX_LEN=2048      # User Simulator only needs short context
    QUANT="bitsandbytes"
else
    GPU_MEM_UTIL=0.95
    MAX_LEN=16384
    QUANT="none"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen2.5-72b-simulator \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_LEN \
    --quantization $QUANT \
    --enforce-eager \
    --swap-space 0 \
    --trust-remote-code
