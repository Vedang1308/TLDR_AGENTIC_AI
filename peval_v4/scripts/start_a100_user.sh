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

# Memory Management for Single vs Multi GPU
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "!!! Detected Single-GPU mode: Lowering memory footprint for sharing !!!"
    GPU_MEM_UTIL=0.30
    MAX_LEN=4096
else
    GPU_MEM_UTIL=0.95
    MAX_LEN=16384
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen2.5-72b-simulator \
    --port $PORT \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_LEN \
    --enforce-eager \
    --trust-remote-code
