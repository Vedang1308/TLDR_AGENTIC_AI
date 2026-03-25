#!/bin/bash
# Script to launch the PEVAL Agent Model on NVIDIA A100 GPU(s)
# 
# Usage:
#   For Single A100 (4B, 8B, 14B): ./start_a100_agent.sh Qwen/Qwen3-14B 1
#   For Multi A100 (32B, 72B-Instruct): ./start_a100_agent.sh Qwen/Qwen3-32B 4

MODEL_PATH=${1:-"Qwen/Qwen3-32B"}
NUM_GPUS=${2:-2} # Default to 2 GPUs for the 32B model
PORT=8222

echo "--- Starting PEVAL Agent Model on $NUM_GPUS x A100 GPUs ---"
echo "Model: $MODEL_PATH"
echo "Serving on Port: $PORT"

# Core vLLM arguments for A100
# - tensor-parallel-size distributes the model across multiple GPUs
# - gpu-memory-utilization prevents OOMs when handling long contexts
# - max-model-len ensures we have enough space for the PEVAL Memory Kernel
# Memory Management for Single vs Multi GPU
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "!!! Detected Single-GPU mode: Lowering memory footprint for sharing !!!"
    GPU_MEM_UTIL=0.30
    MAX_LEN=4096
else
    GPU_MEM_UTIL=0.90
    MAX_LEN=16384
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-32b-agent \
    --port $PORT \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_LEN \
    --enforce-eager \
    --trust-remote-code
