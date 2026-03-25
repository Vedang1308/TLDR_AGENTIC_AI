#!/bin/bash
# Script to launch the PEVAL Agent Model on NVIDIA A100 GPU(s)
# 
# Usage:
#   For Single A100 (4B, 8B, 14B): ./start_a100_agent.sh Qwen/Qwen2.5-14B-Instruct 1
#   For Multi A100 (32B, 72B)    : ./start_a100_agent.sh Qwen/Qwen2.5-32B-Instruct 4

MODEL_PATH=${1:-"Qwen/Qwen2.5-32B-Instruct"}
NUM_GPUS=${2:-2} # Default to 2 GPUs for the 32B model
PORT=8222

echo "--- Starting PEVAL Agent Model on $NUM_GPUS x A100 GPUs ---"
echo "Model: $MODEL_PATH"
echo "Serving on Port: $PORT"

# Core vLLM arguments for A100
# - tensor-parallel-size distributes the model across multiple GPUs
# - gpu-memory-utilization prevents OOMs when handling long contexts
# - max-model-len ensures we have enough space for the PEVAL Memory Kernel
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-32b-agent \
    --port $PORT \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code
