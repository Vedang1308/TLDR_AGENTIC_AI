#!/bin/bash
MODEL="Qwen/Qwen2.5-72B-Instruct"
PORT=8001

# Load Habana environment if on SOL
module load habana 2>/dev/null || true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate vllm_gaudi

# Force cache to scratch
export HF_HOME=/scratch/vavaghad/huggingface_cache
export XDG_CACHE_HOME=/scratch/vavaghad/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Kill existing process on port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

echo "Starting vLLM server for Research Paper User Simulator ($MODEL) on port $PORT (GAUDI)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name "Qwen/Qwen2.5-72B-Instruct-User" \
    --trust-remote-code \
    --port $PORT \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager
