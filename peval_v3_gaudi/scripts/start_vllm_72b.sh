#!/bin/bash
MODEL="Qwen/Qwen2.5-72B-Instruct"
PORT=8224

# Activate environment
eval "$(conda shell.bash hook)"
conda activate vllm_gaudi

# Force cache to /data (Consistent with User Sim)
export HF_HOME=/data/huggingface_cache
export XDG_CACHE_HOME=/data/xdg_cache
export HF_HUB_OFFLINE=1
mkdir -p $HF_HOME $XDG_CACHE_HOME

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

echo "Starting vLLM server for Agent ($MODEL) on port $PORT (Full Gaudi Node)..."

# 72B requires TP-8 to fit well and perform on Gaudi
export HABANA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --device hpu \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
