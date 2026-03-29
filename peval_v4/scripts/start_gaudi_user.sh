#!/bin/bash
# start_gaudi_user.sh
# Optimized for Qwen2.5-72B on ASU SOL

MODEL_PATH=${1:-"Qwen/Qwen2.5-72B-Instruct"}
PORT=8223
TP=4  # 64 heads / 4 = 16 heads per card (Mathematically perfect)

echo "--- Starting PEVAL User Simulator on Gaudi HPU ---"
echo "Model: $MODEL_PATH | TP: $TP"

# Cleanup any zombie processes
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

# vLLM HPU Command
# We use --gpu-memory-utilization 0.85 to leave room for system overhead
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-72b-simulator \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --block-size 128 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code
