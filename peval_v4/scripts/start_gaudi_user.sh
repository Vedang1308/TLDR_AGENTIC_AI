#!/bin/bash
# start_gaudi_user.sh
# Optimized for Intel Gaudi (HL-225) - 96GB per AIP
# Model: Qwen2.5-72B-Instruct
# TP: 3 (Tensor Parallel)

MODEL_PATH=${1:-"Qwen/Qwen2.5-72B-Instruct"}
PORT=8223
TP=3

echo "--- Starting PEVAL User Simulator on Gaudi HPU ---"
echo "Model: $MODEL_PATH | TP: $TP"

# Cleanup
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 5

# vLLM HPU-Optimized Command
# --block-size 128 is critical for HPU MME utilization
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-72b-simulator \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --block-size 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code
