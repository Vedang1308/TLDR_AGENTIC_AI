#!/bin/bash
# start_gaudi_agent.sh
# Optimized for Intel Gaudi (HL-225) 96GB per AIP
# Model: Qwen3-14B
# TP: 1

MODEL_PATH=${1:-"Qwen/Qwen3-14B"}
PORT=8222
TP=1

echo "--- Starting PEVAL Agent Model on Gaudi HPU ---"
echo "Model: $MODEL_PATH | TP: $TP"

# Cleanup
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 5

# vLLM HPU-Optimized Command
# --block-size 128 is critical for HPU MME utilization
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-14b-agent \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --block-size 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --enforce-eager \
    --trust-remote-code
