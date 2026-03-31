#!/bin/bash
# start_gaudi_agent.sh
# Optimized for Intel Gaudi (HL-225) 96GB per AIP
# Model: Qwen3-14B (Default)
# TP: 2 (Optimized for 14B/32B/72B)

# Defaults
MODEL_PATH="Qwen/Qwen3-14B"
PORT=8224
TP=2

# Improved argument parsing:
# If first arg doesn't start with -- it's the model path
if [[ $1 != --* ]] && [ ! -z "$1" ]; then
    MODEL_PATH=$1
    shift
fi

echo "--- Starting PEVAL Agent Server on Gaudi HPU ---"
echo "Model: $MODEL_PATH | TP: $TP"
echo "Extra Args: $@"

# Cleanup
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

# vLLM HPU-Optimized Command
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-agent \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --block-size 128 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enforce-eager \
    --trust-remote-code \
    "$@"
