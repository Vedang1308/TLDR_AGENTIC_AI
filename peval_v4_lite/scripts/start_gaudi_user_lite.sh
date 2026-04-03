#!/bin/bash
# start_gaudi_user.sh
# Optimized for Intel Gaudi (HL-225) - 96GB per AIP
# Model: Qwen2.5-72B-Instruct
# TP: 2 (Tensor Parallel - 64 heads / 2)

# Defaults
MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
PORT=8225
TP=2

# Improved argument parsing:
# If first arg doesn't start with -- it's the model path
if [[ $1 != --* ]] && [ ! -z "$1" ]; then
    MODEL_PATH=$1
    shift
fi

echo "--- Starting PEVAL User Simulator on Gaudi HPU ---"
echo "Model: $MODEL_PATH | TP: $TP"
echo "Extra Args: $@"

# Cleanup
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

# vLLM HPU-Optimized Command
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen-72b-simulator \
    --host 127.0.0.1 \
    --port $PORT \
    --block-size 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code \
    "$@"
