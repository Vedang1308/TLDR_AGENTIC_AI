#!/bin/bash
# start_gaudi_user.sh
# Optimized for Intel Gaudi (HL-225) - 96GB per AIP
# Model: Qwen2.5-72B-Instruct
# TP: 2 (Tensor Parallel - 64 heads / 2)

# Defaults
MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
PORT=8225
TP=2

# Environment Redirects (Avoids home directory disk space errors)
export DATA_DIR="/data"
export HF_DATA_CACHE="$DATA_DIR/huggingface_cache"
export TMPDIR="$DATA_DIR/tmp"

mkdir -p $HF_DATA_CACHE $TMPDIR

# Nuclear Option: Symlink Surgery
# This forces every Hugging Face process to write to /data regardless of env vars
HF_HOME_DEFAULT="$HOME/.cache/huggingface"

if [ ! -L "$HF_HOME_DEFAULT" ]; then
    echo "--- Performing HF Cache Surgery: Redirecting $HF_HOME_DEFAULT to $HF_DATA_CACHE ---"
    # Ensure parent dir exists
    mkdir -p "$(dirname "$HF_HOME_DEFAULT")"
    if [ -d "$HF_HOME_DEFAULT" ]; then
        # If it's a real directory, move content and delete it
        echo "Moving existing cache content to $HF_DATA_CACHE..."
        mv "$HF_HOME_DEFAULT"/* "$HF_DATA_CACHE/" 2>/dev/null
        rm -rf "$HF_HOME_DEFAULT"
    fi
    # Create the symlink
    ln -s "$HF_DATA_CACHE" "$HF_HOME_DEFAULT"
fi

export HF_HOME="$HF_DATA_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_DATA_CACHE"
export TRANSFORMERS_CACHE="$HF_DATA_CACHE"
export HF_HUB_OFFLINE=1

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
    --download-dir $HF_HOME \
    "$@"
