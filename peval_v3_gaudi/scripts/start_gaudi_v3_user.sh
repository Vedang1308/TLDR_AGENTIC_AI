#!/bin/bash

# Configuration for 72B User Simulator on Gaudi HPU
MODEL_ID=${1:-"Qwen/Qwen2.5-72B-Instruct"}
TP_SIZE=4
MEM_UTIL=0.90

echo "--- Starting PEVAL v3 User Simulator (Phase 3 Logic) on Gaudi HPU ---"
echo "Model: $MODEL_ID | TP: $TP_SIZE"

# --- SYMLINK SURGERY: Force HF to high-capacity /data partition ---
mkdir -p /data/huggingface_cache /data/tmp
mkdir -p ~/.cache
if [ ! -L ~/.cache/huggingface ]; then
    echo "[SURGERY] Redirecting ~/.cache/huggingface to /data/huggingface_cache..."
    rm -rf ~/.cache/huggingface
    ln -s /data/huggingface_cache ~/.cache/huggingface
fi

# Set crucial environment overrides
export HF_HOME=/data/huggingface_cache
export VLLM_CACHE=/data/huggingface_cache
export TMPDIR=/data/tmp
export HF_XET_CACHE=/data/huggingface_cache/xet

# Launch vLLM Server (Gaudi fork specifically)
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --port 8225 \
    --host 0.0.0.0 \
    --block-size 128 \
    --max-num-seqs 64 \
    --device hpu \
    --served-model-name "qwen-72b-simulator" 2>&1 | tee user_model.log
