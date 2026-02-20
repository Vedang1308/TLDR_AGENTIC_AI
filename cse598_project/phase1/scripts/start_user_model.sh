MODEL="Qwen/Qwen3-32B"
ALIAS="User-Qwen3-32B"
PORT=8001

export HF_HUB_OFFLINE=1

# Force cache to scratch to avoid Disk Full issues
export HF_HOME=/scratch/vgaduput/huggingface_cache
export XDG_CACHE_HOME=/scratch/vgaduput/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

# Use venv python relative to this script
PYTHON_EXEC="python3"

# echo "Starting vLLM server for User Agent ($MODEL as $ALIAS) on port $PORT..."
# $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --served-model-name $ALIAS \
#     --trust-remote-code \
#     --port $PORT \
#     --dtype bfloat16 \
#     --max-model-len 16384 \
#     --max-num-batched-tokens 16384 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.55 \
#     --quantization bitsandbytes \
#     --load-format bitsandbytes \
#     --enable-prefix-caching
#     # Note: reduced gpu utilization if running alongside another model, 
#     # but practically we might need sequential execution if single GPU.


# CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model "Qwen/Qwen3-32B" \
#     --served-model-name "User-Qwen3-32B" \
#     --port $PORT \
#     --dtype bfloat16 \
#     --max-model-len 32768 \
#     --gpu-memory-utilization 0.90 \
#     --enable-prefix-caching \
#     --trust-remote-code

# quicker
# CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model "Qwen/Qwen3-32B" \
#     --served-model-name "User-Qwen3-32B" \
#     --port 8001 \
#     --dtype bfloat16 \
#     --max-model-len 32768 \
#     --gpu-memory-utilization 0.90 \
#     --enable-prefix-caching \
#     --trust-remote-code \
#     --disable-log-requests


# CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
#     --model "Qwen/Qwen3-32B" \
#     --served-model-name "User-Qwen3-32B" \
#     --port 8001 \
#     --dtype bfloat16 \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.9 \
#     --enable-prefix-caching \
#     --trust-remote-code \
#     --disable-log-requests


CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-32B" \
    --port 8001 \
    --dtype bfloat16 \
    --quantization fp8 \
    --served-model-name "User-Qwen3-32B" \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests