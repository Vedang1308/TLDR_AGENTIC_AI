MODEL="Qwen/Qwen3-32B"
ALIAS="User-Qwen3-32B"
PORT=8001

# Force cache to scratch/svijay46
export HF_HOME=/scratch/svijay46/huggingface_cache
export XDG_CACHE_HOME=/scratch/svijay46/xdg_cache
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing process..."
    kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t)
fi

# Use venv python relative to this script
PYTHON_EXEC="python3"

echo "Starting vLLM server for User Agent ($MODEL as $ALIAS) on port $PORT..."
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name $ALIAS \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.75 \
    --swap-space 16
