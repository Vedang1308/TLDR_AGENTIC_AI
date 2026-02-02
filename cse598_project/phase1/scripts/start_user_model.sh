MODEL="Qwen/Qwen3-32B"
ALIAS="User-Qwen3-32B"
PORT=8001

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

echo "Starting vLLM server for User Agent ($MODEL as $ALIAS) on port $PORT..."
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name $ALIAS \
    --trust-remote-code \
    --port $PORT \
    --dtype float16 \
    --max-model-len 40960 \
    --max-num-batched-tokens 40960 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.50 \
    --quantization bitsandbytes \
    --load-format bitsandbytes 
    # Note: reduced gpu utilization if running alongside another model, 
    # but practically we might need sequential execution if single GPU.
