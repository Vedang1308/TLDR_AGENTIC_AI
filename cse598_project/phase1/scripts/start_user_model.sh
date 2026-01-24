MODEL="Qwen/Qwen3-32B"
ALIAS="User-Qwen3-32B"
PORT=8001

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
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 
    # Note: reduced gpu utilization if running alongside another model, 
    # but practically we might need sequential execution if single GPU.
