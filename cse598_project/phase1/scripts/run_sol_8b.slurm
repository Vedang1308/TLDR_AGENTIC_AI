#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p general
#SBATCH -o slurm-%j.out
#SBATCH --job-name=tau-8b-exp

# Load modules (adjust based on Sol's specific modules, these are common defaults)
module load mamba/latest 2>/dev/null || module load anaconda/2023.07 2>/dev/null
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null
source ~/.bashrc 2>/dev/null

# Activate Environment
# Users should create this env beforehand: 
# mamba create -n tau-bench python=3.10 -y && mamba activate tau-bench && pip install -e . && pip install vllm
conda activate tau-bench

echo "Starting Job on $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# Set Paths
BASE_DIR=$(pwd)
SCRIPT_DIR="$BASE_DIR/scripts"

# Define Ports (Must match run_phase1_experiments.py)
PORT_USER=8001
PORT_AGENT=8000

# Cleanup function
cleanup() {
    echo "Stopping vLLM servers..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT

# ---------------------------------------------------------
# Start User Model (Qwen3-32B) on GPU 0
# ---------------------------------------------------------
echo "Starting User Model (Qwen3-32B) on GPU 0..."
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --served-model-name User-Qwen3-32B \
    --trust-remote-code \
    --port $PORT_USER \
    --dtype float16 \
    --max-model-len 45000 \
    --max-num-batched-tokens 45000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.60 &
PID_USER=$!

# ---------------------------------------------------------
# Start Agent Model (Qwen3-8B) on GPU 0 (Shared) or 1
# ---------------------------------------------------------
echo "Starting Agent Model (Qwen3-8B)..."
# Note: If using 1 GPU, we share it. If 2, we could use 1.
# Assuming single A100-80GB scenario based on recent updates.
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --trust-remote-code \
    --port $PORT_AGENT \
    --dtype float16 \
    --max-model-len 45000 \
    --max-num-batched-tokens 45000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.35 &
PID_AGENT=$!

# ---------------------------------------------------------
# Wait for Servers
# ---------------------------------------------------------
echo "Waiting for servers to start..."
wait_for_server() {
    local port=$1
    local name=$2
    for i in {1..30}; do
        if curl -s http://localhost:$port/v1/models > /dev/null; then
            echo "$name is ready on port $port!"
            return 0
        fi
        sleep 10
        echo "Waiting for $name..."
    done
    echo "Timeout waiting for $name"
    return 1
}

wait_for_server $PORT_USER "User Model"
wait_for_server $PORT_AGENT "Agent Model"

# ---------------------------------------------------------
# Run Experiments
# ---------------------------------------------------------
echo "Running Experiments..."
# Ensure Python finds the tau_bench package
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

# Define Port Map for the script
export TAUBENCH_PORT_MAP='{"Qwen/Qwen3-8B": 8000, "User-Qwen3-32B": 8001}'

# Note: Using exact model names as they appear on HuggingFace/vLLM
# Updated model names in CLI to match the vLLM arguments
python scripts/run_phase1_experiments.py \
    --model Qwen/Qwen3-8B \
    --user-model User-Qwen3-32B \
    --domain all \
    --strategy all \
    --trials 5

echo "Experiments Completed!"
