#!/bin/bash
# scripts/cleanup_gpu.sh

echo "Showing current GPU usage:"
nvidia-smi

echo "Killing all python and vllm processes for user $USER..."
pkill -u $USER -f "vllm" 2>/dev/null
pkill -u $USER -f "python" 2>/dev/null
pkill -u $USER -f "api_server" 2>/dev/null

echo "Waiting for processes to exit..."
sleep 3

echo "GPU usage after cleanup:"
nvidia-smi

echo "Done. You can now start the models."
