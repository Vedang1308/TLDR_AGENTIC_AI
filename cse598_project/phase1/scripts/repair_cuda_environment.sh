#!/bin/bash
set -e

echo "Repairing environment for NVIDIA A100 (CUDA)..."

# 1. Uninstall HPU/Gaudi specific packages
echo "Uninstalling Intel Gaudi packages..."
pip uninstall -y vllm intel-extension-for-pytorch habana-torch-plugin habana-torch-dataloader

# 2. Uninstall Torch and NVIDIA packages to ensure a clean slate
# (The previous aggressive uninstall might have left mismatched versions)
echo "Cleaning up PyTorch and NVIDIA libs..."
pip uninstall -y torch torchvision torchaudio triton
pip list --format=freeze | grep "^nvidia-" | cut -d= -f1 | xargs -r pip uninstall -y

# 3. Install standard vLLM (which pulls compatible torch & cuda libs)
echo "Installing standard vLLM for CUDA..."
# Install existing latest vLLM or a specific compatible version. 
# 0.6.3 is a stable recent version, or just 'vllm' for latest.
pip install vllm

# 4. Verify installation
echo "Verifying CUDA availability..."
python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

echo "Environment repair complete. Please try running the start scripts now."
