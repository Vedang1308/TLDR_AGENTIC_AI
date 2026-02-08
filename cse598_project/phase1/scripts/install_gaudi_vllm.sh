#!/bin/bash
set -e

echo "Starting vLLM installation for Intel Gaudi..."

# Uninstall conflicting packages
echo "Cleaning up existing PyTorch and vLLM installations..."
pip uninstall -y vllm torch torchvision torchaudio intel-extension-for-pytorch

# Directory for cloning
BUILD_DIR="vllm-gaudi-build"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

echo "Cloning HabanaAI/vLLM-fork..."
# Using --depth 1 to speed up clone
git clone --depth 1 -b habana_main https://github.com/HabanaAI/vLLM-fork.git $BUILD_DIR

cd $BUILD_DIR

echo "Installing build dependencies..."
# Install HPU specific requirements 
if [ -f "requirements-hpu.txt" ]; then
    echo "Installing requirements from requirements-hpu.txt..."
    pip install -r requirements-hpu.txt
else
    echo "Warning: requirements-hpu.txt not found, falling back to requirements.txt"
    pip install -r requirements.txt
    # Fallback usually needs explicit IPEX install if not in requirements.txt
    pip install intel-extension-for-pytorch
fi

echo "Building vLLM for Gaudi (this may take a few minutes)..."
export VLLM_TARGET_DEVICE=hpu
pip install .

echo "Cleaning up..."
cd ..
rm -rf $BUILD_DIR

echo "Installation complete!"
echo "Please verify with: pip list | grep vllm"
echo "Expected version should usually have a +gaudi suffix or be a recent version."
