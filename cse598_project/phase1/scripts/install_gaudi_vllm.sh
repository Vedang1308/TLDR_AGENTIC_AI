#!/bin/bash
set -e

echo "Starting vLLM installation for Intel Gaudi..."

# Uninstall incompatible vLLM
echo "Uninstalling incompatible vLLM..."
pip uninstall -y vllm

# Directory for cloning
BUILD_DIR="vllm-gaudi-build"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

echo "Cloning HabanaAI/vLLM-fork..."
# Using --depth 1 to speed up clone
git clone --depth 1 -b habana_main https://github.com/HabanaAI/vLLM-fork.git $BUILD_DIR

cd $BUILD_DIR

echo "Installing build dependencies..."
# Install HPU specific requirements if they exist, otherwise standard
if [ -f "requirements-hpu.txt" ]; then
    pip install -r requirements-hpu.txt
else
    pip install -r requirements.txt
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
