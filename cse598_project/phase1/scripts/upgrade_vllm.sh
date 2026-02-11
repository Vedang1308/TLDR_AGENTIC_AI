#!/bin/bash
# scripts/upgrade_vllm.sh
# Upgrades vLLM and installs bitsandbytes for quantization support

source ~/.bashrc 2>/dev/null
conda activate tau-bench

echo "Upgrading vLLM to latest version..."
pip install --upgrade vllm

echo "Installing bitsandbytes for quantization support..."
pip install bitsandbytes

echo "Verifying installation..."
pip show vllm bitsandbytes

echo "Upgrade complete. Please try running start_user_model.sh again."
