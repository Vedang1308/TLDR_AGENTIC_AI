# TLDR Agentic AI Project

This codebase contains the implementation and experimentation scripts for evaluating various Large Language Models (LLMs) on agentic tasks, particularly using a setup based on the Tau Bench. 

## Overview

This repository branch (`vedang-gaudi`) focuses on running experiments across different domains such as `retail` and `airline` to evaluate LLM reasoning and tool-calling capabilities. **Crucially, this branch includes optimizations and configurations specifically designed for Intel Gaudi HPUs.**

## Structure

- **cse598_project/**: Main project directory containing the experiment framework.
  - **phase1/**: Contains baseline scripts and data for Phase 1.
  - **phase3/**: Contains updated experiment setups with explicit Gaudi HPU fallback logic.
  - **paper_approach/**: Contains PEVAL (research paper) strategy experiments with HPU auto-detection.

## Gaudi specific execution

1. **Environment Setup**: Ensure you are using the `gaudi_paperenv` or `vllm_gaudi` conda environments, which include the necessary `habana-torch` and optimum-habana dependencies.
2. **Start the Models**: Launch the user and agent models using the provided shell scripts (e.g. `start_vllm_*.sh`). The scripts automatically set environment variables like `HABANA_VISIBLE_DEVICES` and run inference using `bfloat16` precision.
3. **Run Experiments**: Execute the python scripts to start trials. The system is designed to prioritize NVIDIA GPUs if available, but will fallback seamlessly to Intel Gaudi HPUs via auto-detection logic using `hl-smi`.
   ```bash
   python cse598_project/paper_approach/run_paper_experiments.py --domain all --strategy all
   ```

## Key Features

- **HPU Auto-Detection**: Scripts natively detect `hl-smi` and configure device mapping for Gaudi accelerators.
- **Port Mapping**: Uses `TAUBENCH_PORT_MAP` to route requests to local inference servers efficiently without hitting external API rate limits.
- **Smart Resume Logic**: Automatically detects completed tasks and skips them to save time on interruptions.
