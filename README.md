# TLDR Agentic AI Project - Phase 1 (Vedang Branch)

This codebase contains the implementation and experimentation scripts for evaluating various Large Language Models (LLMs) on agentic tasks, particularly using a setup based on the Tau Bench. 

## Overview

This repository branch (`vedang`) focuses on running experiments across different domains such as `retail` and `airline` to evaluate LLM reasoning and tool-calling capabilities. It utilizes a local vLLM setup to serve models efficiently for both the Agent and the User simulation.

## Structure

- **cse598_project/**: Main project directory containing the experiment framework.
  - **phase1/**: Contains scripts and data for Phase 1 of the project.
    - **scripts/**: Scripts to start the vLLM servers and run experiments.
      - `run_phase1_experiments.py`: Main script to orchestrate the experiments. Supports multiple strategies (`react`, `act`, `fc`) and models (e.g., `Qwen/Qwen3-4B` up to `32B`).
      - `start_vllm_*.sh`: Scripts to spin up local vLLM instances on specific ports.
      - `analyze_results.py` / `generate_report.py`: Utilities for parsing and summarizing experiment results.
    - **results/**: Output logs and trace files from agent interactions in JSON format.
    - **few_shot_data/**: Contains few-shot examples for the specific domains (`MockAirlineDomainEnv-few_shot.jsonl`, etc.).

## Setup and Execution

1. **Start the Models**: Launch the user and agent models using the provided shell scripts in `cse598_project/phase1/scripts/`. The models are mapped to specific local ports (e.g., 8000 for Agent, 8001 for User).
2. **Run Experiments**: Execute `run_phase1_experiments.py` to start the trials.
   ```bash
   python cse598_project/phase1/scripts/run_phase1_experiments.py --domain all --strategy all
   ```

## Key Features

- **Smart Resume Logic**: Automatically detects completed tasks and skips them to save time on interruptions.
- **Port Mapping**: Uses `TAUBENCH_PORT_MAP` to route requests to local inference servers efficiently without hitting external API rate limits.
