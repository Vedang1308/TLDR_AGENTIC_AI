# TLDR Agentic AI Project (CSE598)

This codebase contains the comprehensive implementation and experimentation framework for evaluating various Large Language Models (LLMs) on complex agentic tasks. We built our evaluation infrastructure around the [Tau Bench](https://github.com/sierra-research/tau-bench) environment, simulating real-world tool-agent-user interactions in the `retail` and `airline` domains. 

The project spans across multiple phases (from baseline evaluations in Phase 1 to advanced multi-agent Plan-Execute-Validate frameworks in Phase 3), including research-oriented approaches and specific hardware optimizations for Intel Gaudi HPUs and deployment on the SOL Supercomputer.

## Project Structure & Phases

### Phase 1: Baseline Single-Agent Evaluation
Phase 1 focuses on benchmarking monolithic LLMs utilizing standard reasoning and tool-calling paradigms.
- **Strategies Evaluated**: `react` (Reasoning and Acting), `act` (Acting only), and `tool-calling` (`fc`).
- **Setup**: Local vLLM instances mapping to specific ports (Agent on 8000, User on 8001).
- **Execution**: Scripts like `run_phase1_experiments.py` orchestrate the simulated environments. Smart resume logic automatically skips already-completed tasks to prevent data loss on interruption.
- **Auto Error Identification**: Utilizes an LLM to assign faults (user, agent, environment) and classify the type of fault when agents fail a task.

### Phase 3: Multi-Agent Plan-Execute-Validate (PEV) Framework
Phase 3 introduces a robust, multi-agent architecture powered by LangGraph, directly integrated into the tau-bench evaluation loop.
- **PEV Architecture**: Makes at least 3 distinct LLM calls (Planner -> Executor -> Validator) per single conversational turn. This enforces logical safety, prevents hallucinations, and avoids premature `transfer_to_human` failures by trading raw token efficiency for higher task reliability (`pass^k`).
- **SOL Supercomputer Deployment**: Designed to query local `vLLM` endpoints deployed on the SOL cluster, completely bypassing external OpenAI API rate limits.
- **Analysis Tools**: Automated generation of inner-monologue trajectory highlights (`generate_trajectory_highlights.py`) and comparative result plots against Phase 1 baselines (`generate_final_results.py`).

### Paper Approach: PEVAL Strategy & Gaudi Optimizations
This section bridges our custom agent frameworks with cutting-edge hardware acceleration.
- **Intel Gaudi HPU Fallback**: Explicit support for `habana-torch` and `optimum-habana` in the `gaudi_paperenv` and `vllm_gaudi` conda environments. 
- **Auto-Detection**: The scripts utilize `hl-smi` to detect Intel Gaudi accelerators. If NVIDIA GPUs aren't found, it seamlessly shifts to `bfloat16` HPU inference using `HABANA_VISIBLE_DEVICES`.
- **PEVAL (Research Strategy)**: Advanced research experiments evaluating performance and latency trade-offs on specialized accelerators.

## Getting Started

### 1. Environment & Model Services
Start the vLLM backends in separate terminals/tmux sessions. For Gaudi branches, ensure you have activated the correct conda environment (e.g., `vllm_gaudi`).
```bash
# Start User Simulator (Port 8001)
./cse598_project/phase3/scripts/start_user_model.sh

# Start Agent Backend (Port 8000 - Select desired parameter size)
./cse598_project/phase3/scripts/start_vllm_8b.sh
```

### 2. Run Evaluations
Run the orchestrator scripts depending on the phase you are testing.
**Phase 1 (Baselines):**
```bash
python cse598_project/phase1/scripts/run_phase1_experiments.py --domain all --strategy all
```
**Phase 3 (Multi-Agent PEV):**
```bash
python cse598_project/phase3/scripts/run_phase3_experiments.py --domain all --model "Qwen/Qwen3-8B" --strategy "multi-agent"
```

### 3. Generate Reports
Parse the output JSON files to view metrics, errors, and trajectory logs.
```bash
python cse598_project/phase3/scripts/generate_final_results.py --results-dir "results/phase3"
```

## Branch Context
All three primary branches (`main`, `vedang`, `vedang-gaudi`) share this unified documentation to give a full picture of the repository capabilities. While the core frameworks remain consistent, ensure you are using the hardware-specific scripts (e.g., Gaudi auto-detection) available in your current checked-out branch.
