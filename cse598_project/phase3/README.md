# Phase 3: Multi-Agent Plan-Execute-Validate Framework

This directory contains the self-contained setup for Phase 3 of the Agentic AI Project. 
The LangGraph-based PEV (Plan-Execute-Validate) Multi-Agent architecture has been integrated directly into a cloned `tau_bench` repository to preserve original evaluation rules while enabling internal loops.

## Deployment on SOL Supercomputer

Do **NOT** try to run these locally. They are designed to query the local `vLLM` servers configured for the SOL cluster.

### 1. Start the vLLM Backends
Open separate terminal sessions or use `tmux`/`screen` on SOL to spawn the required LLM endpoints.

**User Simulator Service (Port 8001)**
```bash
./scripts/start_user_model.sh
```

**Agent Service (Port 8000)**
Pick the model you want to evaluate and run its respective bash script:
```bash
./scripts/start_vllm_4b.sh
# or
./scripts/start_vllm_8b.sh
# or
./scripts/start_vllm_14b.sh
# or
./scripts/start_vllm_32b.sh
```

### 2. Run the Multi-Agent Evaluation
Open a third terminal, activate your python environment, and execute the experiments logic:

```bash
# This will run the multi-agent strategy against the currently running vLLM Agent endpoint
# Make sure to change --model to match whichever vLLM script you ran!
python3 scripts/run_phase3_experiments.py \
    --domain "all" \
    --model "Qwen/Qwen3-4B" \
    --strategy "multi-agent" \
    --trials 1 \
    --start-index 0
```
> **Note**: The script is hardcoded to bypass OpenAI Authentication and route traffic over port 8000 (Agent) and 8001 (User) directly. If the script hangs, ensure your `start_vllm_*.sh` processes are fully active and printing `"Uvicorn running on ..."`

### 3. Generate Trajectory Highlights
After the completion of a JSON evaluation file, run this to scrape the unique `pev_node_logs` inner monologues for the report:

```bash
python3 scripts/generate_trajectory_highlights.py \
    --results-dir "results/phase3" \
    --output "trajectory_highlights.md"
```

### 4. Generate Final Results Table & Plots
To fulfill Phase 3 Requirement #2 (Results Table & Plots vs Baselines):
```bash
python3 scripts/generate_final_results.py --results-dir "results/phase3"
```
This will output `results/phase3_final_results_table.csv` and `results/phase3_method_vs_baseline_plot.png` for you to include in your report.

### Troubleshooting
- `Invalid agent strategy`: Ensure you are running `python3 scripts/run_phase3_experiments.py` from THIS `phase3` directory. It uses the `tau_bench` copy located in `phase3/tau_bench/run.py` which was modified to accept `--agent-strategy multi-agent`.
- `Connection Error`: `vLLM` hasn't finished booting up yet or the port is in use. Check your parallel terminals.

---
## Phase 3 Trade-Off Notes (For Your Report)
When writing your Phase 3 report, you may notice the **Total Token Cost** for the Multi-Agent architecture is strictly higher than the Phase 1 monolithic baselines. 
This is expected behavior. The Plan-Execute-Validate Graph makes at least 3 distinct LLM calls (Planner -> Executor -> Validator) per single conversational turn to enforce logical safety, prevent hallucinations, and avoid premature `transfer_to_human` failures. We are explicitly trading raw token efficiency for higher `pass^k` task reliability.
