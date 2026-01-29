# Running Phase 1 on Sol Supercomputer

This guide details how to run the Qwen3-8B (Agent) vs Qwen3-32B (User) experiments on Sol.

## 1. Setup on Sol

Login to Sol and navigate to your workspace.

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd TLDR_AGENTIC_AI

# OR Pull the latest changes (including the new Slurm script)
git pull origin kris-main
```

## 2. Environment Setup

You need a Conda environment with `vllm` and the project dependencies.

```bash
# Load Mamba (faster Conda) and CUDA (REQUIRED for vLLM)
module load mamba/latest
module load cuda/12.1  # Adjust version if needed (e.g., cuda/11.8)

# Verify CUDA_HOME is set
echo $CUDA_HOME

# Create Environment (skip if already created)
mamba create -n tau-bench python=3.10 -y
conda activate tau-bench


# Install Dependencies
cd cse598_project/phase1
pip install -e .
pip install vllm
pip install bitsandbytes>=0.46.1
```

## 3. Submit Job

Submit the provided Slurm script. This script requests 2 GPUs (one for each model) and runs the experiments.

```bash
cd cse598_project/phase1
sbatch scripts/run_sol_8b.slurm
```

## 4. Monitor Progress

Check the job status and logs.

```bash
# Check queue
squeue -u $USER

# Tail the log file (replace <job-id> with your actual job ID)
tail -f slurm-<job-id>.out
```

## 5. View Results

Results will be saved in `cse598_project/phase1/results/phase1/`.

- **Structure**: `results/phase1/<domain>/<model>/<strategy>/trial_<n>/`
- **Output**: JSON trajectory files.

## 6. Troubleshooting

### "ValueError: (16.00 GiB KV cache is needed, which is larger than the available KV cache memory)"
This means the context length is too large for the available GPU memory.
**Fix**:
1.  Ensure you have the latest script version:
    ```bash
    git pull origin kris-main
    ```
2.  Run `bash scripts/start_user_model.sh` again. It should show `gpu_memory_utilization: 0.9` (or 0.6 for shared GPU) and `max_model_len: 45000`.

### "ValueError: Free memory on device ... is less than desired"
This means other processes (likely zombie vLLM servers) are still holding GPU memory.
**Fix**:
1.  Kill all vLLM processes:
    ```bash
    pkill -f vllm
    ```
2.  Check `nvidia-smi` (if available) to ensure memory is free.
3.  Run the Slurm script (recommended) or start scripts again.
