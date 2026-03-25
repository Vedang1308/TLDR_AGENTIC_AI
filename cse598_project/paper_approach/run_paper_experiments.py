import argparse
import subprocess
import os
import time
import sys
import concurrent.futures
import json
import glob

# Ensure we can import from the paper_approach and tau_bench folders
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cse598_project.paper_approach.utils import setup_paper_env, get_paper_port_map

# ── GPU AUTO-DETECTION ─────────────────────────────────────────────────────────
def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            return len(gpus)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0

def get_gpu_config(gpu_count: int) -> dict:
    if gpu_count >= 2:
        return {
            "mode": "DUAL-GPU (parallel research benchmarks)",
            "max_workers": 5,
            "max_concurrency": 20,
        }
    else:
        return {
            "mode": "SINGLE-GPU (sequential research benchmarks)",
            "max_workers": 1,
            "max_concurrency": 1,
        }
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(domain, model, strategy, user_model, trial, max_concurrency=1):
    """
    Runs a single research paper experiment configuration.
    """
    print(f"\n--- [PAPER EXP]: {domain} | {model} | {strategy} | Trial {trial} ---")
    
    # Setup environment
    setup_paper_env(model, user_model)
    port_map = get_paper_port_map()
    
    # Output path
    model_safe = model.replace("/", "_")
    output_dir = f"results/paper_approach/{domain}/{model_safe}/{strategy}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct Command
    cmd = [
        sys.executable, "cse598_project/paper_approach/run.py", # Local override
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", "llm",
        "--max-concurrency", str(max_concurrency),
        "--seed", str(trial),
        "--log-dir", output_dir,
        "--agent-strategy", "multi-agent" # This triggers the override in run.py
    ]
    
    # Force the PEVAL strategy specifically
    env = os.environ.copy()
    env["AGENT_STRATEGY"] = strategy # [ReAct, FC, Self-Reflection]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"--- [SUCCESS]: {output_dir} ---")
    except subprocess.CalledProcessError as e:
        print(f"--- [FAILED]: {output_dir} | {e} ---")

def main():
    parser = argparse.ArgumentParser(description="Run Research Paper (PEVAL) Experiments")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", nargs="+", default=["ReAct", "FC", "Self-Reflection"])
    parser.add_argument("--model", nargs="+", default=[
        "Qwen/Qwen3-4B-Instruct",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen2.5-72B-Instruct"
    ])
    parser.add_argument("--user-model", type=str, default="Qwen/Qwen2.5-72B-Instruct-User")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    
    # Auto-detect GPU count
    gpu_count = detect_gpu_count()
    gpu_cfg = get_gpu_config(gpu_count)
    
    max_workers = args.max_workers if args.max_workers is not None else gpu_cfg["max_workers"]
    concurrency = args.concurrency if args.concurrency is not None else gpu_cfg["max_concurrency"]
    
    print(f"\n{'='*60}")
    print(f"  GPU AUTO-DETECTION (RESEARCH MODE)")
    print(f"  Detected GPUs : {gpu_count}")
    print(f"  Execution Mode: {gpu_cfg['mode']}")
    print(f"  Max Workers   : {max_workers}")
    print(f"  Max Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    
    experiments = [
        (domain, model, strategy, args.user_model, trial, concurrency)
        for domain in domains
        for model in args.model
        for strategy in args.strategy
        for trial in range(args.trials)
    ]
    
    print(f"Starting {len(experiments)} research trials...")
    
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_experiment, *exp) for exp in experiments]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    else:
        for exp in experiments:
            run_experiment(*exp)

if __name__ == "__main__":
    main()
