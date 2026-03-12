import argparse
import subprocess
import os
import time
import sys
import concurrent.futures

# Ensure the parent phase3 directory is in Python's path so we can resolve the local tau_bench copy
phase3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if phase3_root not in sys.path:
    sys.path.insert(0, phase3_root)

import json
import glob
from tau_bench.envs import get_env


# ── GPU AUTO-DETECTION ─────────────────────────────────────────────────────────
# Runs `nvidia-smi` to count available GPUs.
# Returns the integer GPU count (0 if nvidia-smi is not available / no GPUs).
# This drives two key decisions:
#   1-GPU  → sequential experiments (max_workers=1), lower concurrency,
#             user model shares GPU 0 (same port 8001 but same device).
#   2+ GPU → parallel experiments (max_workers=5), higher concurrency,
#             user model pinned to GPU 1 via CUDA_VISIBLE_DEVICES in the
#             start_user_model.sh script (already configured that way).
# ──────────────────────────────────────────────────────────────────────────────
def detect_gpu_count() -> tuple[int, str]:
    """
    Returns (count, type) where type is 'cuda' or 'hpu'.
    """
    # 1. Try NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            if gpus:
                return len(gpus), "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. Try Intel Gaudi (HPU)
    try:
        result = subprocess.run(
            ["hl-smi", "-q"], # Using -q for count detection
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # hl-smi output parsing is slightly different; if it runs, we usually have HPUs.
            # Based on user's hl-smi output, we can count HL-225 entries or use a simpler check
            # Real-world: hl-smi -q gives detailed info.
            # For simplicity, if hl-smi works and shows AIPs, we count them.
            # Based on user provide output, there were 8 HPUs.
            count = result.stdout.count("HL-225") or result.stdout.count("AIP")
            if count > 0:
                return count, "hpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0, "none"


def get_gpu_config(gpu_count: int, device_type: str) -> dict:
    """
    Returns the execution configuration based on available GPU/HPU count.
    """
    if device_type == "hpu":
        if gpu_count >= 2:
            return {
                "mode": f"DUAL-HPU (parallel, {gpu_count} detected)",
                "max_workers": 5,
                "max_concurrency": 20,
                "user_gpu_device": 1,
            }
        else:
            return {
                "mode": "SINGLE-HPU (sequential)",
                "max_workers": 1,
                "max_concurrency": 1,
                "user_gpu_device": 0,
            }

    if gpu_count >= 2:
        return {
            "mode": "DUAL-GPU (parallel)",
            "max_workers": 5,
            "max_concurrency": 20,
            "user_gpu_device": 1,
        }
    else:
        return {
            "mode": "SINGLE-GPU (sequential, unchanged)",
            "max_workers": 1,
            "max_concurrency": 1,
            "user_gpu_device": 0,
        }


def get_existing_completed_tasks(output_path):
    completed_ids = set()
    search_pattern = os.path.join(output_path, "*.json")
    print(f"Scanning for completed tasks in: {search_pattern}")
    files = glob.glob(search_pattern)
    print(f"Found {len(files)} log files.")
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "task_id" in item:
                            info = item.get("info", {})
                            if info and "error" in info:
                                continue  # Treat crashed tasks as incomplete
                            completed_ids.add(item["task_id"])
        except Exception:
            pass
    return completed_ids


def run_experiment(domain, model, strategy, user_model, user_strategy, trial,
                   start_index=0, max_concurrency=1, results_dir="results/phase3"):
    # Force absolute results path anchored to phase3_root
    # This ensures consistency regardless of where the script is called from
    if not os.path.isabs(results_dir):
        results_dir = os.path.abspath(os.path.join(phase3_root, results_dir))

    print(f"Running Experiment: Domain={domain}, Model={model}, Strategy={strategy}, "
          f"Trial={trial}, ResumeFrom={start_index}, Concurrency={max_concurrency}")
    print(f"Results Directory: {results_dir}")
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)

    # Port map: agent model always on 8000, user model always on 8001
    port_map = {
        "Qwen/Qwen3-4B": 8000,
        "Qwen/Qwen3-8B": 8000,
        "Qwen/Qwen3-14B": 8000,
        "Qwen/Qwen3-32B": 8000,
        "User-Qwen3-32B": 8001,
        "gpt-4o": 8001,
    }

    # Export Port Map to environment so get_env and LiteLLM can find local servers
    os.environ["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    os.environ["OPENAI_API_BASE"] = f"http://localhost:{port_map.get(model, 8000)}/v1"
    os.environ["LITELLM_API_BASE"] = f"http://localhost:{port_map.get(model, 8000)}/v1"

    # Smart Resume Logic: only run tasks not yet completed
    completed_ids = get_existing_completed_tasks(output_path)

    try:
        # Use task_index=0 to avoid random out-of-bounds errors in environment init
        temp_env = get_env(domain, user_strategy=user_strategy, user_model=user_model,
                           user_provider="openai", task_split="test", task_index=0)
        total_tasks = len(temp_env.tasks)
        print(f"Total tasks in dataset: {total_tasks}. Completed so far: {len(completed_ids)}")
    except Exception as e:
        # Fallback safety net (now less likely to be needed after base.py fix)
        total_tasks = 50 if domain == "airline" else 115
        print(f"Warning: Auto-detection failed ({e}). Using safety fallback: {total_tasks}")

    needed_ids = [
        str(i) for i in range(total_tasks)
        if i >= start_index and str(i) not in completed_ids and i not in completed_ids
    ]

    if not needed_ids:
        print(f"All {total_tasks} tasks completed! Skipping.")
        return

    print(f"Resuming/Retrying {len(needed_ids)} tasks: {needed_ids[:5]}...")

    env = os.environ.copy()
    env["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    env["OPENAI_API_KEY"] = "sk-1234"  # Bypass LiteLLM validation for local vLLM

    # Pass model info to multi-agent nodes (used by get_llm() in nodes.py)
    env["AGENT_MODEL_NAME"] = model
    env["AGENT_API_BASE"] = f"http://localhost:{port_map.get(model, 8000)}/v1"

    # Map strategy name to tau_bench strategy + reasoning mode env var
    base_strategy = strategy
    if strategy.startswith("multi-agent"):
        base_strategy = "multi-agent"
        if strategy == "multi-agent-react":
            env["AGENT_REASONING_MODE"] = "react"
        elif strategy == "multi-agent-act":
            env["AGENT_REASONING_MODE"] = "act"
        else:
            env["AGENT_REASONING_MODE"] = "fc"

    run_py = os.path.join(phase3_root, "run.py")
    cmd = [
        sys.executable, run_py,
        "--agent-strategy", "tool-calling" if base_strategy == "fc" else base_strategy,
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", user_strategy,
        "--max-concurrency", str(max_concurrency),
        "--seed", str(trial),
        "--log-dir", output_path,
        "--task-ids"
    ] + needed_ids

    try:
        # Run from the phase3 root so run.py can resolve the local tau_bench copy
        subprocess.run(cmd, check=True, env=env, cwd=phase3_root)
        print(f"Experiment finished successfully. Results in {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Experiments")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", choices=[
        "react", "act", "fc",
        "multi-agent", "multi-agent-act", "multi-agent-react", "multi-agent-fc", "all"
    ], default="multi-agent-fc")
    parser.add_argument("--model", type=str, help="Specific model to run (e.g., Qwen/Qwen3-32B)")
    parser.add_argument("--user-model", type=str, default="User-Qwen3-32B")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    # These two are OVERRIDDEN by GPU auto-detection unless --force-workers / --force-concurrency are passed
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Override auto-detected parallel workers (default: auto from GPU count)")
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="Override auto-detected task concurrency (default: auto from GPU count)")
    args = parser.parse_args()

    # ── AUTO-DETECT GPU/HPU COUNT AND SET EXECUTION CONFIG ─────────────────────
    gpu_count, device_type = detect_gpu_count()
    gpu_cfg = get_gpu_config(gpu_count, device_type)

    # Allow manual overrides, otherwise use auto-detected values
    max_workers = args.max_workers if args.max_workers is not None else gpu_cfg["max_workers"]
    max_concurrency = args.max_concurrency if args.max_concurrency is not None else gpu_cfg["max_concurrency"]

    print(f"\n{'='*60}")
    print(f"  DEVICE AUTO-DETECTION")
    print(f"  Detected Type : {device_type.upper()}")
    print(f"  Detected Count: {gpu_count}")
    print(f"  Execution Mode: {gpu_cfg['mode']}")
    print(f"  Max Workers   : {max_workers}  (parallel experiment threads)")
    print(f"  Max Concurrency: {max_concurrency}  (concurrent tasks per experiment)")
    print(f"{'='*60}\n")
    # ─────────────────────────────────────────────────────────────────────────

    os.environ["OPENAI_API_KEY"] = "sk-1234"

    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]

    if args.strategy == "all":
        strategies = ["fc", "act", "react", "multi-agent-fc", "multi-agent-act", "multi-agent-react"]
    else:
        strategies = [args.strategy]

    models = [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ] if not args.model else [args.model]

    experiments = [
        (domain, model, strategy, args.user_model, "llm", trial, args.start_index, max_concurrency)
        for domain in domains
        for model in models
        for strategy in strategies
        for trial in range(args.trials)
    ]

    if max_workers > 1:
        print(f"Running {len(experiments)} experiments in parallel (max_workers={max_workers})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_experiment, *exp)
                for exp in experiments
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Experiment thread failed: {e}")
    else:
        print(f"Running {len(experiments)} experiments sequentially...")
        for exp in experiments:
            run_experiment(*exp)


if __name__ == "__main__":
    main()
