
import argparse
import subprocess
import os
import time
import sys
import os

# Ensure the parent phase3 directory is in Python's path so we can resolve the local tau_bench copy
phase3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if phase3_root not in sys.path:
    sys.path.insert(0, phase3_root)

import json
import glob
from tau_bench.envs import get_env

def get_existing_completed_tasks(output_path):
    completed_ids = set()
    # Check all json files in the directory
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
                            # Only retry tasks that crashed/errored — NOT reward=0.0 tasks.
                            # reward=0.0 means the LLM ran but gave wrong actions. That is a
                            # valid attempt and counts as a completed task (matching Phase 1 behavior).
                            info = item.get("info", {})
                            if info and "error" in info:
                                continue # Crashed/error → retry it
                            completed_ids.add(item["task_id"])
        except Exception:
            pass # Ignore corrupt files
    return completed_ids

def run_experiment(domain, model, strategy, user_model, user_strategy, trial, start_index=0, results_dir="results/phase3", args=None):
    print(f"Running Experiment: Domain={domain}, Model={model}, Strategy={strategy}, Trial={trial}, ResumeFrom={start_index}")
    
    # Construct output path
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)
    
    # Determine model provider (assuming openai for vLLM/GPT)
    model_provider = "openai" 
    
    # Set up Port Map for Local vLLM
    # Maps specific models to ports 8000 (Agent) and 8001 (User)
    port_map = {
        "Qwen/Qwen3-4B": 8000,
        "Qwen/Qwen3-8B": 8000,
        "Qwen/Qwen3-14B": 8000,
        "Qwen/Qwen3-32B": 8000,
        "User-Qwen3-32B": 8001,
        "gpt-4o": 8001 # Fallback if using gpt-4o as alias for user
    }
    
    # Export Port Map to environment so get_env can find local servers
    os.environ["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    # Give LiteLLM explicit base urls if available
    os.environ["OPENAI_API_BASE"] = f"http://localhost:{port_map.get(model, 8000)}/v1"
    os.environ["LITELLM_API_BASE"] = f"http://localhost:{port_map.get(model, 8000)}/v1"
    
    # SMART RESUME LOGIC
    if args.clean:
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            print(f"[--clean] Wiped result directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        completed_ids = set()
    else:
        completed_ids = get_existing_completed_tasks(output_path)
    
    # Get total tasks count (lightweight init)
    # Note: We assume 'test' split as per default
    try:
        temp_env = get_env(domain, user_strategy=user_strategy, user_model=user_model, user_provider="openai", task_split="test")
        total_tasks = len(temp_env.tasks)
        print(f"Total tasks in dataset: {total_tasks}. Completed so far: {len(completed_ids)}")
    except Exception as e:
        print(f"Warning: Could not determine total tasks ({e}). Falling back to simple start-index.")
        total_tasks = 116 # Default fallback for retail-test
    
    # Identify missing task IDs to build the queue
    needed_ids = []
    for i in range(total_tasks):
        if i >= start_index and str(i) not in completed_ids and i not in completed_ids: # Also check string versions
            needed_ids.append(str(i))
            
    if not needed_ids:
        print(f"All {total_tasks} tasks completed! Skipping.")
        return

    print(f"Resuming/Retrying {len(needed_ids)} tasks: {needed_ids[:5]}...")

    
    env = os.environ.copy()
    env["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    env["OPENAI_API_KEY"] = "sk-1234" # Ensure fake key is present to bypass LiteLLM validation
    
    cmd = [
        sys.executable, "run.py",
        "--agent-strategy", "tool-calling" if strategy == "fc" else strategy,
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", user_strategy,
        "--max-concurrency", "1", # Sequential for local
        "--seed", str(trial),
        "--log-dir", output_path,
        "--task-ids"
    ] + needed_ids
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"Experiment finished successfully. Results in {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Experiments")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", choices=["react", "act", "fc", "all"], default="all")
    parser.add_argument("--model", type=str, help="Specific model to run (e.g., Qwen/Qwen3-4B-Instruct)")
    parser.add_argument("--user-model", type=str, default="User-Qwen3-32B", help="Fixed user model")
    parser.add_argument("--start-index", type=int, default=0, help="Task index to start execution from")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--clean", action="store_true",
                        help="Wipe the result directory and start completely from scratch.")
    
    args = parser.parse_args()
    
    # GLOBAL FIX: Set API Key for get_env calls (User/Task counting)
    os.environ["OPENAI_API_KEY"] = "sk-1234"
    
    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    strategies = ["react", "act", "fc"] if args.strategy == "all" else [args.strategy]
    
    # If no specific model is provided, it defaults to None and fails, 
    # as the user should specify the model they are running servers for.
    models = [args.model] if args.model else []
    
    for domain in domains:
        for model in models:
            for strategy in strategies:
                for trial in range(args.trials):
                   run_experiment(domain, model, strategy, args.user_model, "llm", trial, start_index=args.start_index, results_dir="results/phase3", args=args)

if __name__ == "__main__":
    main()
