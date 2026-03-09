
import argparse
import subprocess
import os
import time
import sys
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
                            # Check if it was a crash/error
                            info = item.get("info", {})
                            if info and "error" in info:
                                continue # Treat as failed/missing
                            completed_ids.add(item["task_id"])
        except Exception:
            pass # Ignore corrupt files
    return completed_ids

def run_experiment(domain, model, strategy, user_model, user_strategy, trial, start_index=0, results_dir="results/phase3"):
    print(f"Running Experiment: Domain={domain}, Model={model}, Strategy={strategy}, Trial={trial}, ResumeFrom={start_index}")
    
    # Construct output path
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)
    
    # Determine model provider (assuming openai for vLLM/GPT)
    model_provider = "openai" 
    
    # No port map for OpenAI test
    os.environ["TAUBENCH_PORT_MAP"] = "{}"
    
    # SMART RESUME LOGIC
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
    
    # For testing, we only want to run a few tasks to verify
    # Let's override total_tasks if we just want a quick test run
    total_tasks = min(total_tasks, 5) # FORCE JUST 5 TASKS FOR TESTING 
    needed_ids = []
    for i in range(total_tasks):
        if i >= start_index and str(i) not in completed_ids and i not in completed_ids: # Also check string versions
            needed_ids.append(str(i))
            
    if not needed_ids:
        print(f"All {total_tasks} tasks completed! Skipping.")
        return

    print(f"Resuming/Retrying {len(needed_ids)} tasks: {needed_ids[:5]}...")

    
    env = os.environ.copy()
    env["TAUBENCH_PORT_MAP"] = "{}"
    if "OPENAI_API_BASE" in env:
        del env["OPENAI_API_BASE"]
    if "LITELLM_API_BASE" in env:
        del env["LITELLM_API_BASE"]
    
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
    parser = argparse.ArgumentParser(description="Run Phase 1 Experiments")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", choices=["react", "act", "fc", "multi-agent", "all"], default="multi-agent")
    parser.add_argument("--model", type=str, help="Specific model to run (e.g., Qwen/Qwen3-4B-Instruct)")
    parser.add_argument("--user-model", type=str, default="User-Qwen3-32B", help="Fixed user model")
    parser.add_argument("--start-index", type=int, default=0, help="Task index to start execution from")
    parser.add_argument("--trials", type=int, default=5)
    
    args = parser.parse_args()
    
    # GLOBAL FIX: Do not overwrite API key for GPT tests unless empty
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "sk-1234":
        print("WARNING: You must set OPENAI_API_KEY to run this test script!")
    
    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    strategies = ["multi-agent"] if args.strategy == "all" else [args.strategy]
    models = ["gpt-4o-mini"] if not args.model else [args.model]
    
    for domain in domains:
        for model in models:
            for strategy in strategies:
                for trial in range(args.trials):
                   run_experiment(domain, model, strategy, args.user_model, "llm", trial, start_index=args.start_index)

if __name__ == "__main__":
    main()
