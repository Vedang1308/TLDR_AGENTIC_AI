import argparse
import subprocess
import os
import sys
import json
import glob
import concurrent.futures

# Import tasks directly to avoid API overhead
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from tau_bench.envs.airline.tasks_test import TASKS as airline_tasks
    from tau_bench.envs.retail.tasks_test import TASKS_TEST as retail_tasks
except ImportError:
    airline_tasks = []
    retail_tasks = []

def get_existing_completed_tasks(output_path):
    """
    Scans for completed task IDs in existing JSON log files.
    """
    completed_ids = set()
    search_pattern = os.path.join(output_path, "*.json")
    files = glob.glob(search_pattern)
    
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and "task_id" in item:
                        # Skip tasks that ended in an error/crash
                        info = item.get("info", {})
                        if info and info.get("error"):
                            continue
                        # Store as string to match CLI requirements
                        completed_ids.add(str(item["task_id"]))
        except Exception:
            pass 
    return completed_ids

def run_experiment(domain, model, strategy, user_model, user_strategy, trial, start_index, results_dir):
    """
    Worker function executed in parallel.
    """
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)

    # Local vLLM Port Map
    port_map = {
        "Qwen/Qwen3-4B": 8000,
        "Qwen/Qwen3-8B": 8000,
        "Qwen/Qwen3-14B": 8000,
        "Qwen/Qwen3-32B": 8000,
        "User-Qwen3-32B": 8001,
        "gpt-4o": 8001
    }

    # Task Counting & Resume Logic
    completed_ids = get_existing_completed_tasks(output_path)
    
    if domain == "airline":
        total_tasks = len(airline_tasks) if airline_tasks else 50
    else:
        total_tasks = len(retail_tasks) if retail_tasks else 115

    needed_ids = [
        str(i) for i in range(total_tasks) 
        if i >= start_index and str(i) not in completed_ids
    ]
            
    if not needed_ids:
        print(f"✅ Trial {trial}: {domain}/{strategy} already complete. Skipping.")
        return

    print(f"🚀 Launching Trial {trial}: {domain}/{strategy} ({len(needed_ids)} tasks left)")

    # Environment Setup
    env = os.environ.copy()
    env["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    env["OPENAI_API_KEY"] = "sk-1234"
    
    # Target the specific local server for the Agent
    local_base = f"http://localhost:{port_map.get(model, 8000)}/v1"
    env["OPENAI_API_BASE"] = local_base
    env["LITELLM_API_BASE"] = local_base
    
    cmd = [
        sys.executable, "run.py",
        "--agent-strategy", "tool-calling" if strategy == "fc" else strategy,
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", user_strategy,
        "--max-concurrency", "4", # Update this if you want to run multiple tasks concurrently within the same trial and have the required resources
        "--seed", str(trial),
        "--log-dir", output_path,
        "--task-ids"
    ] + needed_ids
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✨ Trial {trial} for {domain}/{strategy} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"❌ Trial {trial} for {domain}/{strategy} failed.")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Experiments (Parallel)")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", default="multi-agent")
    parser.add_argument("--model", type=str, help="Specific model to run")
    parser.add_argument("--user-model", type=str, default="User-Qwen3-32B")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5) 
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel trials")
    parser.add_argument("--results-dir", type=str, default="results/phase3")
    
    args = parser.parse_args()
    
    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    strategies = [args.strategy]
    models = [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B"
    ] if not args.model else [args.model]
    
    # Build Experiment List
    experiment_list = []
    for d in domains:
        for m in models:
            for s in strategies:
                for t in range(args.trials):
                    # Signature: (domain, model, strategy, user_model, user_strategy, trial, start_index, results_dir)
                    exp_tuple = (d, m, s, args.user_model, "llm", t, args.start_index, args.results_dir)
                    experiment_list.append(exp_tuple)

    print(f"🛠️  Prepared {len(experiment_list)} total trials. Running {args.max_workers} at a time.")

    # Execute Parallel Pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_experiment, *exp): exp for exp in experiment_list}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Critical error in execution: {e}")

if __name__ == "__main__":
    main()