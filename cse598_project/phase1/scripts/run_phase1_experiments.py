import argparse
import subprocess
import os
import sys
import json
import glob
import concurrent.futures

# Import tasks for resume logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from tau_bench.envs.airline.tasks_test import TASKS as airline_tasks
    from tau_bench.envs.retail.tasks_test import TASKS_TEST as retail_tasks
except ImportError:
    airline_tasks = []
    retail_tasks = []

def get_existing_completed_tasks(output_path):
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
                        if not item.get("info", {}).get("error"):
                            completed_ids.add(int(item["task_id"]))
        except Exception:
            pass 
    return completed_ids

def run_experiment(domain, model, strategy, user_model, user_strategy, trial, start_index, results_dir):
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)
    
    completed_ids = get_existing_completed_tasks(output_path)
    total_tasks = len(airline_tasks) if domain == "airline" else len(retail_tasks)
    if total_tasks == 0: total_tasks = 100 # Fallback

    needed_ids = [str(i) for i in range(total_tasks) if i >= start_index and i not in completed_ids]
            
    if not needed_ids:
        print(f"✅ Trial {trial} for {model} in {domain} ({strategy}) already complete. Skipping.")
        return

    print(f"🚀 Launching Trial {trial}: {domain}/{strategy} ({len(needed_ids)} tasks left)")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    
    cmd = [
        sys.executable, "scripts/custom_run.py",
        "--agent-strategy", "tool-calling" if strategy == "fc" else strategy,
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", user_strategy,
        "--max-concurrency", "20", 
        "--seed", str(trial),
        "--log-dir", output_path,
        "--task-ids"
    ] + needed_ids
    
    try:
        # Use capture_output=False so you can see the progress in the terminal
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial} failed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="airline")
    parser.add_argument("--strategy", choices=["react", "act", "fc", "all"], default="all")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--user-model", type=str, default="User-Qwen3-32B")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5) # Defaulting to 5 as you mentioned
    parser.add_argument("--max-workers", type=int, default=3)
    
    args = parser.parse_args()
    
    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    strategies = ["react", "act", "fc"] if args.strategy == "all" else [args.strategy]
    
    # --- FIX: BUILDING THE LIST CORRECTLY ---
    experiment_list = []
    for d in domains:
        for s in strategies:
            for t in range(args.trials): # This ensures trials 0, 1, 2, 3, 4 are added
                # Create a tuple of 8 arguments matching run_experiment's signature
                exp_tuple = (d, args.model, s, args.user_model, "llm", t, args.start_index, "results/phase1")
                experiment_list.append(exp_tuple)

    print(f"🛠️  Prepared {len(experiment_list)} total trials. Running {args.max_workers} at a time.")

    # --- FIX: UNPACKING THE TUPLE ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Pass the function and then the unpacked tuple
        futures = {executor.submit(run_experiment, *exp): exp for exp in experiment_list}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Critical error: {e}")

if __name__ == "__main__":
    main()