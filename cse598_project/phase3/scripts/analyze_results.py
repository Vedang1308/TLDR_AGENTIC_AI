import os
import json
import argparse
import collections
from math import comb

def is_successful(reward: float) -> bool:
    """Check if the reward indicates a successful task completion."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def calculate_metrics(results):
    """
    Calculates Pass^k metrics across all trials for a given configuration.
    Pass^k = (1 / |Tasks|) * sum_{task} [comb(c_task, k) / comb(n_trials, k)]
    where c_task is the number of successful trials for that task.
    """
    # Identify unique trials present in the dataset
    all_trials = set([r.get('trial') for r in results if 'trial' in r])
    num_trials = len(all_trials)
    
    if num_trials < 1:
        return {}

    # Map: task_id -> count of successful trials
    success_counts = collections.defaultdict(int)
    # Map: task_id -> set of all trials that attempted this task
    task_trials = collections.defaultdict(set)
    
    for result in results:
        task_id = result['task_id']
        trial_id = result.get('trial')
        task_trials[task_id].add(trial_id)
        if is_successful(result.get('reward', 0)):
            success_counts[task_id] += 1
            
    # Calculate Pass^k
    pass_hat_ks = {}
    total_tasks = len(task_trials)
    if total_tasks == 0:
        return {}
    
    # We calculate for k up to the number of trials available (max 5)
    for k in range(1, 6):
        if k > num_trials:
            pass_hat_ks[k] = 0.0
            continue
            
        sum_task_pass_hat_k = 0
        for task_id in task_trials:
            c = success_counts[task_id]
            if c >= k:
                sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        
        pass_hat_ks[k] = sum_task_pass_hat_k / total_tasks

    return pass_hat_ks

def analyze_results(results_dir):
    """
    Aggregates results from the structured directory and prints a Pass^k report.
    Expected structure: {results_dir}/{domain}/{model}/{strategy}/trial_{i}/results.json
    """
    # Map: (domain, model, strategy) -> list of all result objects
    aggregated_data = collections.defaultdict(list)
    
    print(f"Scanning structured results in: {results_dir}...")
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "results.json":
                path = os.path.join(root, file)
                try:
                    parts = path.split(os.sep)
                    # Backwards index to handle different mount points
                    # parts[-5]: domain, parts[-4]: model, parts[-3]: strategy, parts[-2]: trial_i
                    if "trial_" not in parts[-2]:
                        continue

                    domain = parts[-5]
                    model = parts[-4]
                    strategy = parts[-3]
                    
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            aggregated_data[(domain, model, strategy)].extend(data)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    if not aggregated_data:
        print("No valid data found to analyze.")
        return

    # Generate Markdown Table
    print("\n# Phase 3 Results: Pass^k Metrics (Cumulative Learning)")
    print("\n| Domain | Model | Strategy | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Pass^5 |")
    print("|---|---|---|---|---|---|---|---|")
    
    sorted_keys = sorted(aggregated_data.keys())
    for key in sorted_keys:
        domain, model, strategy = key
        results = aggregated_data[key]
        metrics = calculate_metrics(results)
        
        if not metrics:
            continue
            
        row = f"| {domain} | {model} | {strategy} |"
        for k in range(1, 6):
            val = metrics.get(k, 0.0)
            row += f" {val:.3f} |"
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze structured result files.")
    parser.add_argument("--results-dir", default="results/phase3_structured", help="Directory containing structured results")
    args = parser.parse_args()
    
    # Resolve absolute path
    results_abs = os.path.abspath(args.results_dir)
    analyze_results(results_abs)
