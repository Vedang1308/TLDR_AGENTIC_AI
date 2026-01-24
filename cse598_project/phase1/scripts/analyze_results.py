import os
import json
import argparse
import collections
from math import comb

def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def calculate_metrics(results):
    num_trials = len(set([r['trial'] for r in results]))
    if num_trials < 1:
        return {}

    # Group by task_id
    c_per_task_id = collections.defaultdict(int)
    for result in results:
        task_id = result['task_id']
        if is_successful(result['reward']):
            c_per_task_id[task_id] += 1
            
    # Calculate Pass^k
    pass_hat_ks = {}
    total_tasks = len(c_per_task_id) if c_per_task_id else 1 # Avoid div by zero
    
    for k in range(1, 6): # Calculate for k=1 to 5 always
        if k > num_trials:
            pass_hat_ks[k] = 0.0
            continue
            
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            if c >= k:
                # Formula: comb(c, k) / comb(num_trials, k)
                # But typically Pass^k in this context is just success rate? 
                # Let's use the EXACT formula from run.py
                sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        
        pass_hat_ks[k] = sum_task_pass_hat_k / total_tasks

    return pass_hat_ks

def analyze_results(results_dir):
    # Structure: results/phase1/{domain}/{model_name}/{strategy}/trial_{i}/*.json
    # We want to aggregate ALL trials for a specific (domain, model, strategy)
    
    # Map: key=(domain, model, strategy) -> list of all result objects
    aggregated_data = collections.defaultdict(list)
    
    print(f"Scanning {results_dir}...")
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    # Extract metadata from path or file name?
                    # path is .../domain/model/strategy/trial_i/filename.json
                    parts = path.split(os.sep)
                    # Assuming standard structure created by runner
                    # parts[-5] = domain
                    # parts[-4] = model
                    # parts[-3] = strategy
                    
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

    # Generate Markdown Table
    print("\n# Phase 1 Results: Pass^k Metrics\n")
    print("| Domain | Model | Strategy | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Pass^5 |")
    print("|---|---|---|---|---|---|---|---|")
    
    sorted_keys = sorted(aggregated_data.keys())
    for key in sorted_keys:
        domain, model, strategy = key
        results = aggregated_data[key]
        metrics = calculate_metrics(results)
        
        row = f"| {domain} | {model} | {strategy} |"
        for k in range(1, 6):
            val = metrics.get(k, 0.0)
            row += f" {val:.3f} |"
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase1")
    args = parser.parse_args()
    analyze_results(args.results_dir)
