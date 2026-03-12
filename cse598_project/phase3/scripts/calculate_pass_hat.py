import os
import glob
import json
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from math import comb

def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def calculate_pass_hat_metrics(results_dir="results/phase3"):
    """
    Calculates pass_hat(k): The probability that k random trials are ALL successful.
    This metric decreases as k increases, representing model reliability.
    """
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    # FIX: Use recursive glob and look for *.json instead of just .json
    search_pattern = os.path.join(results_dir, "**", "*.json")
    print(f"Searching for files using pattern: {search_pattern}")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No evaluation JSON files found in {results_dir}!")
        return
    
    print(f"Found {len(files)} JSON files. Processing...")
        
    records = []
    for fpath in files:
        # Standardize path for regex
        fpath = fpath.replace("\\", "/")
        
        # Metadata extraction logic
        domain = "airline" if "airline" in fpath else "retail" if "retail" in fpath else "unknown"
        
        # Robust trial extraction
        trial_match = re.search(r'trial_(\d+)', fpath)
        trial_idx = int(trial_match.group(1)) if trial_match else 0
        
        # Robust model extraction
        if "Qwen3-32B" in fpath or "Qwen_Qwen3-32B" in fpath:
            model = "Qwen-32B"
        elif "Qwen3-4B" in fpath or "Qwen_Qwen3-4B" in fpath:
            model = "Qwen-4B"
        else:
            model = "Unknown-Model"
            
        # Robust strategy extraction
        if "multi-agent" in fpath:
            strategy = "multi-agent"
        elif "react" in fpath:
            strategy = "react"
        elif "fc" in fpath:
            strategy = "fc"
        elif "act" in fpath:
            strategy = "act"
        else:
            strategy = "unknown-strategy"
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                # Handle both list and dict formats
                entries = data if isinstance(data, list) else [data]
                for task_data in entries:
                    reward = task_data.get("reward")
                    if reward is None:
                        reward_info = task_data.get("info", {}).get("reward_info")
                        if isinstance(reward_info, dict):
                            reward = reward_info.get("reward", 0.0)
                        else:
                            reward = 0.0
                    
                    # Ensure we have a task_id
                    task_id = task_data.get("task_id")
                    if task_id is None:
                        # Extract task_id from filename (e.g., 0.json -> 0)
                        basename = os.path.basename(fpath)
                        task_id = basename.split(".")[0]
                    
                    records.append({
                        "domain": domain, "model": model, "strategy": strategy,
                        "trial": trial_idx, "task_id": str(task_id),
                        "success": int(is_successful(float(reward)))
                    })
        except Exception as e:
            # Skip non-evaluation files or corrupted ones
            continue

    if not records:
        print("No valid success records found in the JSON files.")
        return

    df = pd.DataFrame(records)
    
    # 1. Group by Task to get c (successes) and n (total trials)
    task_stats = df.groupby(["domain", "model", "strategy", "task_id"]).agg(
        c=("success", "sum"),
        n=("success", "count")
    ).reset_index()

    # 2. Calculate pass_hat(k) for k=1 to 5
    max_k = 5
    results_list = []
    configs = task_stats[["domain", "model", "strategy"]].drop_duplicates()

    for _, config in configs.iterrows():
        subset = task_stats[
            (task_stats["domain"] == config["domain"]) & 
            (task_stats["model"] == config["model"]) & 
            (task_stats["strategy"] == config["strategy"])
        ]
        
        res = config.to_dict()
        for k in range(1, max_k + 1):
            # Pass^hat(k) = mean(comb(c, k) / comb(n, k))
            def calc_task_pass(row):
                if row['n'] < k: return 0
                return comb(row['c'], k) / comb(row['n'], k)
            
            task_probs = subset.apply(calc_task_pass, axis=1)
            res[f"pass@{k}"] = round(task_probs.mean() * 100, 2)
        results_list.append(res)

    summary_df = pd.DataFrame(results_list)
    print("\n=== pass_hat(k) Results (Success Consistency %) ===")
    print(summary_df.to_string(index=False))

    # Save Results
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/pass_hat_k_results.csv", index=False)

    # 3. Plotting the Decay
    plt.figure(figsize=(10, 6))
    for _, row in summary_df.iterrows():
        label = f"{row['model']} ({row['strategy']})"
        y_values = [row[f"pass@{k}"] for k in range(1, max_k + 1)]
        plt.plot(range(1, max_k + 1), y_values, marker='o', label=label)

    plt.title("Reliability Decay: Probability of Consistent Success across k Trials")
    plt.xlabel("k (Number of trials required to succeed)")
    plt.ylabel("Pass Rate (%)")
    plt.xticks(range(1, max_k + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("results/pass_hat_k_decay_plot.png")
    print("\nSaved table and plot to 'results/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/phase3")
    args = parser.parse_args()
    calculate_pass_hat_metrics(args.results_dir)
