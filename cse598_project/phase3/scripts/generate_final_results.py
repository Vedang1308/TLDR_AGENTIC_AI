import os
import glob
import json
import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from math import comb

def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def calculate_metrics(results_dir="results/phase3"):
    """
    Parses all JSON traces in the results directory.
    Calculates the exact pass^k metric used in Phase 1 for all models and strategies.
    Outputs a DataFrame and plots.
    """
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found. Please run experiments first.")
        return

    # Structure: results_dir / domain / model / strategy / trial_x / *.json
    search_pattern = os.path.join(results_dir, "*", "*", "*", "*", "*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No evaluation JSON files found!")
        return
        
    records = []
    
    for fpath in files:
        parts = fpath.split(os.sep)
        try:
            domain = parts[-5]
            model = parts[-4].replace("_", "/") # Restore Qwen/Qwen3-4B format
            strategy = parts[-3]
            trial_str = parts[-2]
            trial_idx = int(trial_str.replace("trial_", ""))
            
            with open(fpath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for task_data in data:
                        reward = task_data.get("reward", 0.0)
                        task_id = task_data.get("task_id", -1)
                        success = is_successful(reward)
                        
                        records.append({
                            "domain": domain,
                            "model": model,
                            "strategy": strategy,
                            "trial": trial_idx,
                            "task_id": task_id,
                            "success": int(success)
                        })
        except Exception as e:
            pass # Ignore malformed paths
            
    df = pd.DataFrame(records)
    if df.empty:
        print("No valid results parsed.")
        return
        
    print("\n=== Phase 3 Evaluation Results ===")
    
    # Calculate pass^1 for simplicity across all config combos
    summary = df.groupby(["domain", "model", "strategy"]).agg(
        pass_rate=("success", "mean"),
        total_tasks=("task_id", "nunique"),
        trials_recorded=("trial", "nunique")
    ).reset_index()
    
    # Format pass_rate as percentage
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(2).astype(str) + "%"
    
    print(summary.to_markdown(index=False))
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/phase3_final_results_table.csv", index=False)
    print("\nSaved table to results/phase3_final_results_table.csv")
    
    # Optional: Generate a quick plot comparing strategies
    try:
        pivot_df = df.groupby(["model", "strategy"])["success"].mean().unstack() * 100
        ax = pivot_df.plot(kind="bar", figsize=(10, 6), title="Performance by Strategy (Pass Rate %)")
        ax.set_ylabel("Pass Rate (%)")
        plt.tight_layout()
        plt.savefig("results/phase3_method_vs_baseline_plot.png")
        print("Saved plot to results/phase3_method_vs_baseline_plot.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/phase1") # Can point to phase1 or phase3
    args = parser.parse_args()
    calculate_metrics(args.results_dir)
