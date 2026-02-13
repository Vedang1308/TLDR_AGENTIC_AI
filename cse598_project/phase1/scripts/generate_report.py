import os
import glob
import json
import math
import sys

# Constants matching our data structure
DOMAINS = ["retail", "airline"]
# Normalized for display vs internal matching
MODELS_DISPLAY = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
MODELS_MAP = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B"
}
STRATEGIES = ["react", "act", "fc"]
TRIALS = range(5)
BASE_DIR = "results/phase1"

DOMAIN_TOTALS = {
    "retail": 115,
    "airline": 50
}

def sanitize_model_name(model):
    return model.replace("/", "_")

def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def get_trial_results(domain, model, strategy, trial):
    """
    Returns a dict {task_id: reward} for the given config.
    """
    safe_model = sanitize_model_name(model)
    trial_dir = os.path.join(BASE_DIR, domain, safe_model, strategy, f"trial_{trial}")
    results = {}
    
    if not os.path.exists(trial_dir):
        return results
        
    files = glob.glob(os.path.join(trial_dir, "*.json"))
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "task_id" in item and "reward" in item:
                            try:
                                tid = int(item["task_id"])
                                reward = float(item["reward"])
                                if tid not in results or reward > results[tid]:
                                    results[tid] = reward
                            except ValueError:
                                pass
        except Exception:
            pass
    return results

def calculate_metrics_row(domain, model, strategy):
    c_per_task_id = {tid: 0 for tid in range(DOMAIN_TOTALS[domain])}
    trial_counts = []

    for t in TRIALS:
        res = get_trial_results(domain, model, strategy, t)
        trial_counts.append(len(res))
        for tid, reward in res.items():
            if is_successful(reward):
                if tid in c_per_task_id:
                    c_per_task_id[tid] += 1

    num_trials = 5
    pass_hat_ks = {}
    total_tasks = DOMAIN_TOTALS[domain]
    
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0.0
        for c in c_per_task_id.values():
            numerator = math.comb(c, k)
            denominator = math.comb(num_trials, k)
            if denominator > 0:
                sum_task_pass_hat_k += numerator / denominator
        pass_hat_ks[k] = sum_task_pass_hat_k / total_tasks

    row_values = []
    # T0 | Pass^1 | T1 | Pass^2 ...
    for i in range(5):
        row_values.append(str(trial_counts[i]))
        k = i + 1
        val = pass_hat_ks[k]
        row_values.append(f"{val:.3f}")
        
    return row_values

def get_user_input():
    print("--- User Information ---")
    asurite = input("Enter ASUrite ID: ").strip()
    asu_id = input("Enter ASU ID: ").strip()
    full_name = input("Enter Full Name: ").strip()
    print("")
    
    print("--- Model Selection ---")
    for idx, m in enumerate(MODELS_DISPLAY):
        print(f"{idx + 1}. {m}")
    print(f"{len(MODELS_DISPLAY) + 1}. All")
    
    while True:
        try:
            choice = int(input(f"Select Model (1-{len(MODELS_DISPLAY) + 1}): ").strip())
            if 1 <= choice <= len(MODELS_DISPLAY):
                selected_model_display = MODELS_DISPLAY[choice - 1]
                selected_model_internal = MODELS_MAP[selected_model_display]
                break
            elif choice == len(MODELS_DISPLAY) + 1:
                selected_model_display = "All"
                selected_model_internal = "all"
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(MODELS_DISPLAY) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print("")
    print("--- Domain Selection ---")
    print("1. Airline")
    print("2. Retail")
    print("3. All")
    
    while True:
        try:
            choice = int(input("Select Domain (1-3): ").strip())
            if choice == 1:
                selected_domain = "airline"
                break
            elif choice == 2:
                selected_domain = "retail"
                break
            elif choice == 3:
                selected_domain = "all"
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return asurite, asu_id, full_name, selected_model_display, selected_model_internal, selected_domain

def main():
    asurite, asu_id, full_name, model_disp, model_internal, domain = get_user_input()
    
    # Logic to select loop targets
    if domain == "all":
        target_domains = DOMAINS
    else:
        target_domains = [domain]

    if model_internal == "all":
        # Sort or maintain order
        target_models_list = [(m, MODELS_MAP[m]) for m in MODELS_DISPLAY]
    else:
        target_models_list = [(model_disp, model_internal)]
    
    print("\ngenerating results...\n")
    
    print("="*60)
    print(f"Generated Report for: {full_name}")
    print(f"ASUrite: {asurite} | ASU ID: {asu_id}")
    selection_str = f"Model={model_disp}, Domain={domain.capitalize()}"
    print(f"Selected Configuration: {selection_str}")
    print("="*60)
    print("")

    # Generate Table Header
    header = f"| {'Domain':<8} | {'Model':<10} | {'Strategy':<8} |"
    for i in TRIALS:
        header += f" {'T'+str(i):<4} | {'pass^'+str(i+1):<6} |"
    
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    # Double loop for All/All
    for cur_domain in target_domains:
        for cur_model_disp, cur_model_internal in target_models_list:
            for strategy in STRATEGIES:
                row_vals = calculate_metrics_row(cur_domain, cur_model_internal, strategy)
                row_str = f"| {cur_domain:<8} | {cur_model_disp:<10} | {strategy:<8} |"
                for val in row_vals:
                    row_str += f" {val:<4} |"
                print(row_str)
        
    print("")
    print("Report Generation Complete.")

if __name__ == "__main__":
    main()
