import os
import json
import collections
import shutil
import argparse

# Define expected structure
DOMAINS = ["airline", "retail"]
MODELS = ["Qwen_Qwen3-4B", "Qwen_Qwen3-8B", "Qwen_Qwen3-14B", "Qwen_Qwen3-32B"]
STRATEGIES = ["act", "fc", "react"]
TRIALS = [0, 1, 2, 3, 4]

def restructure_data(source_dir, output_dir):
    """
    Scans the source directory for result files, reorganizes them into a 
    structured format (Domain/Model/Strategy/Trial), and consolidates 
    duplicate task executions into a single 'results.json' per trial.
    """
    print(f"--- [Step 1] Restructuring Data ---")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} not found.")
        return False

    # Map: (domain, model, strategy, trial_idx) -> list of result files
    found_data = collections.defaultdict(list)
    
    print(f"Scanning {source_dir}...")
    
    # 1. Collect all valid result files
    count = 0
    for root, dirs, files in os.walk(source_dir):
        files.sort() # Sort by name (timestamp usually at end) to prioritize latest
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    parts = path.split(os.sep)
                    if "trial_" not in parts[-2]:
                        continue

                    domain = parts[-5]
                    model = parts[-4]
                    strategy = parts[-3]
                    
                    try:
                        trial_idx = int(parts[-2].replace("trial_", ""))
                        found_data[(domain, model, strategy, trial_idx)].append(path)
                        count += 1
                    except ValueError:
                        pass
                except Exception:
                    pass

    print(f"Found {count} raw result files mapping to {len(found_data)} unique trial slots.")
    
    # 2. Enforce 120-file structure
    os.makedirs(output_dir, exist_ok=True)
    
    total_files_created = 0
    missing_trials = 0
    populated_trials = 0
    
    for domain in DOMAINS:
        for model in MODELS:
            for strategy in STRATEGIES:
                for trial_idx in TRIALS:
                    # Target path
                    target_trial_dir = os.path.join(output_dir, domain, model, strategy, f"trial_{trial_idx}")
                    os.makedirs(target_trial_dir, exist_ok=True)
                    target_file = os.path.join(target_trial_dir, "results.json")
                    
                    # Gather data
                    files = found_data.get((domain, model, strategy, trial_idx), [])
                    
                    if not files:
                        # Create empty placeholder file if missing
                        with open(target_file, 'w') as f:
                            json.dump([], f)
                        missing_trials += 1
                    else:
                        # Consolidate data from all files for this trial
                        # Deduplicate by task_id, keeping latest
                        consolidated_tasks = {}
                        
                        for file_path in files: # Already sorted by timestamp
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        for item in data:
                                            # Ensure trial field is correct
                                            item['trial'] = trial_idx
                                            task_id = item['task_id']
                                            consolidated_tasks[task_id] = item
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                        
                        # Write consolidated result
                        final_results = list(consolidated_tasks.values())
                        final_results.sort(key=lambda x: x['task_id'])
                        
                        with open(target_file, 'w') as f:
                            json.dump(final_results, f, indent=2)
                        populated_trials += 1
                            
                    total_files_created += 1

    print(f"Restructuring complete.")
    print(f"Total categories enforced: {total_files_created}")
    print(f"Populated: {populated_trials}, Missing/Empty: {missing_trials}")
    return True

def validate_data(output_dir):
    """
    Validates that the output directory contains the expected number of 
    structured result categories (120).
    """
    print(f"\n--- [Step 2] Validating Structure ---")
    expected_categories = 120 # 2 domains * 4 models * 3 strategies * 5 trials
    count = 0
    for root, dirs, files in os.walk(output_dir):
        if "results.json" in files:
            count += 1
    
    if count == expected_categories:
        print(f"Validation Successful: Found exactly {count} structured trial directories.")
        return True
    else:
        print(f"Validation Warning: Found {count} trial directories (Expected {expected_categories}).")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and structure raw result files.")
    parser.add_argument("--source", default="results/phase1", help="Directory containing raw results")
    parser.add_argument("--output", default="results/phase1_structured", help="Directory to save structured outcome")
    args = parser.parse_args()
    
    if restructure_data(args.source, args.output):
        if validate_data(args.output):
            print("\nREADY FOR ANALYSIS.")
            print(f"Run: python3 scripts/analyze_results.py --results-dir {args.output}")
