
import argparse
import subprocess
import os
import time

def run_experiment(domain, model, strategy, user_model, user_strategy, trial, results_dir="results/phase1"):
    print(f"Running Experiment: Domain={domain}, Model={model}, Strategy={strategy}, Trial={trial}")
    
    # Construct output path
    model_safe_name = model.replace("/", "_")
    output_path = os.path.join(results_dir, domain, model_safe_name, strategy, f"trial_{trial}")
    os.makedirs(output_path, exist_ok=True)
    
    # Determine model provider (assuming openai for vLLM/GPT)
    model_provider = "openai" 
    
    # Run command
    cmd = [
        "python3", "run.py",
        "--agent-strategy", "tool-calling" if strategy == "fc" else strategy,
        "--env", domain,
        "--model", model,
        "--model-provider", model_provider,
        "--user-model", user_model,
        "--user-model-provider", "openai", # Assuming fixed user model is also OpenAI compatible
        "--user-strategy", user_strategy,
        "--max-concurrency", "5", # Adjust based on capacity
        "--seed", str(trial),
        "--log-dir", output_path
    ]
    
    # Act and ReAct strategies might need specific flags or different handling in run.py
    # But based on README, they are just strategies.
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Experiment finished successfully. Results in {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Experiments")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", choices=["react", "act", "fc", "all"], default="all")
    parser.add_argument("--model", type=str, help="Specific model to run (e.g., Qwen/Qwen3-4B-Instruct)")
    parser.add_argument("--user-model", type=str, default="gpt-4o", help="Fixed user model")
    parser.add_argument("--trials", type=int, default=5)
    
    args = parser.parse_args()
    
    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    strategies = ["react", "act", "fc"] if args.strategy == "all" else [args.strategy]
    models = [
        "Qwen/Qwen3-4B-Instruct",
        "Qwen/Qwen3-8B-Instruct",
        # "Qwen/Qwen3-14B-Instruct", # Uncomment when ready
        # "Qwen/Qwen3-32B-Instruct"  # Uncomment when ready
    ] if not args.model else [args.model]
    
    for domain in domains:
        for model in models:
            for strategy in strategies:
                for trial in range(args.trials):
                   run_experiment(domain, model, strategy, args.user_model, "llm", trial)

if __name__ == "__main__":
    main()
