import sys
import os
import argparse
import json
import time
import requests
from datetime import datetime

# Ensure the local directory is at the front of the path
sys.path.insert(0, os.getcwd())

from peval_v4_lite.src.core.config import PEVConfig
from peval_v4_lite.src.core.logger import PEVLogger
from peval_v3_gaudi.engine import PEVEngine
from peval_v3_gaudi.engine_native import PEVEngineNative

def run_experiment(domain="airline", model_name="qwen-72b-agent", num_tasks=-1, trials=5, use_native=False, strategy="fc"):
    if num_tasks == -1:
        num_tasks = 115 if domain == "retail" else 50

    print(f"=== PEVAL Gaudi Experiment: {domain} ===")
    print(f"Model: {model_name} | Strategy: {strategy} | Orchestration: {'NATIVE' if use_native else 'LITE'} | Tasks: {num_tasks} | Trials: {trials}")
    
    PEVConfig.AGENT_MODEL = model_name
    
    # 1. Setup Environment
    print(f"--- [INIT] Powering up Environment: {domain} ---")
    
    # PROACTIVE VALIDATION: Gaudi/Apptainer toolchains often break if the path has spaces.
    current_path = os.getcwd()
    if " " in current_path:
        print(f"\033[31m[CRITICAL ERROR]\033[0m: Your project path contains spaces: '{current_path}'")
        print("Intel Gaudi compiler toolchains (SynapseAI/vLLM) often fail to compile graphs when paths have spaces.")
        print("Please move your project to a path without spaces (e.g. ~/gaudi_setup/TLDR_AGENTIC_AI) and try again.")
        sys.exit(1)

    os.environ["OPENAI_API_BASE"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_BASE_URL"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_API_KEY"] = PEVConfig.OPENAI_API_KEY or "none"
    
    # Propagate Strategy and Domain to Native Nodes
    os.environ["AGENT_REASONING_MODE"] = strategy
    os.environ["AGENT_DOMAIN"] = domain
    os.environ["AGENT_MODEL_NAME"] = model_name
    os.environ["AGENT_API_BASE"] = PEVConfig.AGENT_ENDPOINT

    def wait_for_server(url, name, target_names=None):
        print(f"--- [CHECK] Waiting for {name} ({url})... ---")
        log_file = "agent_model_4b.log" if "Agent" in name else "user_model.log"
        
        while True:
            # Check for crashes in log file
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_tail = f.read()[-2000:] # Check last 2KB
                    if "RuntimeError" in log_tail or "Engine core initialization failed" in log_tail or "clang++: error" in log_tail:
                        print(f"\n\033[31m[CRASH DETECTED]\033[0m: {name} appears to have failed.")
                        print("Check the end of the log file for details:")
                        print("-" * 40)
                        print(log_tail.splitlines()[-10:])
                        print("-" * 40)
                        sys.exit(1)

            try:
                res = requests.get(f"{url}/models", timeout=30).json()
                if target_names:
                    # Convert single string to list for consistency
                    if isinstance(target_names, str):
                        target_names = [target_names]
                        
                    available_models = [m["id"] for m in res.get("data", [])]
                    # Check if ANY of our targets are in the list
                    found_any = any(t in available_models for t in target_names)
                    
                    if not found_any:
                        print(f"  ... {name} is up, but model {target_names} is still loading...")
                        time.sleep(10)
                        continue
                print(f"--- [SUCCESS] {name} is LIVE ---")
                return
            except Exception:
                print(f"  ... {name} not ready yet (retrying in 5s)...")
                time.sleep(5)

    wait_for_server(PEVConfig.USER_ENDPOINT, "User Simulator", [PEVConfig.USER_MODEL, "qwen-72b-simulator"])
    wait_for_server(PEVConfig.AGENT_ENDPOINT, "Agent Server", [model_name, "qwen-72b-agent"])

    # 2. Load Tau-Bench Environment
    from tau_bench.envs.retail.env import MockRetailDomainEnv
    from tau_bench.envs.airline.env import MockAirlineDomainEnv
    
    if domain == "retail":
        env = MockRetailDomainEnv(user_model=PEVConfig.USER_MODEL, user_provider="openai")
    else:
        env = MockAirlineDomainEnv(user_model=PEVConfig.USER_MODEL, user_provider="openai")
    
    # 3. Initialize Engine
    EngineClass = PEVEngineNative if use_native else PEVEngine
    engine = EngineClass(
        tools_info=env.tools_info,
        wiki=env.wiki,
        log_dir=f"results/gaudi_{domain}_{'native' if use_native else 'lite'}"
    )
    
    # 4. Results Storage
    os.makedirs(PEVConfig.LOG_DIR, exist_ok=True)
    sanitized_model = model_name.replace("/", "_")
    results_file = os.path.join(PEVConfig.LOG_DIR, f"v3_gaudi_{domain}_{sanitized_model}_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    consistency_results = {}
    
    # 5. Benchmark Loop
    for t_idx in range(num_tasks):
        print(f"\n--- [TASK {t_idx}] Starting Evaluation ---")
        consistency_results[f"task_{t_idx}"] = {"rewards": []}
        
        for trial in range(trials):
            print(f"  > Trial {trial + 1}/{trials}")
            result = engine.solve(env, task_index=t_idx)
            consistency_results[f"task_{t_idx}"]["rewards"].append(result.reward)
            
            status = "\033[92m[PASSED]\033[0m" if result.reward == 1.0 else "\033[91m[FAILED]\033[0m"
            print(f"  >>> RESULT: {status} (Reward: {result.reward})")
            
        # Ensure directory exists again just in case before final write
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(consistency_results, f, indent=4)
    
    print(f"\n--- EXPERIMENT CONCLUDED ---")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEVAL Phase 3 Gaudi experiments")
    parser.add_argument("--domain", type=str, default="airline", choices=["retail", "airline"])
    parser.add_argument("--model", type=str, default="qwen-72b-agent")
    parser.add_argument("--num_tasks", type=int, default=-1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--native", action="store_true", help="Use Native Python orchestration (Upgraded Phase 3)")
    parser.add_argument("--strategy", type=str, default="fc", choices=["fc", "react", "irma", "reflection"], help="Reasoning strategy")
    
    args = parser.parse_args()
    run_experiment(domain=args.domain, model_name=args.model, num_tasks=args.num_tasks, trials=args.trials, use_native=args.native, strategy=args.strategy)
