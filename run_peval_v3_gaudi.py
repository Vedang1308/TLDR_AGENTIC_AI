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

def run_experiment(domain="airline", model_name="qwen-72b-agent", num_tasks=-1, trials=5):
    if num_tasks == -1:
        num_tasks = 115 if domain == "retail" else 50

    print(f"=== PEVAL Phase 3 Gaudi-Lite Experiment: {domain} ===")
    print(f"Model: {model_name} | Logic: PHASE 3 (No LangGraph) | Tasks: {num_tasks} | Trials: {trials}")
    
    PEVConfig.AGENT_MODEL = model_name
    
    # 1. Setup Environment
    print(f"--- [INIT] Powering up Environment: {domain} ---")
    os.environ["OPENAI_API_BASE"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_BASE_URL"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_API_KEY"] = PEVConfig.OPENAI_API_KEY or "none"

    def wait_for_server(url, name, model_name=None):
        print(f"--- [CHECK] Waiting for {name} ({url})... ---")
        while True:
            try:
                res = requests.get(f"{url}/models", timeout=30).json()
                if model_name:
                    available_models = [m["id"] for m in res.get("data", [])]
                    if model_name not in available_models:
                        print(f"  ... {name} is up, but model '{model_name}' is still loading...")
                        time.sleep(10)
                        continue
                print(f"--- [SUCCESS] {name} is LIVE ---")
                return
            except Exception:
                print(f"  ... {name} not ready yet (retrying in 5s)...")
                time.sleep(5)

    wait_for_server(PEVConfig.USER_ENDPOINT, "User Simulator", PEVConfig.USER_MODEL)
    wait_for_server(PEVConfig.AGENT_ENDPOINT, "Agent Server", model_name)

    # 2. Load Tau-Bench Environment
    from tau_bench.envs.retail.env import MockRetailDomainEnv
    from tau_bench.envs.airline.env import MockAirlineDomainEnv
    
    if domain == "retail":
        env = MockRetailDomainEnv(user_model=PEVConfig.USER_MODEL, user_provider="openai")
    else:
        env = MockAirlineDomainEnv(user_model=PEVConfig.USER_MODEL, user_provider="openai")
    
    # 3. Initialize the Phase 3 Gaudi Engine
    engine = PEVEngine(
        tools_info=env.tools_info,
        wiki=env.wiki,
        log_dir=f"results/phase3_gaudi_{domain}"
    )
    
    # 4. Results Storage
    os.makedirs(PEVConfig.LOG_DIR, exist_ok=True)
    results_file = os.path.join(PEVConfig.LOG_DIR, f"v3_gaudi_{domain}_{model_name}_results.json")
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
    
    args = parser.parse_args()
    run_experiment(domain=args.domain, model_name=args.model, num_tasks=args.num_tasks, trials=args.trials)
