import sys
import os
import argparse
import json
import time
import requests
from datetime import datetime

# Ensure the local directory and peval_v4 are at the VERY FRONT of the path
# This prevents conflicts with older versions of 'tau_bench' in the conda environment
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "peval_v4"))

from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.envs.airline.env import MockAirlineDomainEnv
from peval_v4.src.graph.agent import PEVALAgent
from peval_v4.src.core.config import PEVConfig

def run_experiment(domain="retail", model_name="qwen-32b-agent", strategy="fc", num_tasks=-1, trials=5):
    # Determine actual task count if -1
    if num_tasks == -1:
        num_tasks = 115 if domain == "retail" else 50

    print(f"=== PEVAL Phase 4 Experiment: {domain} ===")
    print(f"Model: {model_name} | Strategy: {strategy.upper()} | Tasks: {num_tasks} | Trials: {trials}")
    
    # Update Configuration
    PEVConfig.TOOL_STRATEGY = strategy
    PEVConfig.AGENT_MODEL = model_name
    
    # 1. Setup Environment
    print(f"--- [INIT] Powering up Environment: {domain} ---")
    
    # We must tell LiteLLM (used by tau-bench) where our User Simulator is
    os.environ["OPENAI_API_BASE"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_BASE_URL"] = PEVConfig.USER_ENDPOINT
    os.environ["OPENAI_API_KEY"] = PEVConfig.OPENAI_API_KEY or "none"

    # Pre-flight Health Checks (Prevents hanging if vLLM is still warming up)
    def wait_for_server(url, name):
        print(f"--- [CHECK] Waiting for {name} ({url})... ---")
        while True:
            try:
                # vLLM is ready when /v1/models returns 200
                requests.get(f"{url}/models", timeout=2)
                print(f"--- [SUCCESS] {name} is LIVE! ---")
                return
            except:
                time.sleep(5)

    wait_for_server(PEVConfig.USER_ENDPOINT, "User Simulator (Port 8223)")
    wait_for_server(PEVConfig.AGENT_ENDPOINT, "Agent Server (Port 8222)")

    # Completion Heartbeat (Verifies that inference actually works and isn't deadlocked)
    def heartbeat_server(url, model_name, name):
        print(f"--- [HEARTBEAT] Testing inference for {name}... ---")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1
        }
        try:
            start_time = time.time()
            res = requests.post(f"{url}/chat/completions", json=payload, timeout=120)
            res.raise_for_status()
            print(f"--- [SUCCESS] {name} inference is LIVE ({time.time()-start_time:.1f}s) ---")
        except Exception as e:
            print(f"--- [FAIL] {name} inference UNREACHABLE: {e} ---")
            print(f"--- [TIP] Check Terminal logs for {name} for errors or OOM. ---")
            sys.exit(1)

    heartbeat_server(PEVConfig.USER_ENDPOINT, PEVConfig.USER_MODEL, "User Simulator")
    heartbeat_server(PEVConfig.AGENT_ENDPOINT, model_name, "Agent Server")

    print(f"--- [INFO] Environment connecting to User Simulator at {PEVConfig.USER_ENDPOINT} ---")
    print(f"--- [WAIT] Initializing Tau-Bench Environment (This calls the User Simulator)... ---")
    if domain == "retail":
        env = MockRetailDomainEnv(
            user_model=PEVConfig.USER_MODEL,
            user_provider="openai"
        )
    else:
        env = MockAirlineDomainEnv(
            user_model=PEVConfig.USER_MODEL,
            user_provider="openai"
        )
    print(f"--- [SUCCESS] Environment Loaded Successfully! ---")
        
    # 2. Initialize the Multi-Agent Architecture
    print(f"--- [INIT] Assembling PEVAL Architecture (13 Nodes) ---")
    # We pass the tools_info and wiki from the environment directly
    agent = PEVALAgent(
        tools_info=env.tools_info,
        wiki=env.wiki
    )
    print(f"--- [SUCCESS] Architecture Ready ---")
    
    # 3. Create Results Directory
    os.makedirs(PEVConfig.LOG_DIR, exist_ok=True)
    results_file = os.path.join(PEVConfig.LOG_DIR, f"{domain}_{strategy}_{model_name}_results.json")
    
    # 4. Storage for Pass^k calculation
    consistency_results = {}
    
    # 5. Run the Benchmark Loop
    for t_idx in range(num_tasks):
        print(f"\n--- Starting Evaluation for Task {t_idx} ---")
        consistency_results[f"task_{t_idx}"] = {"rewards": []}
        
        for trial in range(trials):
            print(f"  > Trial {trial + 1}/{trials}")
            result = agent.solve(env, task_index=t_idx)
            consistency_results[f"task_{t_idx}"]["rewards"].append(result.reward)
            
            # Explicit Terminal Output for Pass/Fail
            if result.reward == 1.0:
                print(f"  >>> RESULT: \033[92m[PASSED]\033[0m")
            else:
                print(f"  >>> RESULT: \033[91m[FAILED]\033[0m")
            
        # Save atomically after every task to prevent data loss on HPC timeout
        with open(results_file, 'w') as f:
            json.dump(consistency_results, f, indent=4)
    
    print(f"\n--- EXPERIMENT CONCLUDED ---")
    print(f"Results saved to: {results_file}")
    
    # Calculate simple total average
    all_rewards = [r for task in consistency_results.values() for r in task["rewards"]]
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    print(f"Overall Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEVAL Phase 4 Experiments for pass^k calculation")
    parser.add_argument("--domain", type=str, default="retail", choices=["retail", "airline"], help="Tau-Bench domain")
    parser.add_argument("--model", type=str, default="qwen-32b-agent", help="The vLLM served model name to use")
    parser.add_argument("--strategy", type=str, default="fc", choices=["fc", "react", "reflection", "irma"], help="Tool calling strategy to use")
    parser.add_argument("--num_tasks", type=int, default=-1, help="Number of tasks to evaluate (-1 for all: 115 retail / 50 airline)")
    parser.add_argument("--trials", type=int, default=5, help="Number of times to run each task (default: 5 for pass^5)")
    
    args = parser.parse_args()
    run_experiment(
        domain=args.domain, 
        model_name=args.model, 
        strategy=args.strategy, 
        num_tasks=args.num_tasks, 
        trials=args.trials
    )
