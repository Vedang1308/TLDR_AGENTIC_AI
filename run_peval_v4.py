import sys
import os
import argparse

# Ensure the new peval_v4 src is in the path
sys.path.append(os.path.join(os.getcwd(), "peval_v4"))

from tau_bench.envs.retail import RetailEnv
from tau_bench.envs.airline import AirlineEnv
from peval_v4.src.graph.agent import PEVALAgent
from peval_v4.src.core.config import PEVConfig

def run_experiment(domain="retail", model_name="qwen-32b-agent", strategy="fc"):
    print(f"=== PEVAL Phase 4 Experiment: {domain} ===")
    print(f"Model: {model_name} | Strategy: {strategy.upper()}")
    
    # Update Configuration
    PEVConfig.TOOL_STRATEGY = strategy
    PEVConfig.AGENT_MODEL = model_name
    
    # 1. Setup Environment
    if domain == "retail":
        env = RetailEnv()
    else:
        env = AirlineEnv()
        
    # 2. Initialize the Multi-Agent Architecture
    # We pass the tools_info and wiki from the environment directly
    agent = PEVALAgent(
        tools_info=env.tools_info,
        wiki=env.wiki
    )
    
    # 3. Solve a task
    result = agent.solve(env, task_index=0)
    
    print(f"\n--- EXPERIMENT CONCLUDED ---")
    print(f"Reward: {result.reward}")
    print(f"Final Path Length: {len(result.messages)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEVAL Phase 4 Experiments")
    parser.add_argument("--domain", type=str, default="retail", choices=["retail", "airline"], help="Tau-Bench domain")
    parser.add_argument("--model", type=str, default="qwen-32b-agent", help="The vLLM served model name to use")
    parser.add_argument("--strategy", type=str, default="fc", choices=["fc", "react", "reflection", "irma"], 
                        help="Tool calling strategy to use")
    
    args = parser.parse_args()
    run_experiment(domain=args.domain, model_name=args.model, strategy=args.strategy)
