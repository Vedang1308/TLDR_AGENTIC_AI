import os
import sys
import json
import argparse
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
phase3_path = os.path.join(project_root, "cse598_project", "phase3")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if phase3_path not in sys.path:
    sys.path.insert(0, phase3_path)

# Import from the original tau_bench
from cse598_project.phase3.tau_bench.run import run as original_run, RunConfig
from cse598_project.paper_approach.peval_strategy import PevalStrategy

def paper_agent_factory(tools_info, wiki, config):
    strategy = os.getenv("AGENT_STRATEGY", "ReAct")
    print(f"--- [GAUDI-FACTORY]: Creating PevalStrategy with strategy: {strategy} ---")
    return PevalStrategy(
        tools_info=tools_info,
        wiki=wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature,
        agent_strategy=strategy
    )

import cse598_project.phase3.tau_bench.run as original_run_module
original_run_module.agent_factory = paper_agent_factory

if __name__ == "__main__":
    from cse598_project.phase3.run import main
    main()
