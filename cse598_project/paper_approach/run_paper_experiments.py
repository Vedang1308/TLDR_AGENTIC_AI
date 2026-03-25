import argparse
import subprocess
import os
import time
import sys
import concurrent.futures
import json
import glob
import urllib.request

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cse598_project.paper_approach.utils import setup_paper_env, get_paper_port_map

# ── HPU AUTO-DETECTION (GAUDI) ─────────────────────────────────────────────────
def detect_hpu_count() -> int:
    try:
        result = subprocess.run(
            ["hl-smi", "-L"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            count = len([line for line in result.stdout.splitlines() if "AIP" in line or "Gaudi" in line])
            return count
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0

def get_hpu_config(hpu_count: int) -> dict:
    if hpu_count >= 2:
        return {
            "mode": "DUAL-HPU (parallel Gaudi benchmarks)",
            "max_workers": 4, # Slightly more conservative for Gaudi memory
            "max_concurrency": 16,
        }
    else:
        return {
            "mode": "SINGLE-HPU (sequential Gaudi benchmarks)",
            "max_workers": 1,
            "max_concurrency": 1,
        }
# ── vLLM AUTO-DETECTION ────────────────────────────────────────────────────────
def detect_model(port: int) -> str:
    try:
        url = f"http://localhost:{port}/v1/models"
        print(f"--- [DEBUG]: Checking for served model on port {port}... ---")
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            if "data" in data and len(data["data"]) > 0:
                model_id = data["data"][0]["id"]
                print(f"--- [DEBUG]: Found active model on {port}: {model_id} ---")
                return model_id
    except Exception as e:
        print(f"--- [DEBUG]: Port {port} did not respond or failed: {str(e)} ---")
    return None
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(domain, model, strategy, user_model, trial, max_concurrency=1):
    print(f"\n--- [GAUDI PAPER EXP]: {domain} | {model} | {strategy} | Trial {trial} ---")
    setup_paper_env(model, user_model)
    
    model_safe = model.replace("/", "_")
    # Localized structure: results inside paper_approach
    paper_approach_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(paper_approach_dir, "results", domain, model_safe, strategy, f"trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    # Construct Command
    run_script = os.path.join(os.path.dirname(__file__), "run.py")
    
    # Inject PYTHONPATH to ensure tau_bench is discoverable
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    phase3_path = os.path.join(project_root, "cse598_project", "phase3")
    env["PYTHONPATH"] = f"{project_root}:{phase3_path}" + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")
    env["AGENT_STRATEGY"] = strategy # Moved from below
    
    cmd = [
        sys.executable, run_script,
        "--env", domain,
        "--model", model,
        "--model-provider", "openai",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", "llm",
        "--max-concurrency", str(max_concurrency),
        "--seed", str(trial),
        "--log-dir", output_dir,
        "--agent-strategy", "multi-agent"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"\n--- [GAUDI SUCCESS]: {output_dir} ---")
    except subprocess.CalledProcessError as e:
        print(f"\n--- [GAUDI FAILED]: {output_dir} | {e} ---")

def main():
    parser = argparse.ArgumentParser(description="Run Research Paper (PEVAL) Experiments on Gaudi")
    parser.add_argument("--domain", choices=["retail", "airline", "all"], default="all")
    parser.add_argument("--strategy", nargs="+", default=["ReAct", "FC", "Self-Reflection"])
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--user-model", type=str, default=None)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    
    # Auto-detect models if not specified
    if args.model is None:
        active_agent = detect_model(8000)
        if active_agent:
            print(f"--- [AUTO-DETECT]: Found active agent model: {active_agent} ---")
            args.model = [active_agent]
        else:
            args.model = [
                "Qwen/Qwen3-4B-Instruct",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-32B",
                "Qwen/Qwen2.5-72B-Instruct"
            ]

    if args.user_model is None:
        active_user = detect_model(8001)
        if active_user:
            print(f"--- [AUTO-DETECT]: Found active user model: {active_user} ---")
            args.user_model = active_user
        else:
            args.user_model = "Qwen/Qwen2.5-72B-Instruct-User"
    
    hpu_count = detect_hpu_count()
    hpu_cfg = get_hpu_config(hpu_count)
    
    max_workers = args.max_workers if args.max_workers is not None else hpu_cfg["max_workers"]
    concurrency = args.concurrency if args.concurrency is not None else hpu_cfg["max_concurrency"]
    
    print(f"\n{'='*60}")
    print(f"  HPU AUTO-DETECTION (GAUDI RESEARCH)")
    print(f"  Detected HPUs : {hpu_count}")
    print(f"  Execution Mode: {hpu_cfg['mode']}")
    print(f"  Max Workers   : {max_workers}")
    print(f"  Max Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    domains = ["retail", "airline"] if args.domain == "all" else [args.domain]
    experiments = [
        (domain, model, strategy, args.user_model, trial, concurrency)
        for domain in domains
        for model in args.model
        for strategy in args.strategy
        for trial in range(args.trials)
    ]
    
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_experiment, *exp) for exp in experiments]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    else:
        for exp in experiments:
            run_experiment(*exp)

if __name__ == "__main__":
    main()
