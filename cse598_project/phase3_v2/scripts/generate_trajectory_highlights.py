import os
import json
import argparse
from pathlib import Path

def generate_highlights(results_dir: str, output_path: str):
    """
    Scans the completed Phase 3 results and extracts an interesting trajectory
    demonstrating the PEV (Plan-Execute-Validate) LangGraph logs compared to a standard trace.
    """
    
    # Locate a completed multi-agent json trace
    search_path = Path(results_dir)
    multi_agent_files = list(search_path.rglob("multi-agent/**/*.json"))
    fc_files = list(search_path.rglob("fc/**/*.json"))
    
    if not multi_agent_files:
        print(f"No multi-agent JSON files found in {results_dir}! Run the evaluation scripts first.")
        return
        
    print(f"Found {len(multi_agent_files)} multi-agent trajectories. Selecting the first valid one.")
    
    selected_ma_data = None
    target_task_id = None
    
    for f in multi_agent_files:
        with open(f, "r") as json_file:
            try:
                data = json.load(json_file)
                if isinstance(data, list) and len(data) > 0:
                    tr = data[0]
                    if "info" in tr and "pev_node_logs" in tr["info"]:
                        selected_ma_data = tr
                        target_task_id = tr.get("task_id")
                        break
            except Exception:
                pass
                
    if not selected_ma_data:
        print("Could not find a trajectory containing 'pev_node_logs'. Ensure logging is working.")
        return
        
    # Attempt to find the same task_id in the baseline 'fc' strategy
    selected_fc_data = None
    if fc_files and target_task_id is not None:
        for f in fc_files:
            with open(f, "r") as json_file:
                try:
                    data = json.load(json_file)
                    if isinstance(data, list) and len(data) > 0:
                        tr = data[0]
                        if tr.get("task_id") == target_task_id:
                            selected_fc_data = tr
                            break
                except Exception:
                    pass
                    
    # Generate Markdown Highlights
    md = [
        "# Phase 3 Trajectory Highlights (PEV Multi-Agent vs Baseline)",
        "",
        "This document highlights a side-by-side comparison of how a single monolithic completion model (Tool-Calling) "
        "compares against our Plan-Execute-Validate (PEV) LangGraph orchestration.",
        "",
        f"**Task ID Executed**: `{target_task_id}`",
        f"**Multi-Agent Reward**: {selected_ma_data.get('reward', 0.0)}",
        f"**FC Baseline Reward**: {selected_fc_data.get('reward', 0.0) if selected_fc_data else 'N/A'}",
        "",
        "## PEV Multi-Agent Internal Node Logs",
        "The following logs demonstrate the isolated internal thought process of the unique agents (Planner, Executor, Monitor, Validator) "
        "before a final action was dispatched to the Tau-Bench environment.",
        ""
    ]
    
    node_logs = selected_ma_data.get("info", {}).get("pev_node_logs", [])
    if not node_logs:
        md.append("> *No internal node logs found.*")
    else:
        for log in node_logs:
            md.append(f"### Agent: `{log.get('node', 'Unknown')}`")
            md.append(f"**State Segment**: {log.get('event', '')}")
            md.append("```json")
            # Pretty print the payload if it's a stringified JSON
            payload = log.get('data', '')
            try:
                if isinstance(payload, str):
                    parsed = json.loads(payload)
                    md.append(json.dumps(parsed, indent=2))
                else:
                    md.append(json.dumps(payload, indent=2))
            except:
                md.append(str(payload))
            md.append("```")
            md.append("")
            
    md.append("## Baseline Observation (Tool-Calling Mono-Agent)")
    if selected_fc_data:
        md.append("In contrast, the tool-calling baseline simply emits tool calls blindly, resulting in:")
        md.append("```json")
        for msg in selected_fc_data.get("messages", [])[-5:]: # Last 5 messages
            md.append(json.dumps(msg, indent=2))
        md.append("```")
    else:
        md.append("> *A matching baseline FC trace was not found for this exact task ID. Run Phase 1 FC experiments to populate this comparison.*")
        
    with open(output_path, "w") as out:
        out.write("\n".join(md))
        
    print(f"Successfully generated highlights at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/phase3")
    parser.add_argument("--output", type=str, default="trajectory_highlights.md")
    args = parser.parse_args()
    generate_highlights(args.results_dir, args.output)
