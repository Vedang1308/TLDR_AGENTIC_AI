# Phase 1: Benchmark Setup + Baseline Results

## Setup Description
*   **Agent Models**: Qwen3 Base models (4B, 8B, 14B, 32B).
*   **User Model**: Qwen3-32B-Instruct (simulating user via `tau-bench` prompts).
*   **Environment**: `tau-bench` (Retail & Airline domains).
*   **Infrastructure**: Local vLLM servers using `start_vllm_*.sh` scripts with port mapping.

## Individual Contribution
(Describe what you did here: e.g., set up local vLLM, patched tau-bench for base_url support, implemented base model drivers, etc.)

## Results Table (Pass^k)
| Domain | Model | Strategy | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Pass^5 |
|---|---|---|---|---|---|---|---|
| (Paste results from analyze_results.py here) | ... | ... | ... | ... | ... | ... | ... |

*(Run `python3 analyze_results.py` after experiments to generate this table)*

## Terminal Screenshot
(Paste screenshot of your terminal showing the script completing successfully)
