import os
from typing import Dict, Any

class PEVConfig:
    """
    Central configuration for PEVAL Phase 4.
    Ensures 'One Model, All Nodes' logic and prevents port collisions.
    """
    # --- Port Management (To avoid 'Address already in use') ---
    AGENT_PORT = 8222
    USER_PORT = 8223
    
    # --- Model Selection ---
    # Supported: Qwen3-4B, 8B, 14B, 32B, Qwen2.5-72B-Instruct
    AGENT_MODEL = "qwen-32b-agent" 
    USER_MODEL = "qwen2.5-72b-simulator"
    
    # --- Tool Calling Strategy ---
    # Supported: fc, react, reflection, irma
    TOOL_STRATEGY = os.getenv("TOOL_STRATEGY", "fc")
    
    # --- Endpoints ---
    AGENT_ENDPOINT = f"http://localhost:{AGENT_PORT}/v1"
    USER_ENDPOINT = f"http://localhost:{USER_PORT}/v1"
    
    # --- External Summarizer (OpenAI API - No local VRAM cost) ---
    # Using GPT-4o-mini for fast, high-quality distillation
    SUMMARIZER_MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
    
    # --- Architecture Limits ---
    MAX_STEPS = 30
    RECURSION_LIMIT = 25
    CONTEXT_DISTILL_THRESHOLD = 8000 # Max tokens before triggering Distiller
    
    # --- Storage ---
    LOG_DIR = "results/harshith_new"
    WISDOM_FILENAME = "persistent_wisdom.json"
