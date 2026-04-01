import os
from typing import Dict, Any

class PEVConfig:
    """
    Central configuration for PEVAL Phase 4.
    Ensures 'One Model, All Nodes' logic and prevents port collisions.
    """
    # --- Port Management (To avoid 'Address already in use') ---
    AGENT_PORT = 8224
    USER_PORT = 8225
    
    # --- Model Selection ---
    # Supported: Qwen3-4B, 8B, 14B, 32B, Qwen2.5-72B-Instruct
    AGENT_MODEL = "qwen-agent" 
    USER_MODEL = "qwen-72b-simulator"
    
    # --- Tool Calling Strategy ---
    # Supported: fc, react, reflection, irma
    TOOL_STRATEGY = os.getenv("TOOL_STRATEGY", "fc")
    
    # --- Endpoints ---
    AGENT_ENDPOINT = f"http://127.0.0.1:{AGENT_PORT}/v1"
    USER_ENDPOINT = f"http://127.0.0.1:{USER_PORT}/v1"
    
    # --- External Summarizer (No local VRAM cost) ---
    # Default to OpenRouter Google/Gemini or OpenAI 4o-mini
    SUMMARIZER_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
    
    # OpenRouter (Low cost / Free tier support)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"
    
    # --- Architecture Limits ---
    MAX_STEPS = 50
    RECURSION_LIMIT = 50
    CONTEXT_DISTILL_THRESHOLD = 5000 # Max tokens before triggering Distiller
    
    # --- Storage ---
    LOG_DIR = "results/harshith_new"
    WISDOM_FILENAME = os.path.join(LOG_DIR, "persistent_wisdom.json")
