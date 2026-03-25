import os
import json
from typing import Dict, Any

def get_paper_port_map() -> Dict[str, int]:
    """
    Returns the port mapping for the research paper models on Gaudi.
    """
    return {
        # Agent Models (Port 8000)
        "Qwen/Qwen3-4B-Instruct": 8000,
        "Qwen/Qwen3-14B": 8000,
        "Qwen/Qwen3-32B": 8000,
        "Qwen/Qwen2.5-72B-Instruct": 8000,
        
        # User Models (Port 8001)
        "Qwen/Qwen2.5-72B-Instruct-User": 8001, 
    }

def setup_paper_env(agent_model: str, user_model: str = "Qwen/Qwen2.5-72B-Instruct-User"):
    """
    Sets up environment variables for the research paper experiments on Gaudi.
    """
    port_map = get_paper_port_map()
    os.environ["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    
    agent_port = port_map.get(agent_model, 8000)
    os.environ["AGENT_API_BASE"] = f"http://localhost:{agent_port}/v1"
    os.environ["AGENT_MODEL_NAME"] = agent_model
    
    user_port = port_map.get(user_model, 8001)
    os.environ["USER_API_BASE"] = f"http://localhost:{user_port}/v1"
    os.environ["USER_MODEL_NAME"] = user_model
    
    # PEVAL Specific: Ensure all nodes use the target agent model
    os.environ["SUMMARIZER_MODEL"] = agent_model
    os.environ["REFLECTOR_MODEL"] = agent_model
    
    os.environ["OPENAI_API_KEY"] = "sk-1234"
    os.environ["LITELLM_API_BASE"] = os.environ["AGENT_API_BASE"]
    
    # Habana Specific
    os.environ["PT_HPU_LAZY_MODE"] = "1"
