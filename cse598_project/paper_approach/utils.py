import os
import json
from typing import Dict, Any

def get_paper_port_map() -> Dict[str, int]:
    """
    Returns the port mapping for the research paper models.
    Ensures that identical models used for both User and Agent roles are separated by port.
    """
    return {
        # Agent Models (Port 8000)
        "Qwen/Qwen3-4B-Instruct": 8000,
        "Qwen/Qwen3-14B": 8000,
        "Qwen/Qwen3-32B": 8000,
        "Qwen/Qwen2.5-72B-Instruct": 8000,
        
        # User Models (Port 8001)
        "User-72B": 8001,
        "Qwen/Qwen2.5-72B-Instruct-User": 8001, 
    }

def setup_paper_env(agent_model: str, user_model: str = "Qwen/Qwen2.5-72B-Instruct"):
    """
    Sets up environment variables for the research paper experiments.
    """
    port_map = get_paper_port_map()
    os.environ["TAUBENCH_PORT_MAP"] = json.dumps(port_map)
    
    # Map the requested model to the agent port (8000)
    agent_port = port_map.get(agent_model, 8000)
    os.environ["AGENT_API_BASE"] = f"http://localhost:{agent_port}/v1"
    os.environ["AGENT_MODEL_NAME"] = agent_model
    
    # Map the user model to the user port (8001)
    os.environ["USER_API_BASE"] = f"http://localhost:8001/v1"
    os.environ["USER_MODEL_NAME"] = user_model
    
    # Standard security bypass for local vLLM
    os.environ["OPENAI_API_KEY"] = "sk-1234"
    os.environ["LITELLM_API_BASE"] = os.environ["AGENT_API_BASE"]

def get_config_for_model(model_name: str) -> Dict[str, Any]:
    """
    Returns vLLM specific config strings for each model.
    """
    configs = {
        "Qwen/Qwen3-32B": {
            "tensor_parallel": 1,
            "max_model_len": 32768,
            "gpu_utilization": 0.90
        },
        "Qwen/Qwen2.5-72B-Instruct": {
            "tensor_parallel": 2, # Requires at least 2 GPUs for 72B
            "max_model_len": 32768,
            "gpu_utilization": 0.95
        }
    }
    return configs.get(model_name, {"tensor_parallel": 1, "max_model_len": 30000, "gpu_utilization": 0.50})
