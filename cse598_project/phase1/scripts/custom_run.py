import sys
import os
import litellm

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Save the original completion function
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    # 1. Routing logic
    original_model = kwargs.get("model", "")
    if "User-Qwen3-32B" in original_model:
        kwargs["api_base"] = "http://localhost:8001/v1"
        # We use 'openai/' + the name you set in your --served-model-name
        kwargs["model"] = "openai/User-Qwen3-32B"
    else:
        kwargs["api_base"] = "http://localhost:8000/v1"
        # Match the name you set in your vLLM bash script
        kwargs["model"] = "openai/gpt-4-32k"

    # 2. THE KEY: Tell LiteLLM this is a custom model with NO limits
    # This prevents LiteLLM from checking context windows locally.
    kwargs["custom_llm_provider"] = "openai"
    kwargs["mock_response"] = None # Ensure it's not mocking
    
    # 3. STRIP MAX TOKENS
    # This removes the context check entirely.
    kwargs.pop("max_tokens", None)
    kwargs.pop("max_completion_tokens", None)
    
    # 4. Silence local validation
    kwargs["drop_params"] = True
    litellm.suppress_debug_info = True

    return original_completion(*args, **kwargs)
    
# Apply the patch
litellm.completion = patched_completion
print(f"✅ LiteLLM Router active: User (8001) | Agent (8000)")

# Import and execute the main experiment logic
try:
    import run
    if hasattr(run, 'main'):
        run.main()
    else:
        print("Error: run.py does not have a main() function.")
except Exception as e:
    print(f"Error executing run.main(): {e}")