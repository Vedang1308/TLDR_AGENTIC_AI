import sys
import os
import litellm

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Save the original completion function
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    model_name = kwargs.get("model", "")
    
    # 1. Routing logic
    if "User-Qwen3-32B" in model_name:
        kwargs["api_base"] = "http://localhost:8001/v1"
        kwargs["custom_llm_provider"] = "openai"
        model_limit = 32768 # Match your vLLM --max-model-len
    else:
        kwargs["api_base"] = "http://localhost:8000/v1"
        kwargs["custom_llm_provider"] = "openai"
        model_limit = 32768 # Match your vLLM --max-model-len

    # 2. Dynamic max_tokens calculation
    try:
        # Estimate prompt length using LiteLLM's counter
        prompt_tokens = litellm.token_counter(model="gpt-4o", messages=kwargs.get("messages", []))
    except:
        prompt_tokens = 13000 # Fallback estimate

    # Leave a 512 token buffer for safety
    available_for_completion = model_limit - prompt_tokens - 512
    
    # Ensure max_tokens never pushes us over the limit
    requested_max = kwargs.get("max_tokens", 4096)
    kwargs["max_tokens"] = max(1, min(requested_max, available_for_completion))

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