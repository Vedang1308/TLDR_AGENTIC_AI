import sys
import os

# Add the project root to sys.path so we can import tau_bench and run.py
# scripts/ is one level deep, so we need to go up one level.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Monkey-patch litellm.completion BEFORE importing anything else that might use it
try:
    import litellm
except ImportError:
    print("Error: litellm not found. Please ensure it is installed.")
    sys.exit(1)

original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    # Ensure max_tokens is set to prevent infinite generation loops
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 4096
        # print("DEBUG: Enforced max_tokens=4096 in monkey-patched completion")
    return original_completion(*args, **kwargs)

litellm.completion = patched_completion
print(f"Monkey-patched litellm.completion to enforce max_tokens=4096")

# Now import the original run module and execute main
try:
    import run
    if hasattr(run, 'main'):
        # Pass control to the original script
        run.main()
    else:
        print("Error: run.py does not have a main() function.")
        sys.exit(1)
except ImportError as e:
    print(f"Error importing run.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error executing run.main(): {e}")
    sys.exit(1)
