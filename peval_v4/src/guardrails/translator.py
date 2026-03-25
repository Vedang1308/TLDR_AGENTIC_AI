import re
from ..core.state import PEVState
from ..core.logger import PEVLogger

class SemanticTranslator:
    """
    Component: Semantic Tool Translator
    Role: Normalizes parameter names (e.g., 'message' -> 'content') 
    to prevent technical schema mismatches.
    """
    def __init__(self, tools_info: list):
        self.tools_info = tools_info

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Translator", "Normalizing schema parameters...")
        
        action = state.current_action_draft
        tool_name = action.get("name")
        args = action.get("arguments", {})
        
        # 1. Find the target tool schema
        target_tool = next((t for t in self.tools_info if t.get('name') == tool_name), None)
        if not target_tool:
            return {"current_action_draft": action} # Pass through if unknown

        # 2. Map known intelligence-drifts (Fuzzy Parameter Normalization)
        # e.g. models often use 'message' instead of 'content' for 'respond'
        schema = target_tool.get('parameters', {}).get('properties', {})
        expected_params = list(schema.keys())
        
        new_args = {}
        for k, v in args.items():
            if k in expected_params:
                new_args[k] = v
            else:
                # Try to fuzzy-match common drifts
                if k == "message" and "content" in expected_params:
                    new_args["content"] = v
                elif k == "id" and any("id" in pk for pk in expected_params):
                    target_pk = next(pk for pk in expected_params if "id" in pk)
                    new_args[target_pk] = v
                else:
                    new_args[k] = v # Fallback

        action["arguments"] = new_args
        PEVLogger.success(f"Parameters mapped: {list(new_args.keys())}")
        return {
            "current_action_draft": action,
            "node_logs": [{"node": "Translator", "content": f"Normalized: {new_args}"}]
        }
