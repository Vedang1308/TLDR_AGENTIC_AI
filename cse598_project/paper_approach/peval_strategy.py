import os
import json
from typing import List, Dict, Any, Optional
from tau_bench.agents.base import Agent
from tau_bench.types import AgentRunResult
from .peval_graph import create_peval_graph

class PevalStrategy(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki,
        model: str,
        provider: str,
        temperature: float = 0.0,
        agent_strategy: str = "ReAct",
        **kwargs
    ):
        super().__init__(tools_info, wiki)
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.agent_strategy = agent_strategy
        
        # Initialize the PEVAL Graph
        self.graph = create_peval_graph()
        
        # Load existing wisdom if any
        self.wisdom_file = "cse598_project/paper_approach/results/persistent_wisdom.json"
        self.global_wisdom = []
        if os.path.exists(self.wisdom_file):
            try:
                with open(self.wisdom_file, "r") as f:
                    self.global_wisdom = json.load(f)
            except:
                pass

    def solve(self, env: Any, task_index: int) -> AgentRunResult:
        """
        Solves a single task using the PEVAL LangGraph.
        """
        # 1. Initialize State
        state = {
            "messages": [],
            "user_utterance": env.reset(),
            "env_observation": "",
            "strategic_kernel": "Initial turn.",
            "strategic_plan": "",
            "action_draft": {},
            "normalized_action": {},
            "global_wisdom": self.global_wisdom,
            "metadata": {"task_index": task_index, "strategy": self.agent_strategy},
            "retry_count": 0,
            "current_node": "summarizer",
            "is_finished": False,
            "reward": 0.0
        }
        
        # 2. Loop until the graph finishes
        # For a research paper, we enforce a session limit to prevent infinite loops
        max_turns = 15
        turn_count = 0
        
        while not state["is_finished"] and turn_count < max_turns:
            print(f"\n--- [PEVAL TURN {turn_count}] ---")
            # The graph handles one high-level plan execution or a conversation turn
            # In our case, the graph is the loop
            output = self.graph.invoke(state)
            state.update(output)
            
            # Since the graph ends at END, we check if we need to interact with the environment again
            # In a real LangGraph setup, the environment interaction (env.step) is part of a node.
            # I will ensure the validator_node in Peval is actually calling the env.
            
            # For simplicity in this implementation, if the validator returns a tool result,
            # we check if it was a final response.
            action = state.get("normalized_action", {})
            if action.get("name") == "respond":
                obs = env.step(action["arguments"].get("content", ""))
                state["messages"].append({"role": "assistant", "content": action["arguments"].get("content", "")})
                state["messages"].append({"role": "user", "content": obs})
                state["user_utterance"] = obs
                if "###STOP###" in obs or turn_count >= max_turns - 1:
                    state["is_finished"] = True
            else:
                # Real tool call
                obs = env.step(json.dumps(action))
                state["messages"].append({"role": "assistant", "tool_calls": [{"function": action}]})
                state["messages"].append({"role": "tool", "content": obs})
            
            turn_count += 1
            
        # 3. Final Reward Check
        reward = env.get_reward()
        state["reward"] = reward
        
        # 4. Trigger Learning on failure (if not already handled in graph)
        if reward < 1.0:
            learning_output = self.graph.invoke(state) # This will skip to learning_node
            if "global_wisdom" in learning_output:
                self.global_wisdom = learning_output["global_wisdom"]
                # Atomic save
                os.makedirs(os.path.dirname(self.wisdom_file), exist_ok=True)
                with open(self.wisdom_file + ".tmp", "w") as f:
                    json.dump(self.global_wisdom, f, indent=2)
                os.replace(self.wisdom_file + ".tmp", self.wisdom_file)

        return AgentRunResult(
            messages=state["messages"],
            reward=reward,
            info={"turn_count": turn_count}
        )
