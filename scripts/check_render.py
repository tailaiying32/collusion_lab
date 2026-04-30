import sys
sys.path.insert(0, "src")
from collusionlab.agents.memory import AgentMemory

m = AgentMemory(window_size=3)
m.update({"round": 1, "own_action": 50, "all_actions": [50, 50],
          "own_quantity": 0.471, "all_quantities": [0.471, 0.471],
          "own_reward": 76.1, "penalty_applied": False,
          "messages_received": [], "message_sent": None, "own_reasoning": None})
m.update({"round": 2, "own_action": 65, "all_actions": [65, 65],
          "own_quantity": 0.367, "all_quantities": [0.367, 0.367],
          "own_reward": 114.2, "penalty_applied": False,
          "messages_received": [], "message_sent": None, "own_reasoning": None})
print(m.to_prompt_context())
