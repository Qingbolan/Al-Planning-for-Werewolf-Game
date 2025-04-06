"""
Werewolf Game Agents Package
"""
from agents.base_agent import (
    BaseAgent, RandomAgent, HeuristicAgent, create_agent
)

# Try to import RL agent, skip if not available
try:
    from agents.rl_agent import RLAgent, create_rl_agent
    HAS_RL_AGENT = True
except ImportError:
    # Create placeholder functions and classes to prevent import errors
    HAS_RL_AGENT = False
    class RLAgent(BaseAgent):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("RL agent module not implemented")
    
    def create_rl_agent(*args, **kwargs):
        raise NotImplementedError("RL agent module not implemented")

# Export all types and factory functions
__all__ = [
    'BaseAgent',
    'RandomAgent',
    'HeuristicAgent',
    'RLAgent',
    'create_agent',
    'create_rl_agent',
    'HAS_RL_AGENT'
]
