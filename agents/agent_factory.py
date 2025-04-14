"""
Agent factory for creating different types of agents
"""
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def create_agent(agent_type: str, player_id: int) -> BaseAgent:
    """
    Create specified type of agent
    
    Args:
        agent_type: Agent type
        player_id: Player ID
        
    Returns:
        Agent instance
    """
    if agent_type == 'random':
        return RandomAgent(player_id)
    elif agent_type == 'heuristic':
        return HeuristicAgent(player_id)
    else:
        # Default return random agent
        return RandomAgent(player_id) 