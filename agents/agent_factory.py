"""
Agent factory module
创建各种类型的AI代理
"""
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def create_agent(agent_type, **kwargs):
    """创建AI代理"""
    if agent_type == "random":
        return RandomAgent(**kwargs)
    elif agent_type == "heuristic":
        return HeuristicAgent(**kwargs)
    else:
        raise ValueError(f"不支持的代理类型: {agent_type}") 