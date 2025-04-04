"""
狼人杀游戏智能体包
"""
from agents.base_agent import (
    BaseAgent, RandomAgent, HeuristicAgent, create_agent
)
from agents.rl_agent import RLAgent, create_rl_agent

# 导出所有类型和工厂函数
__all__ = [
    'BaseAgent',
    'RandomAgent',
    'HeuristicAgent',
    'RLAgent',
    'create_agent',
    'create_rl_agent'
]
